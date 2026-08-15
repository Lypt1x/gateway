# -*- coding: utf-8 -*-

# Kiro Gateway
# https://github.com/jwadow/kiro-gateway
# Copyright (C) 2025 Jwadow
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
Core streaming logic for parsing Kiro API responses.

This module contains shared logic used by both OpenAI and Anthropic streaming:
- KiroEvent dataclass for unified events
- Kiro SSE stream parsing
- Full response collection
- First token timeout handling

The core layer provides a unified interface that API-specific formatters use
to convert Kiro events to their respective SSE formats.
"""

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncGenerator, Callable, Awaitable, Dict, List, Optional, Tuple

import httpx
from loguru import logger

from kiro.parsers import AwsEventStreamParser, parse_bracket_tool_calls, deduplicate_tool_calls
from kiro.config import (
    FIRST_TOKEN_TIMEOUT,
    FIRST_TOKEN_MAX_RETRIES,
    FAKE_REASONING_ENABLED,
    FAKE_REASONING_HANDLING,
)
from kiro.thinking_parser import ThinkingParser
from kiro.network_errors import is_transport_drop_error, describe_transport_drop

if TYPE_CHECKING:
    from kiro.cache import ModelInfoCache

# Import debug_logger for logging
try:
    from kiro.debug_logger import debug_logger
except ImportError:
    debug_logger = None


# ==================================================================================================
# Data Classes
# ==================================================================================================

@dataclass
class KiroEvent:
    """
    Unified event from Kiro API stream.
    
    This format is API-agnostic and can be converted to both OpenAI and Anthropic formats.
    
    Attributes:
        type: Event type (content, thinking, tool_use, usage, context_usage, error)
        content: Text content (for content events)
        thinking_content: Thinking/reasoning content (for thinking events)
        tool_use: Tool use data (for tool_use events)
        usage: Usage/metering data (for usage events)
        context_usage_percentage: Context usage percentage (for context_usage events)
        is_first_thinking_chunk: Whether this is the first thinking chunk
        is_last_thinking_chunk: Whether this is the last thinking chunk
    """
    type: str
    content: Optional[str] = None
    thinking_content: Optional[str] = None
    tool_use: Optional[Dict[str, Any]] = None
    usage: Optional[Dict[str, Any]] = None
    context_usage_percentage: Optional[float] = None
    is_first_thinking_chunk: bool = False
    is_last_thinking_chunk: bool = False


@dataclass
class StreamResult:
    """
    Result of collecting a complete stream response.
    
    Attributes:
        content: Full text content
        thinking_content: Full thinking/reasoning content
        tool_calls: List of tool calls
        usage: Usage information
        context_usage_percentage: Context usage percentage from Kiro API
    """
    content: str = ""
    thinking_content: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    usage: Optional[Dict[str, Any]] = None
    context_usage_percentage: Optional[float] = None


class FirstTokenTimeoutError(Exception):
    """Exception raised when first token timeout occurs."""
    pass


# ==================================================================================================
# Kiro Stream Parsing
# ==================================================================================================

async def parse_kiro_stream(
    response: httpx.Response,
    first_token_timeout: float = FIRST_TOKEN_TIMEOUT,
    enable_thinking_parser: bool = True
) -> AsyncGenerator[KiroEvent, None]:
    """
    Parses Kiro SSE stream and yields unified events.
    
    This is the core parsing function that converts Kiro's AWS SSE format
    into unified KiroEvent objects that can be formatted for any API.
    
    Args:
        response: HTTP response with data stream
        first_token_timeout: First token wait timeout (seconds)
        enable_thinking_parser: Whether to enable thinking block parsing
    
    Yields:
        KiroEvent objects representing stream events
    
    Raises:
        FirstTokenTimeoutError: If first token not received within timeout
    """
    parser = AwsEventStreamParser()
    first_token_received = False
    
    # Initialize thinking parser if fake reasoning is enabled
    thinking_parser: Optional[ThinkingParser] = None
    if FAKE_REASONING_ENABLED and enable_thinking_parser:
        thinking_parser = ThinkingParser(handling_mode=FAKE_REASONING_HANDLING)
        logger.debug(f"Thinking parser initialized with mode: {FAKE_REASONING_HANDLING}")
    
    try:
        # Create iterator for reading bytes
        byte_iterator = response.aiter_bytes()
        
        # Wait for first chunk with timeout
        try:
            logger.debug(f"Waiting for first token (timeout={first_token_timeout}s)...")
            first_byte_chunk = await asyncio.wait_for(
                byte_iterator.__anext__(),
                timeout=first_token_timeout
            )
            logger.debug("First token received")
        except asyncio.TimeoutError:
            logger.warning(f"[FirstTokenTimeout] Model did not respond within {first_token_timeout}s")
            raise FirstTokenTimeoutError(f"No response within {first_token_timeout} seconds")
        except StopAsyncIteration:
            # Empty response - this is normal, just finish
            logger.debug("Empty response from Kiro API")
            return
        
        # Process first chunk
        if debug_logger:
            debug_logger.log_raw_chunk(first_byte_chunk)
        
        async for event in _process_chunk(parser, first_byte_chunk, thinking_parser):
            if event.type == "content" or event.type == "thinking":
                first_token_received = True
            yield event
        
        # Continue reading remaining chunks
        async for chunk in byte_iterator:
            if debug_logger:
                debug_logger.log_raw_chunk(chunk)
            
            async for event in _process_chunk(parser, chunk, thinking_parser):
                yield event
        
        # Finalize thinking parser and yield any remaining content
        if thinking_parser:
            final_result = thinking_parser.finalize()
            
            if final_result.thinking_content:
                processed_thinking = thinking_parser.process_for_output(
                    final_result.thinking_content,
                    final_result.is_first_thinking_chunk,
                    final_result.is_last_thinking_chunk,
                )
                if processed_thinking:
                    yield KiroEvent(
                        type="thinking",
                        thinking_content=processed_thinking,
                        is_first_thinking_chunk=final_result.is_first_thinking_chunk,
                        is_last_thinking_chunk=final_result.is_last_thinking_chunk,
                    )
            
            if final_result.regular_content:
                yield KiroEvent(type="content", content=final_result.regular_content)
            
            if thinking_parser.found_thinking_block:
                logger.debug("Thinking block processing completed")
        
        # Check bracket-style tool calls in accumulated content
        all_tool_calls = parser.get_tool_calls()
        # Note: bracket tool calls are checked by the caller using full content
        
        # Yield tool calls if any
        for tc in all_tool_calls:
            yield KiroEvent(type="tool_use", tool_use=tc)
            
    except FirstTokenTimeoutError:
        raise
    except GeneratorExit:
        logger.debug("Client disconnected (GeneratorExit)")
        raise
    except asyncio.CancelledError:
        # Task cancellation must always propagate - swallowing it hangs the ASGI task
        raise
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e) if str(e) else "(empty message)"
        # Mid-stream transport drops (issue #129) are upstream behaviour, not a
        # gateway bug. Still re-raised so the formatting layer can close the client
        # stream out as a well-formed truncated turn, but logged as WARNING.
        if is_transport_drop_error(e):
            logger.warning(f"Kiro stream parsing stopped: {describe_transport_drop(e)}")
        else:
            logger.error(f"Error during stream parsing: [{error_type}] {error_msg}", exc_info=True)
        raise


async def _process_chunk(
    parser: AwsEventStreamParser,
    chunk: bytes,
    thinking_parser: Optional[ThinkingParser]
) -> AsyncGenerator[KiroEvent, None]:
    """
    Process a single chunk from Kiro stream.
    
    Args:
        parser: AWS event stream parser
        chunk: Raw bytes chunk
        thinking_parser: Optional thinking parser for fake reasoning
    
    Yields:
        KiroEvent objects
    """
    events = parser.feed(chunk)
    
    for event in events:
        if event["type"] == "content":
            content = event["data"]
            
            # Process through thinking parser if enabled
            if thinking_parser:
                parse_result = thinking_parser.feed(content)
                
                # Yield thinking content if any
                if parse_result.thinking_content:
                    processed_thinking = thinking_parser.process_for_output(
                        parse_result.thinking_content,
                        parse_result.is_first_thinking_chunk,
                        parse_result.is_last_thinking_chunk,
                    )
                    if processed_thinking:
                        yield KiroEvent(
                            type="thinking",
                            thinking_content=processed_thinking,
                            is_first_thinking_chunk=parse_result.is_first_thinking_chunk,
                            is_last_thinking_chunk=parse_result.is_last_thinking_chunk,
                        )
                
                # Yield regular content if any
                if parse_result.regular_content:
                    yield KiroEvent(type="content", content=parse_result.regular_content)
            else:
                # No thinking parser - pass through as-is
                yield KiroEvent(type="content", content=content)
        
        elif event["type"] == "usage":
            yield KiroEvent(type="usage", usage=event["data"])
        
        elif event["type"] == "context_usage":
            yield KiroEvent(type="context_usage", context_usage_percentage=event["data"])


# ==================================================================================================
# Full Response Collection
# ==================================================================================================

async def collect_stream_to_result(
    response: httpx.Response,
    first_token_timeout: float = FIRST_TOKEN_TIMEOUT,
    enable_thinking_parser: bool = True
) -> StreamResult:
    """
    Collects full response from Kiro stream.
    
    This function consumes the entire stream and returns a StreamResult
    with all accumulated data.
    
    Args:
        response: HTTP response with stream
        first_token_timeout: First token wait timeout
        enable_thinking_parser: Whether to enable thinking block parsing
    
    Returns:
        StreamResult with full content, thinking, tool calls, and usage
    """
    result = StreamResult()
    full_content_for_bracket_tools = ""
    
    async for event in parse_kiro_stream(response, first_token_timeout, enable_thinking_parser):
        if event.type == "content" and event.content:
            result.content += event.content
            full_content_for_bracket_tools += event.content
        elif event.type == "thinking" and event.thinking_content:
            result.thinking_content += event.thinking_content
            full_content_for_bracket_tools += event.thinking_content
        elif event.type == "tool_use" and event.tool_use:
            result.tool_calls.append(event.tool_use)
        elif event.type == "usage" and event.usage:
            result.usage = event.usage
        elif event.type == "context_usage" and event.context_usage_percentage is not None:
            result.context_usage_percentage = event.context_usage_percentage
    
    # Check for bracket-style tool calls in full content
    bracket_tool_calls = parse_bracket_tool_calls(full_content_for_bracket_tools)
    if bracket_tool_calls:
        result.tool_calls = deduplicate_tool_calls(result.tool_calls + bracket_tool_calls)
    
    return result


# ==================================================================================================
# Token Counting Utilities
# ==================================================================================================

def calculate_tokens_from_context_usage(
    context_usage_percentage: Optional[float],
    completion_tokens: int,
    model_cache: "ModelInfoCache",
    model: str
) -> Tuple[int, int, str, str]:
    """
    Calculate token counts from Kiro's context usage percentage.
    
    Args:
        context_usage_percentage: Context usage percentage from Kiro API
        completion_tokens: Number of completion tokens (counted via tiktoken)
        model_cache: Model cache for getting max input tokens
        model: Model name
    
    Returns:
        Tuple of (prompt_tokens, total_tokens, prompt_source, total_source)
    """
    if context_usage_percentage is not None and context_usage_percentage > 0:
        max_input_tokens = model_cache.get_max_input_tokens(model)
        total_tokens = int((context_usage_percentage / 100) * max_input_tokens)
        prompt_tokens = max(0, total_tokens - completion_tokens)
        return prompt_tokens, total_tokens, "subtraction", "API Kiro"
    
    # Fallback: no context usage data
    return 0, completion_tokens, "unknown", "tiktoken"


# ==================================================================================================
# First Token Retry Logic
# ==================================================================================================

async def stream_with_first_token_retry(
    make_request: Callable[[], Awaitable[httpx.Response]],
    stream_processor: Callable[[httpx.Response], AsyncGenerator[str, None]],
    initial_response: Optional[httpx.Response] = None,
    max_retries: int = FIRST_TOKEN_MAX_RETRIES,
    first_token_timeout: float = FIRST_TOKEN_TIMEOUT,
    on_http_error: Optional[Callable[[int, str], Exception]] = None,
    on_all_retries_failed: Optional[Callable[[int, float], Exception]] = None,
) -> AsyncGenerator[str, None]:
    """
    Generic streaming with automatic retry on first token timeout.
    
    If model doesn't respond within first_token_timeout seconds,
    request is cancelled and a new one is made. Maximum max_retries attempts.
    
    This is seamless for user - they just see a delay,
    but eventually get a response (or error after all attempts).
    
    Args:
        make_request: Function to create new HTTP request (returns httpx.Response)
        stream_processor: Function that processes response and yields SSE strings.
                         Must use parse_kiro_stream internally for timeout handling.
        initial_response: Optional pre-validated response to use on first attempt.
                         If provided, make_request is only called on retries.
                         This allows reusing an already-opened HTTP 200 response.
        max_retries: Maximum number of attempts
        first_token_timeout: First token wait timeout (seconds)
        on_http_error: Optional callback to create exception for HTTP errors.
                      Receives (status_code, error_text), returns Exception.
                      If None, raises generic Exception.
        on_all_retries_failed: Optional callback to create exception when all retries fail.
                              Receives (max_retries, timeout), returns Exception.
                              If None, raises generic Exception.
    
    Yields:
        Strings in SSE format (format depends on stream_processor)
    
    Raises:
        Exception from on_http_error or on_all_retries_failed callbacks
    
    Example:
        >>> async def make_req():
        ...     return await http_client.request_with_retry("POST", url, payload, stream=True)
        >>> async def process(response):
        ...     async for chunk in stream_kiro_to_openai(response, ...):
        ...         yield chunk
        >>> # With initial response (reuse already-validated 200 response)
        >>> response = await make_req()
        >>> async for chunk in stream_with_first_token_retry(make_req, process, initial_response=response):
        ...     print(chunk)
    """
    last_error: Optional[Exception] = None
    
    for attempt in range(max_retries):
        response: Optional[httpx.Response] = None
        try:
            # Make request
            if attempt > 0:
                logger.warning(f"Retry attempt {attempt + 1}/{max_retries} after first token timeout")
            
            # On first attempt, reuse initial_response if provided
            if attempt == 0 and initial_response is not None:
                response = initial_response
                logger.debug("Reusing initial response for first attempt")
            else:
                response = await make_request()
            
            if response.status_code != 200:
                # Error from API - close response and raise exception
                try:
                    error_content = await response.aread()
                    error_text = error_content.decode('utf-8', errors='replace')
                except Exception:
                    error_text = "Unknown error"
                
                try:
                    await response.aclose()
                except Exception:
                    pass
                
                logger.error(f"Error from Kiro API: {response.status_code} - {error_text}")
                
                if on_http_error:
                    raise on_http_error(response.status_code, error_text)
                else:
                    raise Exception(f"Upstream API error ({response.status_code}): {error_text}")
            
            # Try to stream with first token timeout
            async for chunk in stream_processor(response):
                yield chunk
            
            # Successfully completed - exit
            return
            
        except FirstTokenTimeoutError as e:
            last_error = e
            logger.warning(
                f"[FirstTokenTimeout] Attempt {attempt + 1}/{max_retries} failed - "
                f"model did not respond within {first_token_timeout}s"
            )
            
            # Close current response if open
            if response:
                try:
                    await response.aclose()
                except Exception:
                    pass
            
            # Continue to next attempt
            continue
            
        except Exception as e:
            # Other errors - no retry, propagate
            # Use positional argument to avoid loguru interpreting curly braces in error message as format placeholders
            # f-string with repr() doesn't work because loguru still sees {type} inside the string
            error_msg = str(e) if str(e) else "(empty message)"
            logger.error("Unexpected error during streaming: {}", error_msg, exc_info=True)
            if response:
                try:
                    await response.aclose()
                except Exception:
                    pass
            raise
    
    # All attempts exhausted - raise error
    logger.error(
        f"[FirstTokenTimeout] All {max_retries} attempts exhausted - "
        f"model never responded within {first_token_timeout}s per attempt"
    )
    
    if on_all_retries_failed:
        raise on_all_retries_failed(max_retries, first_token_timeout)
    else:
        raise Exception(
            f"Model did not respond within {first_token_timeout}s after {max_retries} attempts. "
            "Please try again."
        )



# ==================================================================================================
# Mid-stream continuation (issue #129, Phase 2 - gated by MIDSTREAM_RESUME, default FALSE)
# ==================================================================================================

MIDSTREAM_CONTINUATION_INSTRUCTION = (
    "Your previous reply was cut off mid-sentence by a network interruption. "
    "Continue the reply from exactly where it stopped. "
    "Do not repeat any text you already produced, do not restate or summarise it, "
    "and do not start a new introduction - just carry on."
)

# Only the tail of already-sent content is inspected for overlap. Long enough to
# catch a repeated paragraph, short enough to stay cheap.
_OVERLAP_WINDOW = 2000


def trim_continuation_overlap(sent_content: str, continuation: str) -> str:
    """
    Removes the longest overlap between the tail of already-sent content and the
    head of a continuation.

    Prompt-level continuation frequently re-emits some of the text the client has
    already received. Text already sent cannot be un-sent, so the only safe repair
    is to drop the duplicated prefix from the continuation.

    Args:
        sent_content: Everything already delivered to the client
        continuation: Freshly received continuation text

    Returns:
        The continuation with its duplicated leading portion removed

    Example:
        >>> trim_continuation_overlap("The quick brown", " brown fox jumps")
        ' fox jumps'
    """
    if not sent_content or not continuation:
        return continuation

    tail = sent_content[-_OVERLAP_WINDOW:]
    max_overlap = min(len(tail), len(continuation))

    # Longest suffix of `tail` that is also a prefix of `continuation`
    for size in range(max_overlap, 0, -1):
        if tail.endswith(continuation[:size]):
            trimmed = continuation[size:]
            logger.debug(f"Midstream continuation: trimmed {size} overlapping chars")
            return trimmed

    return continuation


def _build_continuation_messages(
    request_messages: Optional[List[Any]],
    partial_content: str,
) -> List[Dict[str, Any]]:
    """
    Builds the continuation message list: original turns + the partial assistant
    reply + an explicit "continue where you stopped" user instruction.

    Args:
        request_messages: Original request messages (Anthropic-shaped dicts)
        partial_content: Text already streamed to the client

    Returns:
        Message list for the continuation request
    """
    messages: List[Dict[str, Any]] = []

    for msg in (request_messages or []):
        if isinstance(msg, dict):
            role = msg.get("role")
            content = msg.get("content")
        else:
            role = getattr(msg, "role", None)
            content = getattr(msg, "content", None)
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "assistant", "content": partial_content})
    messages.append({"role": "user", "content": MIDSTREAM_CONTINUATION_INSTRUCTION})

    return messages


async def stream_midstream_continuation(
    model: str,
    request_messages: Optional[List[Any]],
    partial_content: str,
    auth_manager: Any,
) -> AsyncGenerator[str, None]:
    """
    Attempts ONE continuation round after a mid-stream transport drop and yields
    content-only text fragments, already trimmed of overlap with `partial_content`.

    Structurally modelled on the web-search synthesis continuation: build request ->
    convert -> request_with_retry(stream=True) -> parse_kiro_stream, yielding content
    only. Best-effort: any failure yields nothing and returns, so the caller falls
    through to the Phase 1 graceful close-out. GeneratorExit and CancelledError are
    always propagated.

    Budget (see kiro.config):
      - 1 continuation round
      - MIDSTREAM_RESUME_MAX_ATTEMPTS upstream attempts
      - MIDSTREAM_RESUME_BACKOFFS between attempts
      - MIDSTREAM_RESUME_WALL_CLOCK_CAP hard wall-clock cap

    Args:
        model: Model name for the continuation request
        request_messages: Original request messages
        partial_content: Text already delivered to the client
        auth_manager: KiroAuthManager instance (may be None)

    Yields:
        Text fragments to append to the already-open text block.
    """
    import time as _time

    from kiro.config import (
        MIDSTREAM_RESUME_MAX_ATTEMPTS,
        MIDSTREAM_RESUME_BACKOFFS,
        MIDSTREAM_RESUME_WALL_CLOCK_CAP,
        MIDSTREAM_RESUME_MAX_TOKENS,
    )

    if auth_manager is None or not partial_content:
        logger.debug("Midstream continuation skipped: no auth_manager or no partial content")
        return

    try:
        from kiro.models_anthropic import AnthropicMessagesRequest
        from kiro.converters_anthropic import anthropic_to_kiro
        from kiro.http_client import KiroHttpClient
        from kiro.utils import generate_conversation_id
    except Exception as e:  # pragma: no cover - import guard
        logger.warning(f"Midstream continuation unavailable (import failed): {e}")
        return

    deadline = _time.monotonic() + MIDSTREAM_RESUME_WALL_CLOCK_CAP
    sent_so_far = partial_content
    emitted_any = False

    for attempt in range(MIDSTREAM_RESUME_MAX_ATTEMPTS):
        if _time.monotonic() >= deadline:
            logger.warning("Midstream continuation abandoned: wall-clock cap reached")
            return

        if attempt > 0:
            backoff = MIDSTREAM_RESUME_BACKOFFS[min(attempt - 1, len(MIDSTREAM_RESUME_BACKOFFS) - 1)]
            await asyncio.sleep(backoff)
            if _time.monotonic() >= deadline:
                logger.warning("Midstream continuation abandoned: wall-clock cap reached")
                return

        http_client = None
        try:
            continuation_request = AnthropicMessagesRequest(
                model=model,
                max_tokens=MIDSTREAM_RESUME_MAX_TOKENS,
                stream=True,
                messages=_build_continuation_messages(request_messages, partial_content),
            )

            payload = anthropic_to_kiro(
                continuation_request,
                generate_conversation_id(),
                getattr(auth_manager, "profile_arn", "") or ""
            )

            url = f"{auth_manager.api_host}/generateAssistantResponse"
            logger.info(
                f"Midstream continuation attempt {attempt + 1}/{MIDSTREAM_RESUME_MAX_ATTEMPTS} via {url}"
            )

            http_client = KiroHttpClient(auth_manager, shared_client=None)
            response = await http_client.request_with_retry("POST", url, payload, stream=True)

            if getattr(response, "status_code", None) != 200:
                logger.warning(
                    f"Midstream continuation failed: upstream status "
                    f"{getattr(response, 'status_code', 'unknown')}"
                )
                continue

            async for event in parse_kiro_stream(response):
                if getattr(event, "type", None) != "content":
                    continue
                content = getattr(event, "content", None)
                if not content:
                    continue

                # Trim overlap against everything the client has already seen.
                # Only the first fragments can overlap; once we have emitted
                # something the continuation is by definition new text.
                if not emitted_any:
                    content = trim_continuation_overlap(sent_so_far, content)
                    if not content:
                        continue

                emitted_any = True
                sent_so_far += content
                yield content

            if emitted_any:
                return

            logger.warning("Midstream continuation produced no new content")
            return

        except GeneratorExit:
            raise
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(
                f"Midstream continuation attempt {attempt + 1} failed, degrading: "
                f"[{type(e).__name__}] {e}"
            )
        finally:
            if http_client is not None:
                try:
                    await http_client.close()
                except Exception as e:
                    logger.debug(f"Midstream continuation: error closing HTTP client: {e}")

    logger.warning("Midstream continuation exhausted its budget; falling back to close-out")
