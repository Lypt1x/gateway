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
Module for fast token counting.

Uses tiktoken (OpenAI's Rust library) for approximate
token counting. The cl100k_base encoding is close to Claude tokenization.

Note: This is an approximate count, as the exact Claude tokenizer
is not public. Anthropic does not publish their tokenizer,
so tiktoken with a correction coefficient is used.

The correction coefficient CLAUDE_CORRECTION_FACTOR = 1.15 is based on
empirical observations: Claude tokenizes text approximately 15%
more than GPT-4 (cl100k_base). This is due to differences in BPE vocabularies.
"""

import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger

# Lazy loading of tiktoken to speed up import
_encoding = None

# Correction coefficient for Claude models
# Claude tokenizes text approximately 15% more than GPT-4 (cl100k_base)
# This is an empirical value based on comparison with context_usage from API
CLAUDE_CORRECTION_FACTOR = 1.15

# ------------------------------------------------------------------------------------------------
# Offline resilience for the cl100k_base encoding
# ------------------------------------------------------------------------------------------------
# tiktoken downloads cl100k_base from a CDN on first use. On an offline or locked-down host that
# fails with no retry, which used to make token counting raise. Two mitigations:
#   1. Point tiktoken at a local, writable cache directory so the encoding is downloaded at most
#      once per deployment and resolved from disk afterwards (works fully offline).
#   2. Bounded retry with exponential backoff around the load, then an APPROXIMATE character-based
#      count instead of an exception. Token counting must never take down a request.
# Note: when the real encoder loads, counts are unchanged - exactness matters for usage reporting.

# Number of load attempts (1 initial + retries) before giving up and using approximation.
TOKENIZER_LOAD_ATTEMPTS = 3

# Base delay in seconds for exponential backoff between load attempts.
TOKENIZER_RETRY_BASE_DELAY = 0.5

# Default on-disk cache location, overridable via TIKTOKEN_CACHE_DIR.
DEFAULT_TIKTOKEN_CACHE_DIR = Path(__file__).resolve().parent.parent / ".tiktoken_cache"

# Approximate characters per token used by the fallback estimator.
FALLBACK_CHARS_PER_TOKEN = 4


def _prepare_tiktoken_cache() -> Optional[str]:
    """
    Ensure tiktoken has a local cache directory to read from / write to.

    tiktoken honours the TIKTOKEN_CACHE_DIR environment variable. If the operator already set
    it, it is respected as-is. Otherwise a repository-local directory is used so a single
    successful download keeps working on subsequent offline starts.

    Returns:
        The cache directory path, or None if no usable directory could be prepared.
    """
    configured = os.environ.get("TIKTOKEN_CACHE_DIR")
    cache_dir = Path(configured) if configured else DEFAULT_TIKTOKEN_CACHE_DIR
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.warning(f"[Tokenizer] Cannot prepare tiktoken cache dir '{cache_dir}': {e}")
        return configured if configured else None

    os.environ["TIKTOKEN_CACHE_DIR"] = str(cache_dir)
    return str(cache_dir)


def reset_encoding_cache() -> None:
    """Reset the memoized encoding so it is resolved again on next use (used by tests)."""
    global _encoding
    _encoding = None


def _get_encoding():
    """
    Lazy initialization of tokenizer.
    
    Uses cl100k_base - encoding for GPT-4/ChatGPT,
    which is close enough to Claude tokenization.

    Resolves from a local cache when available, retries transient/network failures with
    exponential backoff, and returns None (triggering approximate counting) instead of raising.

    Returns:
        tiktoken.Encoding or None if the encoding could not be loaded
    """
    global _encoding
    if _encoding is None:
        try:
            import tiktoken
        except ImportError:
            logger.warning(
                "[Tokenizer] tiktoken not installed. "
                "Token counting will use fallback estimation. "
                "Install with: pip install tiktoken"
            )
            _encoding = False  # Marker that import failed
            return None

        cache_dir = _prepare_tiktoken_cache()
        if cache_dir:
            logger.debug(f"[Tokenizer] Using tiktoken cache dir: {cache_dir}")

        _encoding = False
        for attempt in range(1, TOKENIZER_LOAD_ATTEMPTS + 1):
            try:
                _encoding = tiktoken.get_encoding("cl100k_base")
                logger.debug(
                    f"[Tokenizer] Initialized tiktoken with cl100k_base encoding "
                    f"(attempt {attempt}/{TOKENIZER_LOAD_ATTEMPTS})"
                )
                break
            except Exception as e:
                if attempt < TOKENIZER_LOAD_ATTEMPTS:
                    delay = TOKENIZER_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    logger.warning(
                        f"[Tokenizer] Failed to load cl100k_base "
                        f"(attempt {attempt}/{TOKENIZER_LOAD_ATTEMPTS}): {e}. "
                        f"Retrying in {delay:.1f}s"
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"[Tokenizer] Failed to initialize tiktoken after "
                        f"{TOKENIZER_LOAD_ATTEMPTS} attempts: {e}. "
                        "Falling back to approximate token counting."
                    )
    return _encoding if _encoding else None


def approximate_token_count(text: str) -> int:
    """
    Approximate token count used when the real encoder is unavailable.

    Rough estimate of ~4 characters per token for English (~2-3 for other languages).

    Args:
        text: Text to estimate

    Returns:
        Approximate token count without Claude correction applied
    """
    if not text:
        return 0
    return len(text) // FALLBACK_CHARS_PER_TOKEN + 1


def count_tokens(text: str, apply_claude_correction: bool = True) -> int:
    """
    Counts the number of tokens in text.
    
    Args:
        text: Text to count tokens for
        apply_claude_correction: Apply correction coefficient for Claude (default True)
    
    Returns:
        Number of tokens (approximate, with Claude correction)
    """
    if not text:
        return 0
    
    encoding = _get_encoding()
    if encoding:
        try:
            base_tokens = len(encoding.encode(text))
            if apply_claude_correction:
                return int(base_tokens * CLAUDE_CORRECTION_FACTOR)
            return base_tokens
        except Exception as e:
            logger.warning(f"[Tokenizer] Error encoding text: {e}")
    
    # Fallback: approximate character-based estimate (never raises)
    base_estimate = approximate_token_count(text)
    if apply_claude_correction:
        return int(base_estimate * CLAUDE_CORRECTION_FACTOR)
    return base_estimate


def count_message_tokens(messages: List[Dict[str, Any]], apply_claude_correction: bool = True) -> int:
    """
    Counts tokens in a list of chat messages.
    
    Accounts for OpenAI/Claude message structure:
    - role: ~1 token
    - content: text tokens
    - Service tokens between messages: ~3-4 tokens
    
    Args:
        messages: List of messages in OpenAI format
        apply_claude_correction: Apply correction coefficient for Claude
    
    Returns:
        Approximate number of tokens (with Claude correction)
    """
    if not messages:
        return 0
    
    total_tokens = 0
    
    for message in messages:
        # Base tokens per message (role, delimiters)
        total_tokens += 4  # ~4 tokens for service information
        
        # Role tokens (without correction, these are short strings)
        role = message.get("role", "")
        total_tokens += count_tokens(role, apply_claude_correction=False)
        
        # Content tokens
        content = message.get("content")
        if content:
            if isinstance(content, str):
                total_tokens += count_tokens(content, apply_claude_correction=False)
            elif isinstance(content, list):
                # Support OpenAI/Anthropic multi-type content blocks
                for item in content:
                    if isinstance(item, dict):
                        item_type = item.get("type")
                        if item_type == "text":
                            total_tokens += count_tokens(item.get("text", ""), apply_claude_correction=False)
                        elif item_type in {"image_url", "image"}:
                            # Estimate image as fixed cost to avoid significant undercount
                            total_tokens += 100
                        elif item_type == "tool_use":
                            total_tokens += count_tokens(item.get("id", ""), apply_claude_correction=False)
                            total_tokens += count_tokens(item.get("name", ""), apply_claude_correction=False)
                            tool_input_str = json.dumps(item.get("input", {}), ensure_ascii=False)
                            total_tokens += count_tokens(tool_input_str, apply_claude_correction=False)
                        elif item_type == "tool_result":
                            total_tokens += count_tokens(item.get("tool_use_id", ""), apply_claude_correction=False)
                            if item.get("is_error") is not None:
                                total_tokens += count_tokens(str(item.get("is_error")), apply_claude_correction=False)

                            tool_result_content = item.get("content")
                            if isinstance(tool_result_content, str):
                                total_tokens += count_tokens(tool_result_content, apply_claude_correction=False)
                            elif isinstance(tool_result_content, list):
                                for result_block in tool_result_content:
                                    if isinstance(result_block, dict):
                                        result_type = result_block.get("type")
                                        if result_type == "text":
                                            total_tokens += count_tokens(
                                                result_block.get("text", ""),
                                                apply_claude_correction=False
                                            )
                                        elif result_type in {"image_url", "image"}:
                                            total_tokens += 100
                                    else:
                                        total_tokens += count_tokens(str(result_block), apply_claude_correction=False)
                            elif tool_result_content is not None:
                                total_tokens += count_tokens(str(tool_result_content), apply_claude_correction=False)
                        else:
                            # Unknown block fallback: estimate via JSON to avoid undercount
                            total_tokens += count_tokens(
                                json.dumps(item, ensure_ascii=False),
                                apply_claude_correction=False
                            )
                    else:
                        total_tokens += count_tokens(str(item), apply_claude_correction=False)
        
        # tool_calls tokens (if present)
        tool_calls = message.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                total_tokens += 4  # Service tokens
                func = tc.get("function", {})
                total_tokens += count_tokens(func.get("name", ""), apply_claude_correction=False)
                total_tokens += count_tokens(func.get("arguments", ""), apply_claude_correction=False)
        
        # tool_call_id tokens (for tool responses)
        if message.get("tool_call_id"):
            total_tokens += count_tokens(message["tool_call_id"], apply_claude_correction=False)
    
    # Final service tokens
    total_tokens += 3
    
    # Apply correction to total count
    if apply_claude_correction:
        return int(total_tokens * CLAUDE_CORRECTION_FACTOR)
    return total_tokens


def count_tools_tokens(tools: Optional[List[Dict[str, Any]]], apply_claude_correction: bool = True) -> int:
    """
    Counts tokens in tool definitions.
    
    Args:
        tools: List of tools in OpenAI format
        apply_claude_correction: Apply correction coefficient for Claude
    
    Returns:
        Approximate number of tokens (with Claude correction)
    """
    if not tools:
        return 0
    
    total_tokens = 0
    
    for tool in tools:
        total_tokens += 4  # Service tokens

        # Support both OpenAI standard tools and Anthropic/OpenAI flat tools
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            tool_payload = tool.get("function", {})
        else:
            tool_payload = tool

        # Name / description
        total_tokens += count_tokens(tool_payload.get("name", ""), apply_claude_correction=False)
        total_tokens += count_tokens(tool_payload.get("description", ""), apply_claude_correction=False)

        # JSON schema（Anthropic: input_schema, OpenAI: parameters）
        params = tool_payload.get("input_schema")
        if params is None:
            params = tool_payload.get("parameters")
        if params is not None:
            params_str = json.dumps(params, ensure_ascii=False)
            total_tokens += count_tokens(params_str, apply_claude_correction=False)
    
    # Apply correction to total count
    if apply_claude_correction:
        return int(total_tokens * CLAUDE_CORRECTION_FACTOR)
    return total_tokens


def count_system_tokens(system_prompt: Optional[Any], apply_claude_correction: bool = True) -> int:
    """
    Counts tokens in system prompt.

    Supports both plain string and Anthropic block list.

    Args:
        system_prompt: System prompt (str / list of blocks)
        apply_claude_correction: Apply correction coefficient for Claude

    Returns:
        Approximate number of tokens
    """
    if not system_prompt:
        return 0

    total_tokens = 0

    if isinstance(system_prompt, str):
        total_tokens += count_tokens(system_prompt, apply_claude_correction=False)
    elif isinstance(system_prompt, list):
        for block in system_prompt:
            if isinstance(block, dict):
                # Count text content, support prompt caching structure
                total_tokens += count_tokens(block.get("text", ""), apply_claude_correction=False)
                if block.get("cache_control") is not None:
                    total_tokens += count_tokens(
                        json.dumps(block.get("cache_control"), ensure_ascii=False),
                        apply_claude_correction=False
                    )
            else:
                total_tokens += count_tokens(str(block), apply_claude_correction=False)
    else:
        total_tokens += count_tokens(str(system_prompt), apply_claude_correction=False)

    if apply_claude_correction:
        return int(total_tokens * CLAUDE_CORRECTION_FACTOR)
    return total_tokens


def estimate_request_tokens(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    system_prompt: Optional[Any] = None,
    apply_claude_correction: bool = True
) -> Dict[str, int]:
    """
    Estimates total number of tokens in request.
    
    Args:
        messages: List of messages
        tools: List of tools (optional)
        system_prompt: System prompt (optional, string or Anthropic content blocks)
        apply_claude_correction: Apply correction coefficient for Claude
    
    Returns:
        Dictionary with token breakdown:
        - messages_tokens: message tokens
        - tools_tokens: tool tokens
        - system_tokens: system prompt tokens
        - total_tokens: total count
    """
    messages_tokens = count_message_tokens(messages, apply_claude_correction=apply_claude_correction)
    tools_tokens = count_tools_tokens(tools, apply_claude_correction=apply_claude_correction)
    system_tokens = count_system_tokens(system_prompt, apply_claude_correction=apply_claude_correction)
    
    return {
        "messages_tokens": messages_tokens,
        "tools_tokens": tools_tokens,
        "system_tokens": system_tokens,
        "total_tokens": messages_tokens + tools_tokens + system_tokens
    }
