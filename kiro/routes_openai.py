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
FastAPI routes for Kiro Gateway.

Contains all API endpoints:
- / and /health: Health check
- /v1/models: Models list
- /v1/chat/completions: Chat completions
"""

import json
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, Response, Security
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from loguru import logger

from kiro.config import (
    PROXY_API_KEY,
    PROXY_API_KEYS,
    configured_proxy_api_key_count,
    verify_proxy_bearer_token,
    APP_VERSION,
    PROFILE_ARN,
    MODEL_ALIASES,
)
from kiro.models_openai import (
    OpenAIModel,
    ModelList,
    ChatCompletionRequest,
)
from kiro.auth import KiroAuthManager, AuthType
from kiro.cache import ModelInfoCache
from kiro.model_resolver import ModelResolver
from kiro.converters_openai import build_kiro_payload
from kiro.streaming_openai import stream_kiro_to_openai, collect_stream_response, stream_with_first_token_retry
from kiro.http_client import KiroHttpClient
from kiro.utils import generate_conversation_id
from kiro.network_errors import log_streaming_failure
from kiro.config import WEB_SEARCH_ENABLED
from kiro.mcp_tools import handle_native_web_search

# Import debug_logger
try:
    from kiro.debug_logger import debug_logger
except ImportError:
    debug_logger = None


# --- Security scheme ---
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


def build_sse_error_chunk(error: BaseException) -> str:
    """
    Builds an OpenAI-shaped SSE error chunk for a failure that happened mid-stream.

    Once StreamingResponse has emitted its headers, the HTTP status code can no longer
    be changed and re-raising from the generator makes Starlette abort the connection
    with "Caught handled exception, but response already started". The client then sees
    only `data: [DONE]` with no content and no reason. Reporting the failure in-band
    keeps the reason visible, mirroring what /v1/messages does with `event: error`.

    Args:
        error: The exception that ended the stream. For HTTPException the real
               status_code and detail are preserved (e.g. 504 first-token timeout);
               anything else is reported as 500.

    Returns:
        A single SSE chunk: `data: {"error": {...}}\\n\\n`
    """
    status_code = getattr(error, "status_code", 500)
    detail = getattr(error, "detail", None)
    message = str(detail) if detail else (str(error) or type(error).__name__)
    return "data: " + json.dumps({
        "error": {
            "message": message,
            "type": "kiro_api_error",
            "code": status_code,
        }
    }) + "\n\n"


def log_streaming_error(route_label: str, error: BaseException) -> None:
    """
    Logs a mid-stream failure at the accurate status.

    An HTTPException carries the status the route already decided on (e.g. 504 for a
    first-token timeout), so logging it as "HTTP 500" wrongly implies a gateway bug.
    Everything else is delegated to the shared log_streaming_failure() classifier.

    Args:
        route_label: e.g. "POST /v1/chat/completions (streaming)"
        error: The exception that ended the stream
    """
    if isinstance(error, HTTPException):
        detail = str(error.detail) if error.detail else "(no detail)"
        logger.warning(f"HTTP {error.status_code} - {route_label} - {detail[:160]}")
        return
    log_streaming_failure(route_label, error)


async def verify_api_key(auth_header: str = Security(api_key_header)) -> bool:
    """
    Verify API key in Authorization header.
    
    Expects format: "Bearer {PROXY_API_KEY}"
    
    Args:
        auth_header: Authorization header value
    
    Returns:
        True if key is valid
    
    Raises:
        HTTPException: 401 if key is invalid or missing
    """
    if not verify_proxy_bearer_token(auth_header):
        logger.warning(
            f"Access attempt with invalid API key. ({configured_proxy_api_key_count()} key(s) configured)"
        )
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return True


# --- Router ---
router = APIRouter()


@router.get("/")
async def root():
    """
    Health check endpoint.
    
    Returns:
        Status and application version
    """
    return {
        "status": "ok",
        "message": "Kiro Gateway is running",
        "version": APP_VERSION
    }


@router.get("/health")
async def health():
    """
    Detailed health check.
    
    Returns:
        Status, timestamp and version
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": APP_VERSION
    }

def collect_visible_models(request: Request):
    """
    Resolve the set of models this account may use, plus a metadata lookup for them.

    This is the single source of the "visible model set": both /v1/models and
    /integrations/opencode.json call it, so the two endpoints can never disagree.
    HIDDEN_MODELS / HIDDEN_FROM_LIST filtering (which keeps "auto" out) already happens
    inside the resolver / account manager that this helper delegates to.

    Args:
        request: FastAPI Request for accessing app.state

    Returns:
        (model_ids, metadata_for) where metadata_for(model_id) returns the additive
        upstream metadata dict for that model, or {} when nothing is known.
    """
    if request.app.state.account_system:
        # Account system: collect models from all initialized accounts
        available_model_ids = request.app.state.account_manager.get_all_available_models()
        caches = [
            cache for cache in (
                getattr(account, "model_cache", None)
                for account in request.app.state.account_manager.iter_initialized_accounts()
            ) if cache is not None
        ]
    else:
        # Legacy: use resolver from first account
        account = request.app.state.account_manager.get_first_account()
        available_model_ids = account.model_resolver.get_available_models()
        caches = [account.model_cache] if getattr(account, "model_cache", None) else []

    def metadata_for(model_id: str) -> dict:
        """
        First cache that knows the model wins; a miss simply yields no extra fields.

        A public alias (e.g. "auto-kiro") is only an entry in MODEL_ALIASES, never a
        cache entry, so a direct lookup always misses. Such an id therefore falls back
        to the metadata of the model it points at, single-hop, so the alias reports the
        real upstream limits under its own public id. Nothing is fabricated: when the
        target is unknown or metadata-free the result is still an empty dict.
        """
        candidates = [model_id]
        alias_target = MODEL_ALIASES.get(model_id)
        if isinstance(alias_target, str) and alias_target and alias_target != model_id:
            candidates.append(alias_target)

        for candidate in candidates:
            for cache in caches:
                try:
                    metadata = cache.get_public_metadata(candidate)
                except Exception as e:  # never let introspection break the endpoint
                    logger.warning(f"Failed to read metadata for model {candidate}: {e}")
                    continue
                if metadata:
                    return metadata
        return {}

    return available_model_ids, metadata_for


@router.get("/v1/models", dependencies=[Depends(verify_api_key)])
async def get_models(request: Request):
    """
    Return list of available models.
    
    Models are loaded at startup (blocking) and cached.
    This endpoint returns the cached list.
    
    The envelope is unchanged ({"object": "list", "data": [...]}) and every entry keeps
    its original id/object/created/owned_by/description fields. Upstream metadata (token
    limits, input types, prompt caching, rate) is added ADDITIVELY per entry and omitted
    whenever it is unknown — e.g. for static FALLBACK_MODELS entries, which then render
    exactly as before. Because entries carry optional extra keys, the response is
    serialized from validated OpenAIModel base fields plus the metadata overlay instead
    of through a fixed response_model.
    
    Args:
        request: FastAPI Request for accessing app.state
    
    Returns:
        ModelList-shaped dict with available models in consistent format (with dots)
    """
    logger.info("Request to /v1/models")
    
    available_model_ids, metadata_for = collect_visible_models(request)
    
    # Build OpenAI-compatible model list
    openai_models = [
        {
            **OpenAIModel(
                id=model_id,
                owned_by="anthropic",
                description="Claude model via Kiro API"
            ).model_dump(),
            **metadata_for(model_id),
        }
        for model_id in available_model_ids
    ]
    
    return {"object": "list", "data": openai_models}


# OpenCode does not auto-discover models for custom providers (upstream OpenCode issue
# #6231 is still open), so every model must be listed explicitly in opencode.json.
OPENCODE_API_KEY_PLACEHOLDER = "{env:KIRO_GATEWAY_KEY}"
OPENCODE_NPM_PACKAGE = "@ai-sdk/openai-compatible"
OPENCODE_SCHEMA_URL = "https://opencode.ai/config.json"
# Field the gateway uses for OpenAI-style reasoning text, surfaced only when ?reasoning=true.
OPENCODE_REASONING_FIELD = "reasoning_content"


def _derive_base_url(request: Request) -> str:
    """
    Derive the gateway's OpenAI-compatible base URL from the incoming request.

    Using the request's own scheme/host means a caller reaching the gateway through
    Docker, a hostname or a reverse proxy gets a value that actually works for them,
    instead of a hardcoded localhost.

    Returns:
        e.g. "http://localhost:8000/v1" (always ends in "/v1")
    """
    return str(request.base_url).rstrip("/") + "/v1"


def build_opencode_config(
    model_ids,
    metadata_for,
    provider_id: str,
    base_url: str,
    api_key: str,
    reasoning: bool,
) -> dict:
    """
    Build an OpenCode provider config document.

    Only fields documented at https://opencode.ai/docs/en/providers/ are emitted. The
    upstream catalog exposes more metadata (input types, prompt caching, rate), but
    OpenCode has no schema slot for it, so it is deliberately dropped rather than
    invented as non-standard keys.

    `limit` is emitted only when the upstream catalog actually reports the numbers; a
    model with unknown limits gets no `limit` key at all rather than a guessed one.

    Args:
        model_ids: Visible model ids (already hidden-model filtered)
        metadata_for: Callable(model_id) -> upstream metadata dict
        provider_id: Provider key under "provider"
        base_url: Value for options.baseURL
        api_key: Value for options.apiKey — a placeholder string, never a real secret
        reasoning: Whether to emit the (unverified) reasoning/interleaved pair

    Returns:
        A JSON-serializable OpenCode config dict
    """
    models: dict = {}
    for model_id in model_ids:
        metadata = metadata_for(model_id) or {}
        entry: dict = {"name": metadata.get("display_name") or model_id}

        limit = {}
        max_input = metadata.get("max_input_tokens")
        max_output = metadata.get("max_output_tokens")
        if isinstance(max_input, int) and not isinstance(max_input, bool):
            limit["context"] = max_input
        if isinstance(max_output, int) and not isinstance(max_output, bool):
            limit["output"] = max_output
        if limit:
            entry["limit"] = limit

        if reasoning:
            entry["reasoning"] = True
            entry["interleaved"] = {"field": OPENCODE_REASONING_FIELD}

        models[model_id] = entry

    return {
        "$schema": OPENCODE_SCHEMA_URL,
        "provider": {
            provider_id: {
                "npm": OPENCODE_NPM_PACKAGE,
                "name": "Kiro Gateway",
                "options": {
                    "baseURL": base_url,
                    "apiKey": api_key,
                },
                "models": models,
            }
        },
    }


@router.get("/integrations/opencode.json", dependencies=[Depends(verify_api_key)])
async def get_opencode_config(
    request: Request,
    provider: str = "kiro",
    base_url: str = None,
    api_key: str = None,
    reasoning: bool = False,
):
    """
    Return a ready-to-paste OpenCode provider config for THIS account's model catalog.

    Why generated instead of shipped as a static file: /ListAvailableModels is
    authenticated and per-account, so a free-tier account genuinely sees fewer models
    than a paid one. A checked-in opencode.json would misrepresent entitlements. The
    model set comes from the same resolver as /v1/models (see collect_visible_models),
    so the two endpoints can never disagree and hidden models such as "auto" stay out.

    The document is returned as JSON only — nothing is ever written to disk. Users merge
    it into their own opencode.json, which also holds their agents/plugins/MCP config.

    options.apiKey defaults to the literal "{env:KIRO_GATEWAY_KEY}" placeholder, which
    OpenCode resolves itself. The server's configured PROXY_API_KEY is never read here,
    so it cannot leak into the document. A caller-supplied ?api_key= is echoed verbatim.

    Args:
        request: FastAPI Request (also used to derive baseURL)
        provider: Provider id key (default "kiro")
        base_url: Override for options.baseURL (default: derived from this request)
        api_key: Override for the options.apiKey placeholder text
        reasoning: Emit "reasoning": true + "interleaved": {"field": "reasoning_content"}.
                   Defaults to FALSE: the gateway does emit OpenAI reasoning_content and
                   the docs' Poolside example implies this pair is what surfaces it, but
                   that has NOT been verified against a real OpenCode session, so it is
                   opt-in rather than enabled for every model at once.

    Returns:
        Pretty-printed OpenCode config JSON
    """
    logger.info("Request to /integrations/opencode.json")

    model_ids, metadata_for = collect_visible_models(request)

    config = build_opencode_config(
        model_ids=model_ids,
        metadata_for=metadata_for,
        provider_id=provider or "kiro",
        base_url=base_url or _derive_base_url(request),
        api_key=api_key if api_key else OPENCODE_API_KEY_PLACEHOLDER,
        reasoning=reasoning,
    )

    # Pretty-printed so the response is directly pasteable into opencode.json.
    return Response(
        content=json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        media_type="application/json",
    )


@router.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
async def chat_completions(request: Request, request_data: ChatCompletionRequest):
    """
    Chat completions endpoint - compatible with OpenAI API.
    
    Accepts requests in OpenAI format and translates them to Kiro API.
    Supports streaming and non-streaming modes.
    
    Args:
        request: FastAPI Request for accessing app.state
        request_data: Request in OpenAI ChatCompletionRequest format
    
    Returns:
        StreamingResponse for streaming mode
        JSONResponse for non-streaming mode
    
    Raises:
        HTTPException: On validation or API errors
    """
    logger.info(f"Request to /v1/chat/completions (model={request_data.model}, stream={request_data.stream})")
    
    # Note: prepare_new_request() and log_request_body() are now called by DebugLoggerMiddleware
    # This ensures debug logging works even for requests that fail Pydantic validation (422 errors)
    
    # Check for truncation recovery opportunities
    from kiro.truncation_state import get_tool_truncation, get_content_truncation
    from kiro.truncation_recovery import generate_truncation_tool_result, generate_truncation_user_message
    from kiro.models_openai import ChatMessage
    
    modified_messages = []
    tool_results_modified = 0
    content_notices_added = 0
    
    for msg in request_data.messages:
        # Check if this is a tool_result for a truncated tool call
        if msg.role == "tool" and msg.tool_call_id:
            truncation_info = get_tool_truncation(msg.tool_call_id)
            if truncation_info:
                # Modify tool_result content to include truncation notice
                synthetic = generate_truncation_tool_result(
                    tool_name=truncation_info.tool_name,
                    tool_use_id=msg.tool_call_id,
                    truncation_info=truncation_info.truncation_info
                )
                # Prepend truncation notice to original content
                modified_content = f"{synthetic['content']}\n\n---\n\nOriginal tool result:\n{msg.content}"
                
                # Create NEW ChatMessage object (Pydantic immutability)
                modified_msg = msg.model_copy(update={"content": modified_content})
                modified_messages.append(modified_msg)
                tool_results_modified += 1
                logger.debug(f"Modified tool_result for {msg.tool_call_id} to include truncation notice")
                continue  # Skip normal append since we already added modified version
        
        # Check if this is an assistant message with truncated content
        if msg.role == "assistant" and msg.content and isinstance(msg.content, str):
            truncation_info = get_content_truncation(msg.content)
            if truncation_info:
                # Add this message first
                modified_messages.append(msg)
                # Then add synthetic user message about truncation
                synthetic_user_msg = ChatMessage(
                    role="user",
                    content=generate_truncation_user_message()
                )
                modified_messages.append(synthetic_user_msg)
                content_notices_added += 1
                logger.debug(f"Added truncation notice after assistant message (hash: {truncation_info.message_hash})")
                continue  # Skip normal append since we already added it
        
        modified_messages.append(msg)
    
    if tool_results_modified > 0 or content_notices_added > 0:
        request_data.messages = modified_messages
        logger.info(f"Truncation recovery: modified {tool_results_modified} tool_result(s), added {content_notices_added} content notice(s)")
    
    # ==============================================================================
    # WebSearch Support - Path B: Auto-Injection (MCP Tool Emulation)
    # ==============================================================================
    
    # Auto-inject web_search tool if enabled (Path B - MCP emulation)
    if WEB_SEARCH_ENABLED:
        if request_data.tools is None:
            request_data.tools = []
        
        # Check if web_search already exists
        has_ws = any(
            getattr(tool, "type", None) == "function" and
            getattr(getattr(tool, "function", None), "name", None) == "web_search"
            for tool in request_data.tools
        )
        
        if not has_ws:
            from kiro.models_openai import Tool, ToolFunction
            web_search_tool = Tool(
                type="function",
                function=ToolFunction(
                    name="web_search",
                    description="Search the web for current information. Use when you need up-to-date data from the internet.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            }
                        },
                        "required": ["query"]
                    }
                )
            )
            request_data.tools.append(web_search_tool)
            logger.debug("Auto-injected web_search tool for MCP emulation (Path B)")
    
    # ==============================================================================
    # Account System: Account System Failover or Legacy Mode
    # ==============================================================================
    
    if request.app.state.account_system:
        # ==============================================================================
        # ACCOUNT SYSTEM ENABLED: Failover Loop
        # ==============================================================================
        from kiro.account_errors import classify_error, ErrorType
        
        account_manager = request.app.state.account_manager
        all_accounts = list(account_manager._accounts.keys())
        MAX_ATTEMPTS = len(all_accounts) * 2  # Full circle with margin
        
        last_error_message = None
        last_error_status = None
        tried_accounts = set()  # Track tried accounts in current failover loop
        
        for attempt in range(MAX_ATTEMPTS):
            # Get next available account (excluding already tried)
            account = await account_manager.get_next_account(
                request_data.model,
                exclude_accounts=tried_accounts
            )
            
            if account is None:
                # All accounts unavailable
                if len(all_accounts) == 1:
                    # Single account - return original error with original status code
                    raise HTTPException(
                        status_code=last_error_status or 503,
                        detail=last_error_message or "Account unavailable"
                    )
                else:
                    # Multiple accounts - generic error with context
                    detail = "No available accounts for this model."
                    if last_error_message:
                        detail += f" Error from last account: {last_error_message}"
                    raise HTTPException(status_code=503, detail=detail)
            
            # Mark account as tried in current failover loop
            tried_accounts.add(account.id)
            
            # Use objects from account
            auth_manager = account.auth_manager
            model_cache = account.model_cache
            model_resolver = account.model_resolver
            
            # Generate conversation ID
            conversation_id = generate_conversation_id()
            
            # Build payload for Kiro
            # profileArn is required by runtime.kiro.dev for all auth types
            profile_arn_for_payload = auth_manager.profile_arn or PROFILE_ARN or ""
            
            try:
                kiro_payload = build_kiro_payload(
                    request_data,
                    conversation_id,
                    profile_arn_for_payload
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            
            # Log Kiro payload
            try:
                kiro_request_body = json.dumps(kiro_payload, ensure_ascii=False, indent=2).encode('utf-8')
                if debug_logger:
                    debug_logger.log_kiro_request_body(kiro_request_body)
            except Exception as e:
                logger.warning(f"Failed to log Kiro request: {e}")
            
            # Create HTTP client
            url = f"{auth_manager.api_host}/generateAssistantResponse"
            logger.debug(f"Kiro API URL: {url} (account: {account.id})")
            
            if request_data.stream:
                http_client = KiroHttpClient(auth_manager, shared_client=None)
            else:
                shared_client = request.app.state.http_client
                http_client = KiroHttpClient(auth_manager, shared_client=shared_client)
            
            try:
                # Make request to Kiro API
                response = await http_client.request_with_retry(
                    "POST",
                    url,
                    kiro_payload,
                    stream=True
                )
                
                if response.status_code == 200:
                    # SUCCESS - report and return
                    await account_manager.report_success(account.id, request_data.model)
                    
                    # Prepare data for token counting
                    messages_for_tokenizer = [msg.model_dump() for msg in request_data.messages]
                    tools_for_tokenizer = [tool.model_dump() for tool in request_data.tools] if request_data.tools else None
                    
                    if request_data.stream:
                        # Streaming mode
                        async def stream_wrapper():
                            streaming_error = None
                            client_disconnected = False
                            try:
                                async def make_retry_request():
                                    return await http_client.request_with_retry(
                                        "POST", url, kiro_payload, stream=True
                                    )
                                
                                async for chunk in stream_with_first_token_retry(
                                    make_request=make_retry_request,
                                    client=http_client.client,
                                    model=request_data.model,
                                    model_cache=model_cache,
                                    auth_manager=auth_manager,
                                    initial_response=response,
                                    request_messages=messages_for_tokenizer,
                                    request_tools=tools_for_tokenizer
                                ):
                                    yield chunk
                            except GeneratorExit:
                                client_disconnected = True
                                logger.debug("Client disconnected during streaming (GeneratorExit in routes)")
                            except Exception as e:
                                streaming_error = e
                                # Headers are already sent - report in-band and end
                                # the generator normally (see build_sse_error_chunk)
                                try:
                                    yield build_sse_error_chunk(e)
                                    yield "data: [DONE]\n\n"
                                except Exception:
                                    pass  # Client already disconnected
                            finally:
                                await http_client.close()
                                if streaming_error:
                                    log_streaming_error("POST /v1/chat/completions (streaming)", streaming_error)
                                elif client_disconnected:
                                    logger.info(f"HTTP 200 - POST /v1/chat/completions (streaming) - client disconnected")
                                else:
                                    logger.info(f"HTTP 200 - POST /v1/chat/completions (streaming) - completed")
                                if debug_logger:
                                    if streaming_error:
                                        debug_logger.flush_on_error(getattr(streaming_error, "status_code", 500), str(streaming_error))
                                    else:
                                        debug_logger.discard_buffers()
                        
                        return StreamingResponse(stream_wrapper(), media_type="text/event-stream")
                    
                    else:
                        # Non-streaming mode
                        openai_response = await collect_stream_response(
                            http_client.client,
                            response,
                            request_data.model,
                            model_cache,
                            auth_manager,
                            request_messages=messages_for_tokenizer,
                            request_tools=tools_for_tokenizer
                        )
                        
                        await http_client.close()
                        logger.info(f"HTTP 200 - POST /v1/chat/completions (non-streaming) - completed")
                        
                        if debug_logger:
                            debug_logger.discard_buffers()
                        
                        return JSONResponse(content=openai_response)
                
                else:
                    # ERROR - classify and decide
                    try:
                        error_content = await response.aread()
                    except Exception:
                        error_content = b"Unknown error"
                    
                    await http_client.close()
                    error_text = error_content.decode('utf-8', errors='replace')
                    
                    # Extract error reason and save for final return
                    error_reason = None
                    try:
                        error_json = json.loads(error_text)
                        from kiro.kiro_errors import enhance_kiro_error
                        error_info = enhance_kiro_error(error_json)
                        error_reason = error_info.reason
                        last_error_message = error_info.user_message
                        last_error_status = response.status_code
                        logger.debug(f"Original Kiro error: {error_info.original_message} (reason: {error_info.reason})")
                    except (json.JSONDecodeError, KeyError):
                        last_error_message = error_text
                        last_error_status = response.status_code
                    
                    # Classify error
                    error_type = classify_error(response.status_code, error_reason)
                    
                    if error_type == ErrorType.FATAL:
                        # FATAL - return to client immediately
                        await account_manager.report_failure(
                            account.id, request_data.model, error_type,
                            response.status_code, error_reason
                        )
                        
                        logger.warning(f"HTTP {response.status_code} - POST /v1/chat/completions - {last_error_message[:100]}")
                        
                        if debug_logger:
                            debug_logger.flush_on_error(response.status_code, last_error_message)
                        
                        return JSONResponse(
                            status_code=response.status_code,
                            content={
                                "error": {
                                    "message": last_error_message,
                                    "type": "kiro_api_error",
                                    "code": response.status_code
                                }
                            }
                        )
                    
                    else:  # ErrorType.RECOVERABLE
                        # RECOVERABLE - try next account
                        await account_manager.report_failure(
                            account.id, request_data.model, error_type,
                            response.status_code, error_reason
                        )
                        
                        # Single account - no point in failover, break immediately
                        if len(all_accounts) == 1:
                            break
                        
                        continue  # Next iteration
            
            except HTTPException as e:
                await http_client.close()
                
                # Network errors (502/504 from request_with_retry) = RECOVERABLE
                # These are thrown ONLY for network-level issues (timeouts, connection errors)
                # NOT for HTTP-level errors (which are returned as response objects)
                if e.status_code in (502, 504):
                    # Network error → try next account
                    await account_manager.report_failure(
                        account.id, request_data.model, ErrorType.RECOVERABLE,
                        e.status_code, None
                    )
                    
                    last_error_message = str(e.detail)
                    last_error_status = e.status_code
                    
                    # Single account - no point in failover, break immediately
                    if len(all_accounts) == 1:
                        break
                    
                    logger.warning(f"Network error on account {account.id}, trying next account")
                    continue  # Try next account
                
                # All other HTTPException (400, 500, etc.) = application errors
                # These come from build_kiro_payload() or other places → re-raise immediately
                logger.error(f"HTTP {e.status_code} - POST /v1/chat/completions - {e.detail}")
                if debug_logger:
                    debug_logger.flush_on_error(e.status_code, str(e.detail))
                raise
            except Exception as e:
                await http_client.close()
                logger.error(f"Internal error: {e}", exc_info=True)
                logger.error(f"HTTP 500 - POST /v1/chat/completions - {str(e)[:100]}")
                if debug_logger:
                    debug_logger.flush_on_error(500, str(e))
                raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
        
        # All attempts exhausted
        if len(all_accounts) == 1:
            # Single account - return its original error
            # last_error_status and last_error_message are guaranteed to be set
            raise HTTPException(
                status_code=last_error_status,
                detail=last_error_message
            )
        else:
            # Multiple accounts - generic error with context
            detail = "All accounts failed after full circle."
            if last_error_message:
                detail += f" Error from last account: {last_error_message}"
            raise HTTPException(status_code=503, detail=detail)
    
    else:
        # ==============================================================================
        # LEGACY MODE: Single Account (no failover)
        # ==============================================================================
        account = request.app.state.account_manager.get_first_account()
        if not account.auth_manager:
            logger.error("No initialized accounts available (legacy mode)")
            raise HTTPException(503, "No initialized accounts available")
        auth_manager = account.auth_manager
        model_cache = account.model_cache
        model_resolver = account.model_resolver
    
    # Generate conversation ID for Kiro API (random UUID, not used for tracking)
    conversation_id = generate_conversation_id()
    
    # Build payload for Kiro
    # profileArn is required by runtime.kiro.dev for all auth types
    profile_arn_for_payload = auth_manager.profile_arn or PROFILE_ARN or ""
    
    try:
        kiro_payload = build_kiro_payload(
            request_data,
            conversation_id,
            profile_arn_for_payload
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Log Kiro payload
    try:
        kiro_request_body = json.dumps(kiro_payload, ensure_ascii=False, indent=2).encode('utf-8')
        if debug_logger:
            debug_logger.log_kiro_request_body(kiro_request_body)
    except Exception as e:
        logger.warning(f"Failed to log Kiro request: {e}")
    
    # Create HTTP client with retry logic
    # For streaming: use per-request client to avoid CLOSE_WAIT leak on VPN disconnect (issue #54)
    # For non-streaming: use shared client for connection pooling
    url = f"{auth_manager.api_host}/generateAssistantResponse"
    logger.debug(f"Kiro API URL: {url}")
    
    if request_data.stream:
        # Streaming mode: per-request client prevents orphaned connections
        # when network interface changes (VPN disconnect/reconnect)
        http_client = KiroHttpClient(auth_manager, shared_client=None)
    else:
        # Non-streaming mode: shared client for efficient connection reuse
        shared_client = request.app.state.http_client
        http_client = KiroHttpClient(auth_manager, shared_client=shared_client)
    try:
        # Make request to Kiro API (for both streaming and non-streaming modes)
        # Important: we wait for Kiro response BEFORE returning StreamingResponse,
        # so that 200 OK means Kiro accepted the request and started responding
        response = await http_client.request_with_retry(
            "POST",
            url,
            kiro_payload,
            stream=True
        )
        
        if response.status_code != 200:
            try:
                error_content = await response.aread()
            except Exception:
                error_content = b"Unknown error"
            
            await http_client.close()
            error_text = error_content.decode('utf-8', errors='replace')
            
            # Try to parse JSON response from Kiro to extract error message
            error_message = error_text
            try:
                error_json = json.loads(error_text)
                # Enhance Kiro API errors with user-friendly messages
                from kiro.kiro_errors import enhance_kiro_error
                error_info = enhance_kiro_error(error_json)
                error_message = error_info.user_message
                # Log original error for debugging
                logger.debug(f"Original Kiro error: {error_info.original_message} (reason: {error_info.reason})")
            except (json.JSONDecodeError, KeyError):
                pass
            
            # Log access log for error (before flush, so it gets into app_logs)
            logger.warning(
                f"HTTP {response.status_code} - POST /v1/chat/completions - {error_message[:100]}"
            )
            
            # Flush debug logs on error ("errors" mode)
            if debug_logger:
                debug_logger.flush_on_error(response.status_code, error_message)
            
            # Return error in OpenAI API format
            return JSONResponse(
                status_code=response.status_code,
                content={
                    "error": {
                        "message": error_message,
                        "type": "kiro_api_error",
                        "code": response.status_code
                    }
                }
            )
        
        # Prepare data for fallback token counting
        # Convert Pydantic models to dicts for tokenizer
        messages_for_tokenizer = [msg.model_dump() for msg in request_data.messages]
        tools_for_tokenizer = [tool.model_dump() for tool in request_data.tools] if request_data.tools else None
        
        if request_data.stream:
            # Streaming mode with first token retry
            async def stream_wrapper():
                streaming_error = None
                client_disconnected = False
                try:
                    # Create retry request function for retries
                    async def make_retry_request():
                        return await http_client.request_with_retry(
                            "POST", url, kiro_payload, stream=True
                        )
                    
                    # Use retry wrapper with initial response
                    async for chunk in stream_with_first_token_retry(
                        make_request=make_retry_request,
                        client=http_client.client,
                        model=request_data.model,
                        model_cache=model_cache,
                        auth_manager=auth_manager,
                        initial_response=response,
                        request_messages=messages_for_tokenizer,
                        request_tools=tools_for_tokenizer
                    ):
                        yield chunk
                except GeneratorExit:
                    # Client disconnected - this is normal
                    client_disconnected = True
                    logger.debug("Client disconnected during streaming (GeneratorExit in routes)")
                except Exception as e:
                    streaming_error = e
                    # Headers are already sent - report the error in-band and end the
                    # generator normally so Starlette never sees a post-header raise
                    try:
                        yield build_sse_error_chunk(e)
                        yield "data: [DONE]\n\n"
                    except Exception:
                        pass  # Client already disconnected
                finally:
                    await http_client.close()
                    # Log access log for streaming (success or error)
                    if streaming_error:
                        log_streaming_error("POST /v1/chat/completions (streaming)", streaming_error)
                    elif client_disconnected:
                        logger.info(f"HTTP 200 - POST /v1/chat/completions (streaming) - client disconnected")
                    else:
                        logger.info(f"HTTP 200 - POST /v1/chat/completions (streaming) - completed")
                    # Write debug logs AFTER streaming completes
                    if debug_logger:
                        if streaming_error:
                            debug_logger.flush_on_error(getattr(streaming_error, "status_code", 500), str(streaming_error))
                        else:
                            debug_logger.discard_buffers()
            
            return StreamingResponse(stream_wrapper(), media_type="text/event-stream")
        
        else:
            
            # Non-streaming mode - collect entire response
            openai_response = await collect_stream_response(
                http_client.client,
                response,
                request_data.model,
                model_cache,
                auth_manager,
                request_messages=messages_for_tokenizer,
                request_tools=tools_for_tokenizer
            )
            
            await http_client.close()
            
            # Log access log for non-streaming success
            logger.info(f"HTTP 200 - POST /v1/chat/completions (non-streaming) - completed")
            
            # Write debug logs after non-streaming request completes
            if debug_logger:
                debug_logger.discard_buffers()
            
            return JSONResponse(content=openai_response)
    
    except HTTPException as e:
        await http_client.close()
        
        # Network errors (502/504 from request_with_retry) = RECOVERABLE
        # In legacy mode, we still log them but re-raise (no failover available)
        if e.status_code in (502, 504):
            logger.warning(f"Network error (legacy mode, no failover available)")
        
        # Log access log for HTTP error
        logger.error(f"HTTP {e.status_code} - POST /v1/chat/completions - {e.detail}")
        # Flush debug logs on HTTP error ("errors" mode)
        if debug_logger:
            debug_logger.flush_on_error(e.status_code, str(e.detail))
        raise
    except Exception as e:
        await http_client.close()
        logger.error(f"Internal error: {e}", exc_info=True)
        # Log access log for internal error
        logger.error(f"HTTP 500 - POST /v1/chat/completions - {str(e)[:100]}")
        # Flush debug logs on internal error ("errors" mode)
        if debug_logger:
            debug_logger.flush_on_error(500, str(e))
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")