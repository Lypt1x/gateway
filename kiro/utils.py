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
Utility functions for Kiro Gateway.

Contains functions for fingerprint generation, header formatting,
and other common utilities.
"""

import hashlib
import json
import uuid
from typing import TYPE_CHECKING, List, Dict, Any, Optional

from loguru import logger

from kiro.config import CODEWHISPERER_OPTOUT, KIRO_AGENT_MODE, PROFILE_ARN

if TYPE_CHECKING:
    from kiro.auth import KiroAuthManager


# ==================================================================================================
# Client identity (User-Agent) — smithy-rs / AWS SDK for Rust shape
# ==================================================================================================
#
# The real client is a Rust binary using the smithy-rs code-generated client
# `amzn-codewhisperer-streaming-client`, NOT a JavaScript aws-sdk-js client.
# Every token below is centralised here so it can be bumped in one place.
#
# Evidence (byte offsets in /home/prism/.local/bin/kiro-cli-chat):
#   aws-sdk-<name>/<ver>            smithy-rs SdkMetadata Display; `aws-sdk-` fragment present,
#                                   version taken from the bundled runtime crate below
#   aws-smithy-runtime-1.11.1       contiguous literal (crate path), source of UA_SDK_VERSION
#   ua/                             @6868722 (UA metadata prefix; smithy-rs 1.11.1 emits 2.1)
#   os/ + linux|macos|windows|...   @6868730 ("...windowslinuxmacosandroidiosotheros//ua/")
#   lang/                           @6868805
#   md/                             @6868810
#   codewhispererstreaming 0.1.17975 @6728714, adjacent to `ApiMetadata service_id version`
#   md/appVersion + 2.18.0          @399751182 (user_agent_override_interceptor.rs)
#   app/AmazonQ-For-CLI             @398430936 (contiguous literal)
#
# The literal `KiroIDE` appears ZERO times in the real binary, and no real client appends a
# per-install sha256 to its User-Agent. Both were gateway inventions and have been removed.
UA_SDK_NAME: str = "rust"
UA_SDK_VERSION: str = "1.11.1"
UA_METADATA_VERSION: str = "2.1"
UA_API_SERVICE_ID: str = "codewhispererstreaming"
UA_API_VERSION: str = "0.1.17975"
UA_OS_FAMILY: str = "linux"
UA_LANG: str = "rust"
UA_APP_VERSION: str = "2.18.0"
UA_APP_NAME: str = "AmazonQ-For-CLI"


def build_kiro_user_agent() -> str:
    """
    Builds the smithy-rs UA-2.1 User-Agent string used by the real kiro-cli.

    Token order follows the smithy-rs `AwsUserAgent` Display implementation:
    sdk metadata, ua metadata, api metadata, os metadata, language metadata,
    additional metadata, app name.

    Returns:
        The full User-Agent value (identical to x-amz-user-agent).
    """
    return (
        f"aws-sdk-{UA_SDK_NAME}/{UA_SDK_VERSION} "
        f"ua/{UA_METADATA_VERSION} "
        f"api/{UA_API_SERVICE_ID}/{UA_API_VERSION} "
        f"os/{UA_OS_FAMILY} "
        f"lang/{UA_LANG} "
        f"md/appVersion/{UA_APP_VERSION} "
        f"app/{UA_APP_NAME}"
    )


#: Precomputed User-Agent. smithy-rs sends the same value on both UA headers.
KIRO_USER_AGENT: str = build_kiro_user_agent()

#: Default max attempts advertised in `amz-sdk-request` (smithy-rs StandardRetryStrategy default).
SDK_MAX_ATTEMPTS: int = 3


def new_invocation_id() -> str:
    """
    Creates an `amz-sdk-invocation-id` value.

    smithy-rs' `InvocationIdInterceptor` generates this ONCE per logical operation
    (in `read_before_execution`) and reuses it for every retry of that operation.
    Callers must therefore create it outside their retry loop and pass it in to
    :func:`get_kiro_headers` on each attempt.

    Returns:
        A fresh UUID4 string.
    """
    return str(uuid.uuid4())


def get_machine_fingerprint() -> str:
    """
    Generates a unique machine fingerprint based on hostname and username.
    
    Used for User-Agent formation to identify a specific gateway installation.
    
    Returns:
        SHA256 hash of the string "{hostname}-{username}-kiro-gateway"
    """
    try:
        import socket
        import getpass
        
        hostname = socket.gethostname()
        username = getpass.getuser()
        unique_string = f"{hostname}-{username}-kiro-gateway"
        
        return hashlib.sha256(unique_string.encode()).hexdigest()
    except Exception as e:
        logger.warning(f"Failed to get machine fingerprint: {e}")
        return hashlib.sha256(b"default-kiro-gateway").hexdigest()


def get_kiro_headers(
    auth_manager: "KiroAuthManager",
    token: str,
    invocation_id: Optional[str] = None,
    attempt: int = 1,
    max_attempts: int = SDK_MAX_ATTEMPTS,
) -> dict:
    """
    Builds headers for Kiro API requests, matching the real kiro-cli on the wire.

    Args:
        auth_manager: Authentication manager (source of the profile ARN)
        token: Access token for authorization
        invocation_id: `amz-sdk-invocation-id` for the logical operation. MUST be
            stable across retries of the same call — create it once with
            :func:`new_invocation_id` outside the retry loop. A fresh id is
            generated when omitted (single-attempt callers).
        attempt: 1-based attempt number for `amz-sdk-request`.
        max_attempts: Max attempts advertised in `amz-sdk-request`.

    Returns:
        Dictionary with headers for HTTP request
    """
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/x-amz-json-1.0",
        "x-amz-target": "AmazonCodeWhispererStreamingService.GenerateAssistantResponse",
        # smithy-rs sets both UA headers to the same value.
        "User-Agent": KIRO_USER_AGENT,
        "x-amz-user-agent": KIRO_USER_AGENT,
        # PRIVACY OVERRIDE: the real client sends "false"; we default to "true" so the
        # user's content is not used for service improvement. See CODEWHISPERER_OPTOUT.
        "x-amzn-codewhisperer-optout": "true" if CODEWHISPERER_OPTOUT else "false",
        "x-amzn-kiro-agent-mode": KIRO_AGENT_MODE,
        "amz-sdk-invocation-id": invocation_id or new_invocation_id(),
        "amz-sdk-request": f"attempt={attempt}; max={max_attempts}",
    }

    # Real header (confirmed literal @6730542). Sourced exactly like the routes source
    # the body's profileArn. Omitted entirely when unknown — never sent blank.
    profile_arn = getattr(auth_manager, "profile_arn", None) or PROFILE_ARN
    if profile_arn:
        headers["x-amzn-kiro-profile-arn"] = profile_arn

    return headers


def generate_completion_id() -> str:
    """
    Generates a unique ID for chat completion.
    
    Returns:
        ID in format "chatcmpl-{uuid_hex}"
    """
    return f"chatcmpl-{uuid.uuid4().hex}"


def generate_conversation_id(messages: List[Dict[str, Any]] = None) -> str:
    """
    Generates a stable conversation ID based on message history.
    
    For truncation recovery, we need a stable ID that persists across requests
    in the same conversation. This is generated from a hash of key messages.
    
    If no messages provided, falls back to random UUID (for backward compatibility).
    
    Args:
        messages: List of messages in the conversation (optional)
    
    Returns:
        Stable conversation ID (16-char hex) or random UUID
    
    Example:
        >>> messages = [
        ...     {"role": "user", "content": "Hello"},
        ...     {"role": "assistant", "content": "Hi there!"}
        ... ]
        >>> conv_id = generate_conversation_id(messages)
        >>> # Same messages will always produce same ID
    """
    if not messages:
        # Fallback to random UUID for backward compatibility
        return str(uuid.uuid4())
    
    # Use first 3 messages + last message for stability
    # This ensures the ID stays the same as conversation grows,
    # but changes if the conversation history is different
    if len(messages) <= 3:
        key_messages = messages
    else:
        key_messages = messages[:3] + [messages[-1]]
    
    # Extract role and first 100 chars of content for hashing
    # This makes the hash stable even if content has minor formatting differences
    simplified_messages = []
    for msg in key_messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        
        # Handle different content formats (string, list, dict)
        if isinstance(content, str):
            content_str = content[:100]
        elif isinstance(content, list):
            # For Anthropic-style content blocks
            content_str = json.dumps(content, sort_keys=True)[:100]
        else:
            content_str = str(content)[:100]
        
        simplified_messages.append({
            "role": role,
            "content": content_str
        })
    
    # Generate stable hash
    content_json = json.dumps(simplified_messages, sort_keys=True)
    hash_digest = hashlib.sha256(content_json.encode()).hexdigest()
    
    # Return first 16 chars for readability (still 64 bits of entropy)
    return hash_digest[:16]


def generate_tool_call_id() -> str:
    """
    Generates a unique ID for tool call.
    
    Returns:
        ID in format "call_{uuid_hex[:8]}"
    """
    return f"call_{uuid.uuid4().hex[:8]}"