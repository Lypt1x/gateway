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
Converters for transforming Anthropic Messages API format to Kiro format.

This module is an adapter layer that converts Anthropic-specific formats
to the unified format used by converters_core.py.
"""

import base64
from typing import Any, Dict, List, Optional

from loguru import logger

from kiro.config import HIDDEN_MODELS
from kiro.model_resolver import get_model_id_for_kiro
from kiro.models_anthropic import (
    AnthropicMessagesRequest,
    AnthropicMessage,
    AnthropicTool,
)
from kiro.converters_core import (
    UnifiedMessage,
    UnifiedTool,
    ThinkingConfig,
    build_kiro_payload,
    extract_text_content,
    extract_images_from_content,
)


def convert_anthropic_content_to_text(content: Any) -> str:
    """
    Extracts text content from Anthropic message content.

    Anthropic content can be:
    - String: "Hello, world!"
    - List of content blocks: [{"type": "text", "text": "Hello"}]

    Args:
        content: Anthropic message content

    Returns:
        Extracted text content
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            elif hasattr(block, "type") and block.type == "text":
                text_parts.append(block.text)
        return "".join(text_parts)

    return str(content) if content else ""


def extract_system_prompt(system: Any) -> str:
    """
    Extracts system prompt text from Anthropic system field.

    Anthropic API supports system in two formats:
    1. String: "You are helpful"
    2. List of content blocks: [{"type": "text", "text": "...", "cache_control": {...}}]

    The second format is used for prompt caching with cache_control.
    We extract only the text, ignoring cache_control (not supported by Kiro).

    Args:
        system: System prompt in string or list format

    Returns:
        Extracted system prompt as string
    """
    if system is None:
        return ""

    if isinstance(system, str):
        return system

    if isinstance(system, list):
        text_parts = []
        for block in system:
            if isinstance(block, dict):
                # Handle {"type": "text", "text": "...", "cache_control": {...}}
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            elif hasattr(block, "type") and block.type == "text":
                # Handle Pydantic model
                text_parts.append(getattr(block, "text", ""))
        return "\n".join(text_parts)

    return str(system)


def normalize_inline_system_messages(
    messages: List[AnthropicMessage],
    system: Any = None,
) -> tuple:
    """
    Hoists inline ``role == "system"`` messages out of ``messages``.

    Some clients place system instructions inside the ``messages`` array instead
    of using the top-level ``system`` field. Such messages are removed from the
    conversation and merged into the effective system prompt **before** the
    top-level ``system`` content, preserving their relative order. Any remaining
    non-standard role (e.g. ``"tool"``, ``"developer"``) is coerced to ``"user"``
    so the turn is kept rather than dropped.

    Both content shapes are supported: a plain string or a list of content
    blocks (dicts or Pydantic models).

    Args:
        messages: List of Anthropic messages (possibly containing system entries)
        system: Top-level system field (str, list of blocks, or None)

    Returns:
        Tuple of (remaining_messages, merged_system) where merged_system is
        suitable input for extract_system_prompt().
    """
    remaining: List[AnthropicMessage] = []
    hoisted_blocks: List[Dict[str, Any]] = []

    for msg in messages:
        role = getattr(msg, "role", None)

        if role == "system":
            text = convert_anthropic_content_to_text(msg.content)
            if text:
                hoisted_blocks.append({"type": "text", "text": text})
            continue

        if role not in ("user", "assistant"):
            # Keep the turn, but attribute it to the user rather than dropping it.
            logger.debug(f"Coercing non-standard message role '{role}' to 'user'")
            try:
                msg = msg.model_copy(update={"role": "user"})
            except Exception:  # pragma: no cover - defensive
                msg.role = "user"

        remaining.append(msg)

    if not hoisted_blocks:
        return remaining, system

    # Append the original top-level system content after the hoisted blocks.
    tail_text = extract_system_prompt(system)
    if tail_text:
        hoisted_blocks.append({"type": "text", "text": tail_text})

    logger.debug(
        f"Hoisted {len(hoisted_blocks)} system block(s) from messages "
        f"({len(messages) - len(remaining)} inline system message(s) removed)"
    )

    return remaining, hoisted_blocks


# ==================================================================================================
# Document content blocks (issue #176)
# ==================================================================================================
#
# UPSTREAM LIMITATION - READ BEFORE "FIXING" THIS.
#
# Anthropic clients (Claude Code) may send {"type": "document", "source": {...}} blocks, typically a
# base64 PDF. The Kiro / CodeWhisperer `generateAssistantResponse` payload has NO document channel:
# the only binary attachment slot is `userInputMessage.images`, which carries image formats only
# (see kiro.converters_core.convert_images_to_kiro_format). There is therefore NO way to forward
# PDF bytes upstream, and this gateway does NOT support native PDF reading.
#
# What we do instead is a truthful, observable degradation:
#   * the block is accepted (no HTTP 422 - that was the actual bug in #176);
#   * text-like documents (text/plain, text/markdown, text/csv, application/json, ...) are decoded
#     and inlined verbatim, which is the obvious safe thing and genuinely works;
#   * anything else (PDF, docx, images-as-documents, URL sources, malformed sources) is replaced by
#     an explicit text placeholder naming the document and its media type, so the model can tell the
#     user it cannot see the file instead of silently hallucinating its contents.
#
# If Kiro ever gains a document channel, replace the placeholder path - do not remove the block
# acceptance.

# Media types whose base64 payload is safe to decode and inline as text.
_TEXTUAL_DOCUMENT_MEDIA_TYPES = frozenset(
    {
        "text/plain",
        "text/markdown",
        "text/md",
        "text/csv",
        "text/html",
        "text/xml",
        "application/json",
        "application/xml",
    }
)

# Cap for inlined document text so a large attachment cannot blow up the payload.
_MAX_INLINE_DOCUMENT_CHARS = 200_000


def _as_dict(obj: Any) -> Dict[str, Any]:
    """Best-effort conversion of a content block / source to a plain dict."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:  # pragma: no cover - defensive
            pass
    return {}


def render_document_block(block: Any) -> str:
    """
    Renders an Anthropic ``document`` content block as plain text.

    See the module-level note above: Kiro cannot accept document bytes, so a
    document is either inlined as text (when its media type is textual) or
    represented by an explicit placeholder.

    Args:
        block: A ``{"type": "document", ...}`` block (dict or Pydantic model)

    Returns:
        Text to inject into the conversation. Never raises.
    """
    try:
        data = _as_dict(block)
        title = data.get("title") or "untitled document"
        source = data.get("source")
        source_dict = _as_dict(source)

        source_type = source_dict.get("type")
        media_type = source_dict.get("media_type") or ""

        if source_type == "text":
            text = source_dict.get("data") or source_dict.get("text") or ""
            if text:
                return (
                    f"[Attached document: {title} ({media_type or 'text/plain'})]\n"
                    f"{str(text)[:_MAX_INLINE_DOCUMENT_CHARS]}"
                )

        if source_type == "base64":
            raw = source_dict.get("data") or ""
            if media_type in _TEXTUAL_DOCUMENT_MEDIA_TYPES and raw:
                try:
                    decoded = base64.b64decode(raw, validate=False).decode(
                        "utf-8", errors="replace"
                    )
                except Exception as exc:
                    logger.warning(f"Failed to decode textual document '{title}': {exc}")
                else:
                    if decoded:
                        return (
                            f"[Attached document: {title} ({media_type})]\n"
                            f"{decoded[:_MAX_INLINE_DOCUMENT_CHARS]}"
                        )

            size_hint = f", ~{len(raw) * 3 // 4} bytes" if raw else ""
            return (
                f"[Attached document: {title} (media_type={media_type or 'unknown'}{size_hint}). "
                "This gateway cannot forward document contents to the upstream model, which accepts "
                "text and images only. The document was NOT read. Ask the user to paste the relevant "
                "text, or use a file-reading tool.]"
            )

        if source_type == "url":
            url = source_dict.get("url") or ""
            return (
                f"[Attached document: {title} (URL: {url}). This gateway cannot fetch or forward "
                "document contents to the upstream model. The document was NOT read.]"
            )

        # Unknown / malformed source: still degrade gracefully rather than fail.
        return (
            f"[Attached document: {title} with an unsupported or malformed source "
            f"(source type={source_type or 'missing'}). The document was NOT read.]"
        )
    except Exception as exc:  # pragma: no cover - defensive, must never 500
        logger.warning(f"Failed to render document content block: {exc}")
        return "[Attached document could not be processed and was NOT read.]"


def extract_documents_from_content(content: Any) -> List[str]:
    """
    Collects rendered text for every ``document`` block in a content list.

    Args:
        content: Anthropic message content (list of content blocks)

    Returns:
        List of rendered text fragments (empty if there are no document blocks).
    """
    rendered: List[str] = []

    if not isinstance(content, list):
        return rendered

    for block in content:
        block_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
        if block_type == "document":
            rendered.append(render_document_block(block))

    if rendered:
        logger.warning(
            f"{len(rendered)} document block(s) degraded to text: the upstream Kiro API has no "
            "document channel (see issue #176)"
        )

    return rendered


def extract_tool_results_from_anthropic_content(content: Any) -> List[Dict[str, Any]]:
    """
    Extracts tool results from Anthropic message content.

    Looks for content blocks with type="tool_result".

    Args:
        content: Anthropic message content (list of content blocks)

    Returns:
        List of tool results in unified format
    """
    tool_results = []

    if not isinstance(content, list):
        return tool_results

    for block in content:
        block_type = None
        tool_use_id = None
        result_content = ""

        if isinstance(block, dict):
            block_type = block.get("type")
            tool_use_id = block.get("tool_use_id")
            result_content = block.get("content", "")
        elif hasattr(block, "type"):
            block_type = block.type
            tool_use_id = getattr(block, "tool_use_id", None)
            result_content = getattr(block, "content", "")

        if block_type == "tool_result" and tool_use_id:
            # Convert content to text if it's a list
            if isinstance(result_content, list):
                # Documents inside a tool_result are degraded to text as well (issue #176).
                document_texts = extract_documents_from_content(result_content)
                result_content = extract_text_content(result_content)
                if document_texts:
                    result_content = "\n\n".join(
                        part for part in [result_content, *document_texts] if part
                    )
            elif not isinstance(result_content, str):
                result_content = str(result_content) if result_content else ""

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": result_content or "(empty result)",
                }
            )

    return tool_results


def extract_images_from_tool_results(content: Any) -> List[Dict[str, Any]]:
    """
    Extracts images from tool_result content blocks.

    Tool results in Anthropic format can contain images (e.g., screenshots from browser tools).
    This function extracts those images so they can be passed to the model.

    Args:
        content: Anthropic message content (list of content blocks)

    Returns:
        List of images in unified format: [{"media_type": "image/jpeg", "data": "base64..."}]
    """
    images: List[Dict[str, Any]] = []

    if not isinstance(content, list):
        return images

    for block in content:
        block_type = None
        result_content = None

        if isinstance(block, dict):
            block_type = block.get("type")
            result_content = block.get("content")
        elif hasattr(block, "type"):
            block_type = block.type
            result_content = getattr(block, "content", None)

        if block_type == "tool_result" and isinstance(result_content, list):
            # Extract images from the tool_result's content
            tool_result_images = extract_images_from_content(result_content)
            images.extend(tool_result_images)

    if images:
        logger.debug(f"Extracted {len(images)} image(s) from tool_result content")

    return images

    return tool_results


def extract_tool_uses_from_anthropic_content(content: Any) -> List[Dict[str, Any]]:
    """
    Extracts tool uses from Anthropic assistant message content.

    Looks for content blocks with type="tool_use".

    Args:
        content: Anthropic message content (list of content blocks)

    Returns:
        List of tool calls in unified format
    """
    tool_calls = []

    if not isinstance(content, list):
        return tool_calls

    for block in content:
        block_type = None
        tool_id = None
        tool_name = None
        tool_input = {}

        if isinstance(block, dict):
            block_type = block.get("type")
            tool_id = block.get("id")
            tool_name = block.get("name")
            tool_input = block.get("input", {})
        elif hasattr(block, "type"):
            block_type = block.type
            tool_id = getattr(block, "id", None)
            tool_name = getattr(block, "name", None)
            tool_input = getattr(block, "input", {})

        if block_type == "tool_use" and tool_id and tool_name:
            tool_calls.append(
                {
                    "id": tool_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": tool_input
                        if isinstance(tool_input, str)
                        else tool_input,
                    },
                }
            )

    return tool_calls


def convert_anthropic_messages(
    messages: List[AnthropicMessage],
) -> List[UnifiedMessage]:
    """
    Converts Anthropic messages to unified format.

    Handles:
    - Text content (string or list of text blocks)
    - Tool use blocks (assistant messages)
    - Tool result blocks (user messages)

    Args:
        messages: List of Anthropic messages

    Returns:
        List of messages in unified format
    """

    unified_messages = []
    total_tool_calls = 0
    total_tool_results = 0
    total_images = 0
    total_documents = 0

    for msg in messages:
        role = msg.role
        content = msg.content

        # Extract text content
        text_content = convert_anthropic_content_to_text(content)

        # Documents cannot be forwarded to Kiro; degrade them to text so the block is
        # neither rejected (HTTP 422) nor silently dropped (issue #176).
        document_texts = extract_documents_from_content(content)
        if document_texts:
            total_documents += len(document_texts)
            text_content = "\n\n".join(
                part for part in [text_content, *document_texts] if part
            )

        # Extract tool-related data and images based on role
        tool_calls = None
        tool_results = None
        images = None

        if role == "assistant":
            # Assistant messages may contain tool_use blocks
            tool_calls = extract_tool_uses_from_anthropic_content(content)
            if tool_calls:
                total_tool_calls += len(tool_calls)

        elif role == "user":
            # User messages may contain tool_result blocks and images
            tool_results = extract_tool_results_from_anthropic_content(content)
            if tool_results:
                total_tool_results += len(tool_results)

            # Extract images from user messages (both top-level and inside tool_results)
            images = extract_images_from_content(content)

            # Also extract images from inside tool_result content blocks
            # (e.g., screenshots returned by browser MCP tools)
            tool_result_images = extract_images_from_tool_results(content)
            if tool_result_images:
                if images:
                    images.extend(tool_result_images)
                else:
                    images = tool_result_images

            if images:
                total_images += len(images)

        unified_msg = UnifiedMessage(
            role=role,
            content=text_content,
            tool_calls=tool_calls if tool_calls else None,
            tool_results=tool_results if tool_results else None,
            images=images if images else None,
        )
        unified_messages.append(unified_msg)

    # Log summary if any tool content or images were found
    if total_tool_calls > 0 or total_tool_results > 0 or total_images > 0 or total_documents > 0:
        logger.debug(
            f"Converted {len(messages)} Anthropic messages: "
            f"{total_tool_calls} tool_calls, {total_tool_results} tool_results, "
            f"{total_images} images, {total_documents} documents"
        )

    return unified_messages


def convert_anthropic_tools(
    tools: Optional[List[AnthropicTool]],
) -> Optional[List[UnifiedTool]]:
    """
    Converts Anthropic tools to unified format.

    Args:
        tools: List of Anthropic tools

    Returns:
        List of tools in unified format, or None if no tools
    """
    if not tools:
        return None

    unified_tools = []
    for tool in tools:
        # Handle both dict and Pydantic model
        if isinstance(tool, dict):
            name = tool.get("name", "")
            description = tool.get("description")
            input_schema = tool.get("input_schema", {})
        else:
            name = tool.name
            description = tool.description
            input_schema = tool.input_schema

        unified_tools.append(
            UnifiedTool(name=name, description=description, input_schema=input_schema)
        )

    return unified_tools if unified_tools else None


def extract_thinking_config_from_anthropic(request: AnthropicMessagesRequest) -> ThinkingConfig:
    """
    Extract thinking configuration from Anthropic request.
    
    Handles thinking parameter:
    - {"type": "enabled", "budget_tokens": N} → enabled with budget
    - {"type": "disabled"} → disabled
    - None → enabled with default budget
    
    Args:
        request: Anthropic MessagesRequest
    
    Returns:
        ThinkingConfig for core layer
    
    Examples:
        >>> # No thinking specified → use defaults
        >>> request = AnthropicMessagesRequest(model="claude-sonnet-4.5", messages=[...], max_tokens=4096)
        >>> extract_thinking_config_from_anthropic(request)
        ThinkingConfig(enabled=True, budget_tokens=None)
        
        >>> # Explicitly disabled
        >>> request.thinking = {"type": "disabled"}
        >>> extract_thinking_config_from_anthropic(request)
        ThinkingConfig(enabled=False, budget_tokens=None)
        
        >>> # Enabled with custom budget
        >>> request.thinking = {"type": "enabled", "budget_tokens": 8000}
        >>> extract_thinking_config_from_anthropic(request)
        ThinkingConfig(enabled=True, budget_tokens=8000)
    """
    if not request.thinking:
        # No thinking specified → use defaults
        return ThinkingConfig(enabled=True, budget_tokens=None)
    
    if not isinstance(request.thinking, dict):
        # Invalid format → use defaults
        return ThinkingConfig(enabled=True, budget_tokens=None)
    
    thinking_type = request.thinking.get("type")
    
    if thinking_type == "disabled":
        # Explicitly disabled
        return ThinkingConfig(enabled=False, budget_tokens=None)
    
    if thinking_type == "enabled":
        # Extract budget_tokens
        budget = request.thinking.get("budget_tokens")
        if budget:
            logger.debug(f"Extracted thinking config from Anthropic: type='enabled', budget={budget}")
        return ThinkingConfig(enabled=True, budget_tokens=budget)
    
    # Unknown type → use defaults
    return ThinkingConfig(enabled=True, budget_tokens=None)


def anthropic_to_kiro(
    request: AnthropicMessagesRequest, conversation_id: str, profile_arn: str
) -> dict:
    """
    Converts Anthropic Messages API request to Kiro API payload.

    This is the main entry point for Anthropic → Kiro conversion.

    Key differences from OpenAI:
    - System prompt is a separate field (not in messages)
    - Content can be string or list of content blocks
    - Tool format uses input_schema instead of parameters

    Args:
        request: Anthropic MessagesRequest
        conversation_id: Unique conversation ID
        profile_arn: AWS CodeWhisperer profile ARN

    Returns:
        Payload dictionary for POST request to Kiro API

    Raises:
        ValueError: If there are no messages to send
    """
    # Hoist inline role="system" messages into the effective system prompt and
    # coerce any remaining non-standard roles to "user" (see FIX-01).
    normalized_messages, effective_system = normalize_inline_system_messages(
        request.messages, request.system
    )

    if not normalized_messages:
        raise ValueError("No messages to send after removing inline system messages")

    # Convert messages to unified format
    unified_messages = convert_anthropic_messages(normalized_messages)

    # Convert tools to unified format
    unified_tools = convert_anthropic_tools(request.tools)

    # System prompt is already separate in Anthropic format!
    # It can be a string or list of content blocks (for prompt caching)
    system_prompt = extract_system_prompt(effective_system)

    # Get model ID for Kiro API (normalizes + resolves hidden models)
    # Pass-through principle: we normalize and send to Kiro, Kiro decides if valid
    model_id = get_model_id_for_kiro(request.model, HIDDEN_MODELS)

    # Extract thinking configuration from thinking parameter
    thinking_config = extract_thinking_config_from_anthropic(request)

    logger.debug(
        f"Converting Anthropic request: model={request.model} -> {model_id}, "
        f"messages={len(unified_messages)}, tools={len(unified_tools) if unified_tools else 0}, "
        f"system_prompt_length={len(system_prompt)}, "
        f"thinking_enabled={thinking_config.enabled}, thinking_budget={thinking_config.budget_tokens}"
    )

    # Use core function to build payload
    result = build_kiro_payload(
        messages=unified_messages,
        system_prompt=system_prompt,
        model_id=model_id,
        tools=unified_tools,
        conversation_id=conversation_id,
        profile_arn=profile_arn,
        thinking_config=thinking_config,
    )

    return result.payload
