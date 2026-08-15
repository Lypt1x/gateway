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
Parsers for AWS Event Stream format.

Contains classes and functions for:
- Parsing binary AWS SSE stream
- Extracting JSON events
- Processing tool calls
- Content deduplication
"""

import json
import re
from contextvars import ContextVar
from typing import Any, Dict, List, Optional

from loguru import logger

from kiro.utils import generate_tool_call_id


# ==================================================================================================
# Per-turn upstream stop reason (metadataEvent.stopReason)
# ==================================================================================================
#
# The routing parser already records the last `stopReason` on itself
# (:attr:`EventStreamRoutingParser.last_stop_reason`), but the parser instance is
# created inside ``streaming_core.parse_kiro_stream`` and never handed to the
# dialect formatters, so the value used to be recorded and then discarded.
#
# This ContextVar is that missing accessor: the parser records the value, and the
# dialect layer (streaming_openai / streaming_anthropic) reads it after the stream
# ends. A ContextVar - not a module global - so concurrent requests, which run in
# separate asyncio tasks with their own copied context, cannot see each other's
# value. Callers reset it at the start of a turn.
_STREAM_STOP_REASON: ContextVar[Optional[str]] = ContextVar(
    "kiro_stream_stop_reason", default=None
)


def record_stream_stop_reason(stop_reason: Optional[str]) -> None:
    """Records the upstream stop reason for the current turn."""
    _STREAM_STOP_REASON.set(stop_reason)


def get_stream_stop_reason() -> Optional[str]:
    """Returns the last upstream stop reason seen in the current turn (or None)."""
    return _STREAM_STOP_REASON.get()


def reset_stream_stop_reason() -> None:
    """Clears the stop reason. Called by the dialect layer before a turn starts."""
    _STREAM_STOP_REASON.set(None)


def find_matching_brace(text: str, start_pos: int) -> int:
    """
    Finds the position of the closing brace considering nesting and strings.
    
    Uses bracket counting for correct parsing of nested JSON.
    Accounts for quoted strings and escape sequences.
    
    Args:
        text: Text to search
        start_pos: Position of opening brace '{'
    
    Returns:
        Position of closing brace or -1 if not found
    
    Example:
        >>> find_matching_brace('{"a": {"b": 1}}', 0)
        14
        >>> find_matching_brace('{"a": "{}"}', 0)
        10
    """
    if start_pos >= len(text) or text[start_pos] != '{':
        return -1
    
    brace_count = 0
    in_string = False
    escape_next = False
    
    for i in range(start_pos, len(text)):
        char = text[i]
        
        if escape_next:
            escape_next = False
            continue
        
        if char == '\\' and in_string:
            escape_next = True
            continue
        
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        
        if not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return i
    
    return -1


def parse_bracket_tool_calls(response_text: str) -> List[Dict[str, Any]]:
    """
    Parses tool calls in [Called func_name with args: {...}] format.
    
    Some models return tool calls in text format instead of
    structured JSON. This function extracts them.
    
    Args:
        response_text: Model response text
    
    Returns:
        List of tool calls in OpenAI format
    
    Example:
        >>> text = "[Called get_weather with args: {\"city\": \"London\"}]"
        >>> calls = parse_bracket_tool_calls(text)
        >>> calls[0]["function"]["name"]
        'get_weather'
    """
    if not response_text or "[Called" not in response_text:
        return []
    
    tool_calls = []
    pattern = r'\[Called\s+(\w+)\s+with\s+args:\s*'
    
    for match in re.finditer(pattern, response_text, re.IGNORECASE):
        func_name = match.group(1)
        args_start = match.end()
        
        # Find JSON start
        json_start = response_text.find('{', args_start)
        if json_start == -1:
            continue
        
        # Find JSON end considering nesting
        json_end = find_matching_brace(response_text, json_start)
        if json_end == -1:
            continue
        
        json_str = response_text[json_start:json_end + 1]
        
        try:
            args = json.loads(json_str)
            tool_call_id = generate_tool_call_id()
            # index will be added later when forming the final response
            tool_calls.append({
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": json.dumps(args)
                }
            })
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse tool call arguments: {json_str[:100]}")
    
    return tool_calls


def deduplicate_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Removes duplicate tool calls.
    
    Deduplication occurs by two criteria:
    1. By id - if there are multiple tool calls with the same id, keep the one with
       more arguments (not empty "{}")
    2. By name+arguments - remove complete duplicates
    
    Args:
        tool_calls: List of tool calls
    
    Returns:
        List of unique tool calls
    """
    # First deduplicate by id - keep tool call with non-empty arguments
    by_id: Dict[str, Dict[str, Any]] = {}
    for tc in tool_calls:
        tc_id = tc.get("id", "")
        if not tc_id:
            # Without id - add as is (will be deduplicated by name+args)
            continue
        
        existing = by_id.get(tc_id)
        if existing is None:
            by_id[tc_id] = tc
        else:
            # Duplicate by id exists - keep the one with more arguments
            existing_args = existing.get("function", {}).get("arguments", "{}")
            current_args = tc.get("function", {}).get("arguments", "{}")
            
            # Prefer non-empty arguments
            if current_args != "{}" and (existing_args == "{}" or len(current_args) > len(existing_args)):
                logger.debug(f"Replacing tool call {tc_id} with better arguments: {len(existing_args)} -> {len(current_args)}")
                by_id[tc_id] = tc
    
    # Collect tool calls: first those with id, then without id
    result_with_id = list(by_id.values())
    result_without_id = [tc for tc in tool_calls if not tc.get("id")]
    
    # Now deduplicate by name+arguments for all
    seen = set()
    unique = []
    
    for tc in result_with_id + result_without_id:
        # Protection against None in function
        func = tc.get("function") or {}
        func_name = func.get("name") or ""
        func_args = func.get("arguments") or "{}"
        key = f"{func_name}-{func_args}"
        if key not in seen:
            seen.add(key)
            unique.append(tc)
    
    if len(tool_calls) != len(unique):
        logger.debug(f"Deduplicated tool calls: {len(tool_calls)} -> {len(unique)}")
    
    return unique


class AwsEventStreamParser:
    """
    Parser for AWS Event Stream format.
    
    AWS returns events in binary format with :message-type...event delimiters.
    This class extracts JSON events from the stream and converts them to a convenient format.
    
    Supported event types:
    - content: Text content of response
    - tool_start: Start of tool call (name, toolUseId)
    - tool_input: Continuation of input for tool call
    - tool_stop: End of tool call
    - usage: Credit consumption information
    - context_usage: Context usage percentage
    
    Attributes:
        buffer: Buffer for accumulating data
        last_content: Last processed content (for deduplication)
        current_tool_call: Current incomplete tool call
        tool_calls: List of completed tool calls
    
    Example:
        >>> parser = AwsEventStreamParser()
        >>> events = parser.feed(chunk)
        >>> for event in events:
        ...     if event["type"] == "content":
        ...         print(event["data"])
    """
    
    # Patterns for finding JSON events
    EVENT_PATTERNS = [
        ('{"content":', 'content'),
        ('{"name":', 'tool_start'),
        ('{"input":', 'tool_input'),
        ('{"stop":', 'tool_stop'),
        ('{"followupPrompt":', 'followup'),
        ('{"usage":', 'usage'),
        ('{"contextUsagePercentage":', 'context_usage'),
    ]
    
    def __init__(self):
        """Initializes the parser."""
        self.buffer = ""
        self.last_content: Optional[str] = None  # For deduplicating repeating content
        self.current_tool_call: Optional[Dict[str, Any]] = None
        self.tool_calls: List[Dict[str, Any]] = []
    
    def feed(self, chunk: bytes) -> List[Dict[str, Any]]:
        """
        Adds chunk to buffer and returns parsed events.
        
        Args:
            chunk: Bytes of data from stream
        
        Returns:
            List of events in {"type": str, "data": Any} format
        """
        try:
            self.buffer += chunk.decode('utf-8', errors='ignore')
        except Exception:
            return []
        
        events = []
        
        while True:
            # Find nearest pattern
            earliest_pos = -1
            earliest_type = None
            
            for pattern, event_type in self.EVENT_PATTERNS:
                pos = self.buffer.find(pattern)
                if pos != -1 and (earliest_pos == -1 or pos < earliest_pos):
                    earliest_pos = pos
                    earliest_type = event_type
            
            if earliest_pos == -1:
                break
            
            # Find JSON end
            json_end = find_matching_brace(self.buffer, earliest_pos)
            if json_end == -1:
                # JSON not complete, wait for more data
                break
            
            json_str = self.buffer[earliest_pos:json_end + 1]
            self.buffer = self.buffer[json_end + 1:]
            
            try:
                data = json.loads(json_str)
                event = self._process_event(data, earliest_type)
                if event:
                    events.append(event)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON: {json_str[:100]}")
        
        return events
    
    def _process_event(self, data: dict, event_type: str) -> Optional[Dict[str, Any]]:
        """
        Processes a parsed event.
        
        Args:
            data: Parsed JSON
            event_type: Event type
        
        Returns:
            Processed event or None
        """
        if event_type == 'content':
            return self._process_content_event(data)
        elif event_type == 'tool_start':
            return self._process_tool_start_event(data)
        elif event_type == 'tool_input':
            return self._process_tool_input_event(data)
        elif event_type == 'tool_stop':
            return self._process_tool_stop_event(data)
        elif event_type == 'usage':
            return {"type": "usage", "data": data.get('usage', 0)}
        elif event_type == 'context_usage':
            return {"type": "context_usage", "data": data.get('contextUsagePercentage', 0)}
        
        return None
    
    def _process_content_event(self, data: dict) -> Optional[Dict[str, Any]]:
        """Processes content event."""
        content = data.get('content', '')
        
        # Skip followupPrompt
        if data.get('followupPrompt'):
            return None
        
        # Deduplicate repeating content
        if content == self.last_content:
            return None
        
        self.last_content = content
        
        return {"type": "content", "data": content}
    
    def _process_tool_start_event(self, data: dict) -> Optional[Dict[str, Any]]:
        """Processes tool call start."""
        # Finalize previous tool call if exists
        if self.current_tool_call:
            self._finalize_tool_call()
        
        # input can be string or object
        input_data = data.get('input', '')
        # Discriminated accumulator: a non-empty dict fragment starts dict mode so that
        # later dict fragments can be merged instead of string-concatenated (invalid JSON).
        # String fragments (the common partial-JSON streaming case) stay in string mode.
        args: Any
        if isinstance(input_data, dict):
            # Empty dict {}: fragments will follow, start in string mode with ''
            args = dict(input_data) if input_data else ''
        else:
            args = str(input_data) if input_data else ''
        
        self.current_tool_call = {
            "id": data.get('toolUseId', generate_tool_call_id()),
            "type": "function",
            "function": {
                "name": data.get('name', ''),
                "arguments": args
            }
        }
        
        if data.get('stop'):
            self._finalize_tool_call()
        
        return None
    
    def _process_tool_input_event(self, data: dict) -> Optional[Dict[str, Any]]:
        """Processes input continuation for tool call."""
        if self.current_tool_call:
            # input can be string or object
            input_data = data.get('input', '')
            fn = self.current_tool_call['function']
            current = fn['arguments']
            
            if isinstance(input_data, dict):
                if not input_data:
                    return None
                if isinstance(current, dict):
                    # Dict mode: merge fragments, later keys win
                    current.update(input_data)
                elif isinstance(current, str) and not current:
                    # Nothing accumulated yet: enter dict mode
                    fn['arguments'] = dict(input_data)
                else:
                    # Mixed shapes: prefer the string path (previous behaviour)
                    fn['arguments'] = current + json.dumps(input_data)
            else:
                input_str = str(input_data) if input_data else ''
                if not input_str:
                    return None
                if isinstance(current, dict):
                    # Mixed shapes: fall back to string concatenation as before
                    fn['arguments'] = json.dumps(current) + input_str
                else:
                    fn['arguments'] = current + input_str
        return None
    
    def _process_tool_stop_event(self, data: dict) -> Optional[Dict[str, Any]]:
        """Processes tool call end."""
        if self.current_tool_call and data.get('stop'):
            self._finalize_tool_call()
        return None
    
    def _finalize_tool_call(self) -> None:
        """Finalizes current tool call and adds to list."""
        if not self.current_tool_call:
            return
        
        # Try to parse and normalize arguments as JSON
        args = self.current_tool_call['function']['arguments']
        tool_name = self.current_tool_call['function'].get('name', 'unknown')
        
        logger.debug(f"Finalizing tool call '{tool_name}' with raw arguments: {repr(args)[:200]}")
        
        if isinstance(args, str):
            if args.strip():
                try:
                    parsed = json.loads(args)
                    # Ensure result is a JSON string
                    self.current_tool_call['function']['arguments'] = json.dumps(parsed)
                    logger.debug(f"Tool '{tool_name}' arguments parsed successfully: {list(parsed.keys()) if isinstance(parsed, dict) else type(parsed)}")
                except json.JSONDecodeError as e:
                    # Analyze the failure to provide better diagnostics
                    truncation_info = self._diagnose_json_truncation(args)
                    
                    if truncation_info["is_truncated"]:
                        # Mark for recovery system
                        self.current_tool_call['_truncation_detected'] = True
                        self.current_tool_call['_truncation_info'] = truncation_info
                        
                        # Check if recovery is enabled
                        from kiro.config import TRUNCATION_RECOVERY
                        tool_id = self.current_tool_call.get('id', 'unknown')
                        
                        # Clear error message: this is Kiro API's fault, not ours
                        logger.error(
                            f"Tool call truncated by Kiro API: "
                            f"tool='{tool_name}', id={tool_id}, size={truncation_info['size_bytes']} bytes, "
                            f"reason={truncation_info['reason']}. "
                            f"This is a Kiro API limitation. "
                            f"{'Model will be notified automatically about truncation.' if TRUNCATION_RECOVERY else 'Set TRUNCATION_RECOVERY=true in .env to auto-notify model about truncation.'}"
                        )
                    else:
                        # Regular JSON parse error
                        logger.warning(f"Failed to parse tool '{tool_name}' arguments: {e}. Raw: {args[:200]}")
                    
                    self.current_tool_call['function']['arguments'] = "{}"
            else:
                # Empty string - use empty object
                # This is normal behavior for duplicate tool calls from Kiro
                logger.debug(f"Tool '{tool_name}' has empty arguments string (will be deduplicated)")
                self.current_tool_call['function']['arguments'] = "{}"
        elif isinstance(args, dict):
            # If already an object - serialize to string
            self.current_tool_call['function']['arguments'] = json.dumps(args)
            logger.debug(f"Tool '{tool_name}' arguments already dict with keys: {list(args.keys())}")
        else:
            # Unknown type - empty object
            logger.warning(f"Tool '{tool_name}' has unexpected arguments type: {type(args)}")
            self.current_tool_call['function']['arguments'] = "{}"
        
        self.tool_calls.append(self.current_tool_call)
        self.current_tool_call = None
    
    def _diagnose_json_truncation(self, json_str: str) -> Dict[str, Any]:
        """
        Analyzes a malformed JSON string to determine if it was truncated.
        
        This helps distinguish between upstream issues (Kiro API cutting off
        large tool call arguments) and actual malformed JSON from the model.
        
        Args:
            json_str: The raw JSON string that failed to parse
        
        Returns:
            Dictionary with diagnostic information:
            - is_truncated: True if the JSON appears to be cut off
            - reason: Human-readable explanation of why it's truncated
            - size_bytes: Size of the received data
        """
        size_bytes = len(json_str.encode('utf-8'))
        stripped = json_str.strip()
        
        # Check for obvious truncation signs
        if not stripped:
            return {"is_truncated": False, "reason": "empty string", "size_bytes": size_bytes}
        
        # Count braces and brackets (simplified, doesn't account for strings perfectly)
        open_braces = stripped.count('{')
        close_braces = stripped.count('}')
        open_brackets = stripped.count('[')
        close_brackets = stripped.count(']')
        
        # Check if JSON starts with { but doesn't end with }
        if stripped.startswith('{') and not stripped.endswith('}'):
            missing = open_braces - close_braces
            return {
                "is_truncated": True,
                "reason": f"missing {missing} closing brace(s)",
                "size_bytes": size_bytes
            }
        
        # Check if JSON starts with [ but doesn't end with ]
        if stripped.startswith('[') and not stripped.endswith(']'):
            missing = open_brackets - close_brackets
            return {
                "is_truncated": True,
                "reason": f"missing {missing} closing bracket(s)",
                "size_bytes": size_bytes
            }
        
        # Check for unbalanced braces/brackets
        if open_braces != close_braces:
            diff = open_braces - close_braces
            return {
                "is_truncated": True,
                "reason": f"unbalanced braces ({open_braces} open, {close_braces} close)",
                "size_bytes": size_bytes
            }
        
        if open_brackets != close_brackets:
            diff = open_brackets - close_brackets
            return {
                "is_truncated": True,
                "reason": f"unbalanced brackets ({open_brackets} open, {close_brackets} close)",
                "size_bytes": size_bytes
            }
        
        # Check for unclosed string (ends with backslash or inside quotes)
        # This is a heuristic - count unescaped quotes
        quote_count = 0
        i = 0
        while i < len(stripped):
            if stripped[i] == '\\' and i + 1 < len(stripped):
                i += 2  # Skip escaped character
                continue
            if stripped[i] == '"':
                quote_count += 1
            i += 1
        
        if quote_count % 2 != 0:
            return {
                "is_truncated": True,
                "reason": "unclosed string literal",
                "size_bytes": size_bytes
            }
        
        # Doesn't look truncated, probably just malformed
        return {"is_truncated": False, "reason": "malformed JSON", "size_bytes": size_bytes}
    
    def get_tool_calls(self) -> List[Dict[str, Any]]:
        """
        Returns all collected tool calls.
        
        Finalizes current tool call if not finished.
        Removes duplicates.
        
        Returns:
            List of unique tool calls
        """
        if self.current_tool_call:
            self._finalize_tool_call()
        return deduplicate_tool_calls(self.tool_calls)
    
    def reset(self) -> None:
        """Resets parser state."""
        self.buffer = ""
        self.last_content = None
        self.current_tool_call = None
        self.tool_calls = []



# ==================================================================================================
# Real event-stream framing (RECOMMENDATION 1)
# ==================================================================================================

# COMPLETE upstream payload reference (captured from real traffic across six
# scenarios: plain text, reasoning, tool call, gpt-5.6-terra, qwen3-coder-next,
# long output). Every frame observed carried `:message-type = event`.
#
# :event-type              payload keys                       internal event(s)
# ------------------------ ---------------------------------- ------------------
# assistantResponseEvent   content   (str, token fragment)    content
#                          modelId   (str, e.g.              (recorded as
#                                     "claude-sonnet-4.5")    last_model_id)
#                          NOTE: Claude emits `<thinking>`    (ThinkingParser
#                          tags INSIDE `content`.             splits those out)
# reasoningContentEvent    text      (str, native reasoning)  thinking
#                          signature (str, opaque blob)       (carried on the
#                                                             event, never
#                                                             logged)
# toolUseEvent             name      (str, tool name)         tool accumulation
#                          toolUseId (str, identity - repeats  via the legacy
#                                     in EVERY frame)          FIX-03 handlers;
#                          input     (str, partial-JSON        results come from
#                                     fragment)                get_tool_calls()
#                          stop      (bool, final frame)
# metadataEvent            stopReason (str, e.g. "END_TURN")  metadata
#                                                             (+ last_stop_reason)
# contextUsageEvent        contextUsagePercentage (float)     context_usage
# meteringEvent            usage       (float, e.g. 0.0453)   usage (numeric,
#                          unit        (str, "credit")         unchanged shape)
#                          unitPlural  (str, "credits")       + last_metering
#
# Anything not listed is surfaced verbatim as "unknown_event" (logged once per
# unseen `:event-type` at DEBUG), and frames whose `:message-type` is not
# `event` are surfaced as "exception". See _route_frame for that extension point.
#
# Alias policy: camelCase names above are REAL (observed on the wire). The
# snake_case entries are PLAUSIBLE-ONLY variants kept because the Rust client
# uses snake_case internally; they cost nothing and are marked as such. The
# previously-listed `messageMetadata` / `messageMetadataEvent` /
# `message_metadata_event` names were PHANTOMS (no such Smithy member exists)
# and have been removed.
FRAME_EVENT_TYPE_MAP: Dict[str, str] = {
    # --- REAL, observed on the wire ---
    "assistantResponseEvent": "content",
    "reasoningContentEvent": "reasoning",
    "toolUseEvent": "tool_use",
    "metadataEvent": "metadata",
    "contextUsageEvent": "metadata",
    "meteringEvent": "metadata",
    # --- REAL but not reproduced in the six captured scenarios ---
    "followupPrompt": "followup",
    "followupPromptEvent": "followup",
    # --- PLAUSIBLE-ONLY snake_case variants (never observed) ---
    "assistant_response_event": "content",
    "reasoning_content_event": "reasoning",
    "tool_use_event": "tool_use",
    "metadata_event": "metadata",
    "context_usage_event": "metadata",
    "metering_event": "metadata",
    "followup_prompt_event": "followup",
}


class EventStreamRoutingParser:
    """
    Stream parser that decodes real AWS event-stream frames and routes them on
    the ``:event-type`` header, with the legacy prefix-scraping parser as an
    always-live automatic fallback.

    Interface-compatible with :class:`AwsEventStreamParser` (``feed``,
    ``get_tool_calls``, ``reset``), so ``streaming_core`` can use either.

    Two modes, decided from the first bytes of the stream and logged once:

    - ``framed``: the head of the stream is a CRC-valid event-stream prelude.
      Frames are reassembled across chunk boundaries, both CRC32s are checked,
      and payloads are routed by header.
    - ``legacy``: the stream is not framed (or ``EVENTSTREAM_DECODER=false``, or
      framing turned out to be corrupt mid-stream). Every buffered byte is
      handed to :class:`AwsEventStreamParser` unchanged.

    Tool-call accumulation is *always* delegated to the embedded legacy parser,
    so the dict-merge-vs-string-concat semantics from FIX-03 are shared verbatim
    by both paths.
    """

    def __init__(self, enabled: Optional[bool] = None):
        """
        Args:
            enabled: Force the decoder on/off. When None, reads
                ``kiro.config.EVENTSTREAM_DECODER`` (read at construction time so
                tests can patch the module attribute).
        """
        if enabled is None:
            from kiro import config as _config
            enabled = bool(getattr(_config, "EVENTSTREAM_DECODER", True))

        self._decoder_enabled = enabled
        self._legacy = AwsEventStreamParser()
        self._decoder: Optional[Any] = None
        self._pending_legacy: bytes = b""
        # None = undecided, "framed" / "legacy" once resolved.
        self.mode: Optional[str] = None
        self._logged_mode = False

        # --- additive, retrievable payload surface (no consumer changes) ------
        # Last `meteringEvent` seen, as {"usage": float, "unit": str,
        # "unitPlural": str}. Source for a future credit display.
        self.last_metering: Optional[Dict[str, Any]] = None
        # Last `metadataEvent.stopReason` (e.g. "END_TURN").
        self.last_stop_reason: Optional[str] = None
        # Last `assistantResponseEvent.modelId` (e.g. "claude-sonnet-4.5").
        self.last_model_id: Optional[str] = None
        # `:event-type` values already reported as unknown (log-once bookkeeping).
        self._seen_unknown_event_types: set = set()

        if enabled:
            from kiro.eventstream import EventStreamDecoder
            self._decoder = EventStreamDecoder()
        else:
            self._set_mode("legacy", "EVENTSTREAM_DECODER=false")

    # ------------------------------------------------------------------ state

    def _set_mode(self, mode: str, reason: str) -> None:
        """Record the chosen path and log it once per stream at DEBUG."""
        self.mode = mode
        if not self._logged_mode:
            self._logged_mode = True
            logger.debug(f"Kiro stream parse path: {mode} ({reason})")

    def _fall_back(self, reason: str, pending: bytes) -> None:
        """Switch permanently to the legacy parser, keeping undecoded bytes."""
        if self.mode != "legacy":
            if self._logged_mode:
                logger.warning(f"Kiro stream falling back to legacy parsing: {reason}")
            self._logged_mode = False
            self._set_mode("legacy", reason)
        self._decoder = None
        self._pending_legacy = pending

    # ------------------------------------------------------------------- feed

    def feed(self, chunk: bytes) -> List[Dict[str, Any]]:
        """
        Adds a chunk and returns parsed events.

        Args:
            chunk: Raw bytes from the stream.

        Returns:
            List of ``{"type": str, "data": Any}`` events, in the same vocabulary
            the legacy parser produces (plus the passthrough types documented on
            :meth:`_route_frame`).
        """
        if self.mode == "legacy" or self._decoder is None:
            pending = getattr(self, "_pending_legacy", b"")
            if pending:
                self._pending_legacy = b""
                chunk = pending + (chunk or b"")
            return self._legacy.feed(chunk or b"")

        self._decoder.feed(chunk or b"")

        # Resolve framing on the very first bytes. `None` means we do not have
        # enough bytes to tell yet - keep buffering rather than guessing.
        if self.mode is None:
            probe = self._decoder.looks_like_eventstream()
            if probe is None:
                return []
            if probe is False:
                self._fall_back_initial()
                pending = self._pending_legacy
                self._pending_legacy = b""
                return self._legacy.feed(pending)
            self._set_mode("framed", "valid event-stream prelude")

        events: List[Dict[str, Any]] = []
        while True:
            try:
                frame = self._decoder.next_frame()
            except Exception as e:
                # Corrupt framing. Everything still buffered is handed to the
                # legacy parser rather than failing the request.
                pending = self._decoder.take_buffer() if self._decoder else b""
                self._fall_back(f"frame decode error: {e}", b"")
                events.extend(self._legacy.feed(pending))
                return events

            if frame is None:
                break

            events.extend(self._route_frame(frame))

        return events

    def _fall_back_initial(self) -> None:
        """Stream is not framed at all: legacy from the first byte."""
        pending = self._decoder.take_buffer() if self._decoder else b""
        self._set_mode("legacy", "stream is not event-stream framed")
        self._decoder = None
        self._pending_legacy = pending

    def flush(self) -> List[Dict[str, Any]]:
        """
        Drains anything still buffered at end of stream.

        A stream shorter than a prelude never resolves its mode; on completion
        those bytes are given to the legacy parser so nothing is lost.
        """
        if self._decoder is None:
            pending = getattr(self, "_pending_legacy", b"")
            if pending:
                self._pending_legacy = b""
                return self._legacy.feed(pending)
            return []

        if self.mode is None:
            self._fall_back_initial()
            return self.flush()

        # Framed mode with leftover bytes: a truncated trailing frame. The real
        # client calls this "data left over in the event stream response stream".
        leftover = self._decoder.buffered_bytes
        if leftover:
            logger.warning(
                f"Event stream ended with {leftover} undecoded trailing byte(s) "
                "(truncated frame); discarding"
            )
            self._decoder.take_buffer()
        return []

    # ---------------------------------------------------------------- routing

    def _route_frame(self, frame: Any) -> List[Dict[str, Any]]:
        """
        Turns one decoded frame into zero or more internal events.

        EXTENSION POINT for exception / invalidState handling: the two branches
        below emit ``{"type": "exception", ...}`` for any frame whose
        ``:message-type`` is not ``event``, and ``{"type": "unknown_event", ...}``
        for any unmapped ``:event-type`` (invalidStateEvent, reasoningContentEvent,
        codeReferenceEvent, ...). Both carry the full ``headers`` dict, the raw
        ``payload`` bytes and the parsed ``data`` when the payload is JSON, so a
        follow-up change can act on them without touching the decoder or the
        routing table. ``streaming_core._process_chunk`` currently ignores both
        types, which keeps today's behaviour byte-identical.

        Args:
            frame: An :class:`kiro.eventstream.EventStreamFrame`.

        Returns:
            List of internal events.
        """
        message_type = frame.message_type
        event_type = frame.event_type
        data = self._parse_payload(frame)

        # --- exception / error frames (in-band) -------------------------------
        if (message_type is not None and message_type != "event") or frame.exception_type:
            return [{
                "type": "exception",
                "data": data,
                "headers": dict(frame.headers),
                "message_type": message_type,
                "exception_type": frame.exception_type,
                "payload": frame.payload,
            }]

        internal = FRAME_EVENT_TYPE_MAP.get(event_type or "")

        # --- unmapped event types --------------------------------------------
        if internal is None:
            key = event_type or "<missing>"
            if key not in self._seen_unknown_event_types:
                self._seen_unknown_event_types.add(key)
                # DEBUG only, and never the payload: reasoning signatures are
                # opaque blobs and payloads may contain user text.
                logger.debug(
                    f"Kiro stream: unmapped :event-type '{key}' "
                    "(passed through as unknown_event)"
                )
            return [{
                "type": "unknown_event",
                "data": data,
                "headers": dict(frame.headers),
                "event_type": event_type,
                "payload": frame.payload,
            }]

        if not isinstance(data, dict):
            logger.debug(f"Frame '{event_type}' payload is not a JSON object; ignoring")
            return []

        if internal == "content":
            return self._route_content(data)
        if internal == "reasoning":
            return self._route_reasoning(data)
        if internal == "tool_use":
            return self._route_tool_use(data)
        if internal == "metadata":
            return self._route_metadata(data, frame)
        if internal == "followup":
            return [{"type": "followup", "data": data.get("followupPrompt", data)}]

        return []

    @staticmethod
    def _parse_payload(frame: Any) -> Any:
        """Parse a frame payload as JSON, or return the decoded text on failure."""
        if not frame.payload:
            return None
        text = frame.payload_text()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.debug(f"Frame payload is not JSON: {text[:120]}")
            return text

    def _route_content(self, data: dict) -> List[Dict[str, Any]]:
        """assistantResponseEvent -> content (or followup)."""
        # `modelId` rides along on every frame; keep the last one retrievable.
        model_id = data.get("modelId")
        if model_id:
            self.last_model_id = model_id

        # A followup prompt can ride inside an assistant response payload.
        if data.get("followupPrompt"):
            return [{"type": "followup", "data": data["followupPrompt"]}]

        content = data.get("content")
        if not content:
            return []
        # NOTE: no adjacent-duplicate suppression here, unlike the legacy path.
        # That suppression exists only because prefix scraping could observe the
        # same JSON twice; framing delivers each payload exactly once, so
        # dropping a legitimately repeated token would lose real output.
        return [{"type": "content", "data": content}]

    def _route_reasoning(self, data: dict) -> List[Dict[str, Any]]:
        """
        reasoningContentEvent -> the existing internal ``thinking`` event.

        Non-Claude models (gpt-5.6-terra, ...) emit native reasoning here instead
        of inlining ``<thinking>`` tags in ``content``. Previously unmapped, so it
        was dropped. ``signature`` is an opaque upstream blob: it is carried on the
        event for future consumers and never logged.
        """
        text = data.get("text")
        if not text:
            return []
        event: Dict[str, Any] = {"type": "thinking", "data": text}
        signature = data.get("signature")
        if signature:
            event["signature"] = signature
        return [event]

    def _route_tool_use(self, data: dict) -> List[Dict[str, Any]]:
        """
        toolUseEvent -> the tool_start / tool_input / tool_stop sequence.

        Routed through the embedded legacy parser's handlers so the FIX-03
        accumulation semantics (dict merge vs string concat) are literally the
        same code on both paths. Those handlers return None by design - completed
        tool calls are collected via ``get_tool_calls()``.
        """
        # Upstream repeats `name` and `toolUseId` in EVERY toolUseEvent frame, and
        # streams `input` as successive partial-JSON string fragments. Routing on
        # the presence of `name` therefore re-started the tool call on every
        # fragment and discarded the accumulated arguments. Route on toolUseId
        # identity instead: a frame starts a new call only when no call is open or
        # when it belongs to a different toolUseId.
        tool_use_id = data.get("toolUseId")
        current = self._legacy.current_tool_call
        current_id = current.get("id") if current else None

        is_new_call = current is None or (
            bool(tool_use_id) and tool_use_id != current_id
        )

        if is_new_call and (data.get("name") or current is not None):
            # Start finalises any previous call and absorbs `input` / `stop`
            # carried in the same frame (name+input+stop is legal).
            self._legacy._process_tool_start_event(data)
            return []

        # Continuation of the open call: an `input` fragment is always accumulated,
        # even when `name` rides along. Empty strings are ignored by the handler.
        if "input" in data:
            self._legacy._process_tool_input_event(data)

        if data.get("stop"):
            self._legacy._process_tool_stop_event(data)

        return []

    def _route_metadata(self, data: dict, frame: Any) -> List[Dict[str, Any]]:
        """
        metadataEvent / contextUsageEvent / meteringEvent -> usage /
        context_usage / metadata passthrough.

        Additive only: ``usage`` keeps its exact numeric shape, while the credit
        unit names and the stop reason are recorded on the parser
        (:attr:`last_metering`, :attr:`last_stop_reason`) for future consumers.
        """
        events: List[Dict[str, Any]] = []

        if "usage" in data:
            usage = data.get("usage", 0)
            # meteringEvent credit fields: preserved so a future credit display
            # can render "0.0454 credits" without another upstream round trip.
            self.last_metering = {
                "usage": usage,
                "unit": data.get("unit"),
                "unitPlural": data.get("unitPlural"),
            }
            events.append({"type": "usage", "data": usage})
        if "contextUsagePercentage" in data:
            events.append({
                "type": "context_usage",
                "data": data.get("contextUsagePercentage", 0),
            })
        if data.get("stopReason"):
            self.last_stop_reason = data["stopReason"]
            # Also published on the per-turn ContextVar so the dialect formatters
            # (which never see this parser instance) can act on it.
            record_stream_stop_reason(data["stopReason"])

        if not events:
            # conversationId / messageId and friends. Ignored downstream today;
            # kept as a distinct type so it is available without re-plumbing.
            events.append({
                "type": "metadata",
                "data": data,
                "headers": dict(frame.headers),
            })

        return events

    # ----------------------------------------------------------- tool results

    def get_tool_calls(self) -> List[Dict[str, Any]]:
        """Returns all collected tool calls (delegated to the legacy accumulator)."""
        return self._legacy.get_tool_calls()

    def reset(self) -> None:
        """Resets all parser state."""
        self._legacy.reset()
        self._pending_legacy = b""
        self.mode = None
        self._logged_mode = False
        self.last_metering = None
        self.last_stop_reason = None
        self.last_model_id = None
        self._seen_unknown_event_types = set()
        if self._decoder_enabled:
            from kiro.eventstream import EventStreamDecoder
            self._decoder = EventStreamDecoder()
        else:
            self._decoder = None
            self._set_mode("legacy", "EVENTSTREAM_DECODER=false")

    # Convenience passthroughs so callers that poked at the legacy parser's
    # attributes keep working.
    @property
    def tool_calls(self) -> List[Dict[str, Any]]:
        return self._legacy.tool_calls

    @property
    def current_tool_call(self) -> Optional[Dict[str, Any]]:
        return self._legacy.current_tool_call
