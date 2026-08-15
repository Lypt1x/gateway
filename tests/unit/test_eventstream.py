# -*- coding: utf-8 -*-

"""
Tests for the real AWS event-stream frame decoder (RECOMMENDATION 1).

All offline: frames are hand-constructed, no network and no live API calls.
Covers the decoder itself (kiro/eventstream.py) and the routing/fallback parser
(kiro.parsers.EventStreamRoutingParser).
"""

import json
import struct
import zlib

import pytest

from kiro.eventstream import (
    MAX_FRAME_SIZE,
    PRELUDE_LENGTH,
    EventStreamDecoder,
    EventStreamError,
    encode_frame,
)
from kiro.parsers import AwsEventStreamParser, EventStreamRoutingParser


# ==================================================================================================
# Helpers
# ==================================================================================================

def event_frame(event_type: str, payload: dict) -> bytes:
    """Build a normal `:message-type: event` frame carrying a JSON payload."""
    return encode_frame(
        {
            ":event-type": event_type,
            ":message-type": "event",
            ":content-type": "application/json",
        },
        json.dumps(payload).encode("utf-8"),
    )


def corrupt_prelude_crc(frame: bytes) -> bytes:
    """Flip the prelude CRC of an otherwise valid frame."""
    bad = bytearray(frame)
    bad[8:12] = struct.pack(">I", 0xDEADBEEF)
    return bytes(bad)


def corrupt_message_crc(frame: bytes) -> bytes:
    """Flip the trailing message CRC of an otherwise valid frame."""
    bad = bytearray(frame)
    bad[-4:] = struct.pack(">I", 0x01020304)
    return bytes(bad)


# ==================================================================================================
# Decoder: happy paths
# ==================================================================================================

class TestFrameDecoding:

    def test_valid_frame_decodes_headers_and_payload(self):
        """
        What it does: Decodes one hand-constructed frame.
        Goal: Headers and payload bytes come back exactly.
        """
        frame = event_frame("assistantResponseEvent", {"content": "Hello"})

        decoder = EventStreamDecoder()
        decoder.feed(frame)
        decoded = decoder.next_frame()

        assert decoded is not None
        assert decoded.headers[":event-type"] == "assistantResponseEvent"
        assert decoded.headers[":message-type"] == "event"
        assert decoded.headers[":content-type"] == "application/json"
        assert decoded.event_type == "assistantResponseEvent"
        assert json.loads(decoded.payload_text()) == {"content": "Hello"}
        assert decoder.buffered_bytes == 0
        # Buffer drained: no second frame.
        assert decoder.next_frame() is None

    def test_frame_split_across_three_chunks_reassembles(self):
        """
        What it does: Feeds one frame as three arbitrary slices.
        Goal: Nothing decodes early, and the frame reassembles intact.
        """
        frame = event_frame("assistantResponseEvent", {"content": "split me up"})
        # Cut inside the prelude, then inside the headers, then the tail.
        parts = [frame[:5], frame[5:PRELUDE_LENGTH + 3], frame[PRELUDE_LENGTH + 3:]]
        assert all(parts)

        decoder = EventStreamDecoder()
        decoder.feed(parts[0])
        assert decoder.next_frame() is None
        decoder.feed(parts[1])
        assert decoder.next_frame() is None
        decoder.feed(parts[2])

        decoded = decoder.next_frame()
        assert decoded is not None
        assert json.loads(decoded.payload_text())["content"] == "split me up"

    def test_multiple_frames_in_one_chunk_all_decode(self):
        """
        What it does: Feeds three concatenated frames in a single chunk.
        Goal: All three decode, in order.
        """
        blob = b"".join(
            event_frame("assistantResponseEvent", {"content": text})
            for text in ("a", "bb", "ccc")
        )

        decoder = EventStreamDecoder()
        decoder.feed(blob)

        contents = []
        while True:
            frame = decoder.next_frame()
            if frame is None:
                break
            contents.append(json.loads(frame.payload_text())["content"])

        assert contents == ["a", "bb", "ccc"]
        assert decoder.frames_decoded == 3

    def test_zero_length_payload_and_headers(self):
        """
        What it does: Decodes a frame with no headers and no payload.
        Goal: The 16-byte minimum frame is valid, not treated as corruption.
        """
        frame = encode_frame({}, b"")
        assert len(frame) == 16

        decoder = EventStreamDecoder()
        decoder.feed(frame)
        decoded = decoder.next_frame()

        assert decoded is not None
        assert decoded.headers == {}
        assert decoded.payload == b""
        assert decoded.event_type is None

    def test_non_string_header_values_round_trip(self):
        """
        What it does: Encodes bool/int/bytes headers.
        Goal: All supported header value types decode back.
        """
        frame = encode_frame({"b": True, "f": False, "n": 42, "raw": b"\x00\xff"}, b"x")

        decoder = EventStreamDecoder()
        decoder.feed(frame)
        decoded = decoder.next_frame()

        assert decoded.headers == {"b": True, "f": False, "n": 42, "raw": b"\x00\xff"}

    def test_utf8_payload_split_mid_character(self):
        """
        What it does: Splits a frame in the middle of a multi-byte character.
        Goal: No character corruption - framing resolves before any decoding.
        """
        text = "\u4f60\u597d\u4e16\u754c"
        frame = event_frame("assistantResponseEvent", {"content": text})

        decoder = EventStreamDecoder()
        for i in range(0, len(frame), 3):
            decoder.feed(frame[i:i + 3])
        decoded = decoder.next_frame()

        assert json.loads(decoded.payload_text())["content"] == text


# ==================================================================================================
# Decoder: rejection and robustness
# ==================================================================================================

class TestFrameRejection:

    def test_bad_prelude_crc_is_rejected(self):
        """
        What it does: Feeds a frame with a wrong prelude CRC32.
        Goal: EventStreamError, no crash, buffer left intact for fallback.
        """
        frame = corrupt_prelude_crc(event_frame("assistantResponseEvent", {"content": "x"}))

        decoder = EventStreamDecoder()
        decoder.feed(frame)

        assert decoder.looks_like_eventstream() is False
        with pytest.raises(EventStreamError, match="prelude CRC32"):
            decoder.next_frame()
        assert decoder.buffered_bytes == len(frame)

    def test_bad_message_crc_is_rejected(self):
        """
        What it does: Feeds a frame with a valid prelude but wrong message CRC32.
        Goal: EventStreamError after framing looked plausible; bytes preserved.
        """
        frame = corrupt_message_crc(event_frame("assistantResponseEvent", {"content": "x"}))

        decoder = EventStreamDecoder()
        decoder.feed(frame)

        assert decoder.looks_like_eventstream() is True
        with pytest.raises(EventStreamError, match="message CRC32"):
            decoder.next_frame()
        assert decoder.buffered_bytes == len(frame)

    def test_oversized_total_length_rejected_without_buffering(self):
        """
        What it does: Crafts a CRC-valid prelude claiming a 1 GB frame.
        Goal: Rejected on the length check, never waited for.
        """
        prelude = struct.pack(">II", MAX_FRAME_SIZE + 1, 0)
        prelude += struct.pack(">I", zlib.crc32(prelude) & 0xFFFFFFFF)

        decoder = EventStreamDecoder()
        decoder.feed(prelude)

        with pytest.raises(EventStreamError, match="implausible total length"):
            decoder.next_frame()

    def test_undersized_total_length_rejected(self):
        """
        What it does: Claims a total length below the 16-byte minimum.
        Goal: Rejected rather than producing negative slice arithmetic.
        """
        prelude = struct.pack(">II", 8, 0)
        prelude += struct.pack(">I", zlib.crc32(prelude) & 0xFFFFFFFF)

        decoder = EventStreamDecoder()
        decoder.feed(prelude)

        with pytest.raises(EventStreamError, match="implausible total length"):
            decoder.next_frame()

    def test_headers_longer_than_frame_rejected(self):
        """
        What it does: Claims a headers block larger than the frame body.
        Goal: Rejected on the headers-length check.
        """
        prelude = struct.pack(">II", 32, 100)
        prelude += struct.pack(">I", zlib.crc32(prelude) & 0xFFFFFFFF)

        decoder = EventStreamDecoder()
        decoder.feed(prelude)

        with pytest.raises(EventStreamError, match="implausible headers length"):
            decoder.next_frame()

    def test_malformed_header_block_rejected(self):
        """
        What it does: Builds a CRC-valid frame whose header block is garbage.
        Goal: Header decoding raises rather than looping or mis-parsing.
        """
        # name_len 5 but only 1 byte follows
        bad_headers = bytes([5]) + b"a"
        total_len = 16 + len(bad_headers)
        prelude = struct.pack(">II", total_len, len(bad_headers))
        prelude += struct.pack(">I", zlib.crc32(prelude) & 0xFFFFFFFF)
        body = prelude + bad_headers
        frame = body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

        decoder = EventStreamDecoder()
        decoder.feed(frame)

        with pytest.raises(EventStreamError):
            decoder.next_frame()

    def test_truncated_trailing_bytes_do_not_hang(self):
        """
        What it does: Feeds one whole frame plus half of the next, repeatedly polling.
        Goal: next_frame() returns None promptly and forever; no infinite loop.
        """
        good = event_frame("assistantResponseEvent", {"content": "done"})
        tail = event_frame("assistantResponseEvent", {"content": "never finished"})[:20]

        decoder = EventStreamDecoder()
        decoder.feed(good + tail)

        assert decoder.next_frame() is not None
        for _ in range(50):
            assert decoder.next_frame() is None
        assert decoder.buffered_bytes == len(tail)

    def test_empty_and_short_feeds_are_undecided(self):
        """
        What it does: Feeds fewer bytes than a prelude.
        Goal: looks_like_eventstream() answers None (unknown), not False.
        """
        decoder = EventStreamDecoder()
        assert decoder.looks_like_eventstream() is None
        decoder.feed(b"")
        assert decoder.next_frame() is None
        decoder.feed(b"{\"cont")
        assert decoder.looks_like_eventstream() is None
        assert decoder.next_frame() is None


# ==================================================================================================
# Routing parser: framed path
# ==================================================================================================

class TestFramedRouting:

    def test_assistant_response_event_routes_to_content(self):
        """
        What it does: Feeds framed assistantResponseEvent frames.
        Goal: 'content' events in the existing internal vocabulary.
        """
        parser = EventStreamRoutingParser(enabled=True)
        blob = event_frame("assistantResponseEvent", {"content": "Hello"}) + \
            event_frame("assistant_response_event", {"content": " World"})

        events = parser.feed(blob)

        assert parser.mode == "framed"
        assert events == [
            {"type": "content", "data": "Hello"},
            {"type": "content", "data": " World"},
        ]

    def test_message_metadata_routes_to_usage_and_context_usage(self):
        """
        What it does: Feeds the real metering/context/metadata frames.
        Goal: 'usage' and 'context_usage' events, plus metadata passthrough.
        """
        parser = EventStreamRoutingParser(enabled=True)

        events = parser.feed(
            event_frame("meteringEvent", {"usage": 7})
            + event_frame("contextUsageEvent", {"contextUsagePercentage": 42.5})
            + event_frame("metadataEvent", {"conversationId": "c1", "messageId": "m1"})
        )

        assert {"type": "usage", "data": 7} in events
        assert {"type": "context_usage", "data": 42.5} in events
        metadata = [e for e in events if e["type"] == "metadata"]
        assert metadata and metadata[0]["data"]["conversationId"] == "c1"

    def test_followup_prompt_routes_to_followup(self):
        """
        What it does: Feeds a followupPrompt frame.
        Goal: A 'followup' event, and no content leakage.
        """
        parser = EventStreamRoutingParser(enabled=True)

        events = parser.feed(
            event_frame("followupPrompt", {"followupPrompt": {"content": "Ask me more"}})
        )

        assert events == [{"type": "followup", "data": {"content": "Ask me more"}}]

    def test_exception_frame_is_exposed_not_dropped(self):
        """
        What it does: Feeds an in-band exception frame.
        Goal: Exposed as a distinct 'exception' event with headers intact,
              which is the extension point for the follow-up error work.
        """
        parser = EventStreamRoutingParser(enabled=True)
        frame = encode_frame(
            {
                ":message-type": "exception",
                ":exception-type": "ThrottlingError",
                ":content-type": "application/json",
            },
            json.dumps({"message": "slow down", "reason": "RATE_LIMIT_EXCEEDED"}).encode(),
        )

        events = parser.feed(frame)

        assert len(events) == 1
        assert events[0]["type"] == "exception"
        assert events[0]["exception_type"] == "ThrottlingError"
        assert events[0]["data"]["reason"] == "RATE_LIMIT_EXCEEDED"
        assert events[0]["headers"][":message-type"] == "exception"

    def test_unknown_event_type_is_exposed_not_dropped(self):
        """
        What it does: Feeds invalidStateEvent and reasoningContentEvent frames.
        Goal: Exposed as 'unknown_event' passthroughs carrying event_type.
        """
        parser = EventStreamRoutingParser(enabled=True)

        events = parser.feed(
            event_frame("invalidStateEvent", {"reason": "INVALID_STATE", "message": "bad"})
            + event_frame("reasoningContent", {"content": "thinking..."})
        )

        assert [e["type"] for e in events] == ["unknown_event", "unknown_event"]
        assert events[0]["event_type"] == "invalidStateEvent"
        assert events[1]["event_type"] == "reasoningContent"

    def test_framed_stream_split_across_chunks_yields_all_content(self):
        """
        What it does: Streams several frames in 7-byte chunks.
        Goal: Full text reassembles in order.
        """
        parser = EventStreamRoutingParser(enabled=True)
        blob = b"".join(
            event_frame("assistantResponseEvent", {"content": word})
            for word in ("The ", "quick ", "brown ", "fox")
        )

        text = ""
        for i in range(0, len(blob), 7):
            for event in parser.feed(blob[i:i + 7]):
                if event["type"] == "content":
                    text += event["data"]

        assert text == "The quick brown fox"
        assert parser.flush() == []


# ==================================================================================================
# Routing parser: tool use
# ==================================================================================================

TOOL_SEQUENCE_PAYLOADS = [
    {"toolUseId": "tu-1", "name": "get_weather", "input": ""},
    {"toolUseId": "tu-1", "input": '{"city": "Lo'},
    {"toolUseId": "tu-1", "input": 'ndon", "unit": "c"}'},
    {"toolUseId": "tu-1", "stop": True},
]


class TestToolUseRouting:

    def test_tool_sequence_accumulates_identically_to_legacy(self):
        """
        What it does: Runs the same toolUseEvent sequence through the framed path
                      and through the legacy prefix path.
        Goal: Byte-identical accumulated arguments - FIX-03 semantics unchanged.
        """
        framed = EventStreamRoutingParser(enabled=True)
        framed.feed(b"".join(event_frame("toolUseEvent", p) for p in TOOL_SEQUENCE_PAYLOADS))
        framed_calls = framed.get_tool_calls()

        legacy = AwsEventStreamParser()
        # The legacy parser discriminates on the FIRST json key, so the legacy
        # byte stream is written with the discriminating key first - which is the
        # ordering the live wire actually uses today.
        legacy_stream = b"".join(
            json.dumps(payload).encode()
            for payload in (
                {"name": "get_weather", "toolUseId": "tu-1", "input": ""},
                {"input": '{"city": "Lo'},
                {"input": 'ndon", "unit": "c"}'},
                {"stop": True},
            )
        )
        legacy.feed(legacy_stream)
        legacy_calls = legacy.get_tool_calls()

        assert framed_calls == legacy_calls
        assert len(framed_calls) == 1
        assert framed_calls[0]["id"] == "tu-1"
        assert framed_calls[0]["function"]["name"] == "get_weather"
        assert json.loads(framed_calls[0]["function"]["arguments"]) == {
            "city": "London", "unit": "c"
        }

    def test_dict_input_fragments_are_merged_not_concatenated(self):
        """
        What it does: Sends dict-shaped input fragments (FIX-03 dict mode).
        Goal: Fragments merge into one object instead of string-concatenating.
        """
        parser = EventStreamRoutingParser(enabled=True)
        parser.feed(
            event_frame("toolUseEvent", {"toolUseId": "tu-2", "name": "f", "input": {}})
            + event_frame("toolUseEvent", {"toolUseId": "tu-2", "input": {"a": 1}})
            + event_frame("toolUseEvent", {"toolUseId": "tu-2", "input": {"b": 2}})
            + event_frame("toolUseEvent", {"toolUseId": "tu-2", "stop": True})
        )

        calls = parser.get_tool_calls()
        assert json.loads(calls[0]["function"]["arguments"]) == {"a": 1, "b": 2}

    def test_tool_use_input_and_stop_in_one_frame(self):
        """
        What it does: Terminates a tool call with input+stop in the same frame.
        Goal: The call finalizes with complete arguments.
        """
        parser = EventStreamRoutingParser(enabled=True)
        parser.feed(
            event_frame("toolUseEvent", {"toolUseId": "tu-3", "name": "g", "input": '{"x":'})
            + event_frame("toolUseEvent", {"toolUseId": "tu-3", "input": "1}", "stop": True})
        )

        calls = parser.get_tool_calls()
        assert len(calls) == 1
        assert json.loads(calls[0]["function"]["arguments"]) == {"x": 1}

    def test_tool_use_emits_no_streamed_events(self):
        """
        What it does: Checks the event list from a tool sequence.
        Goal: tool_* handling stays side-effect-only, as on the legacy path.
        """
        parser = EventStreamRoutingParser(enabled=True)
        events = parser.feed(
            b"".join(event_frame("toolUseEvent", p) for p in TOOL_SEQUENCE_PAYLOADS)
        )
        assert events == []


# ==================================================================================================
# Automatic fallback
# ==================================================================================================

LEGACY_JSON_STREAM = (
    b'\x00\x00\x01{"content":"Hello"}'
    b'garbage{"content":" World"}'
    b'{"usage":3}'
    b'{"contextUsagePercentage":12.5}'
)


class TestAutomaticFallback:

    def test_non_framed_stream_falls_back_and_matches_legacy_exactly(self):
        """
        What it does: Feeds a raw JSON (non-framed) byte stream.
        Goal: Falls back to prefix parsing and yields events identical to today's.
        """
        parser = EventStreamRoutingParser(enabled=True)
        new_events = parser.feed(LEGACY_JSON_STREAM)

        legacy = AwsEventStreamParser()
        legacy_events = legacy.feed(LEGACY_JSON_STREAM)

        assert parser.mode == "legacy"
        assert new_events == legacy_events
        assert new_events == [
            {"type": "content", "data": "Hello"},
            {"type": "content", "data": " World"},
            {"type": "usage", "data": 3},
            {"type": "context_usage", "data": 12.5},
        ]

    def test_fallback_survives_chunk_splitting(self):
        """
        What it does: Feeds the same non-framed stream in 5-byte chunks.
        Goal: Identical events to feeding the legacy parser the same chunks.
        """
        chunks = [LEGACY_JSON_STREAM[i:i + 5] for i in range(0, len(LEGACY_JSON_STREAM), 5)]

        parser = EventStreamRoutingParser(enabled=True)
        legacy = AwsEventStreamParser()
        new_events, legacy_events = [], []
        for chunk in chunks:
            new_events.extend(parser.feed(chunk))
            legacy_events.extend(legacy.feed(chunk))

        assert new_events == legacy_events
        assert [e["data"] for e in new_events if e["type"] == "content"] == ["Hello", " World"]

    def test_env_flag_false_forces_legacy(self, monkeypatch):
        """
        What it does: Sets EVENTSTREAM_DECODER=false via the config module.
        Goal: Legacy path chosen from the first byte, even for framed input.
        """
        monkeypatch.setattr("kiro.config.EVENTSTREAM_DECODER", False)

        parser = EventStreamRoutingParser()
        assert parser.mode == "legacy"

        # Framed bytes: the legacy parser still scrapes the JSON payload out,
        # which is exactly today's behaviour.
        events = parser.feed(event_frame("assistantResponseEvent", {"content": "Hi"}))
        assert parser.mode == "legacy"
        assert events == [{"type": "content", "data": "Hi"}]

    def test_env_flag_true_by_default(self, monkeypatch):
        """
        What it does: Sets EVENTSTREAM_DECODER=True via the config module.
        Goal: Decoder path is used for framed input.
        """
        monkeypatch.setattr("kiro.config.EVENTSTREAM_DECODER", True)

        parser = EventStreamRoutingParser()
        parser.feed(event_frame("assistantResponseEvent", {"content": "Hi"}))
        assert parser.mode == "framed"

    def test_config_default_is_decode(self):
        """
        What it does: Reads the shipped default.
        Goal: EVENTSTREAM_DECODER defaults to True.
        """
        from kiro import config
        assert config.EVENTSTREAM_DECODER is True

    def test_midstream_corruption_falls_back_without_failing(self):
        """
        What it does: Sends one valid frame, then corrupt framing.
        Goal: Valid frame is kept, parser degrades to legacy, no exception.
        """
        parser = EventStreamRoutingParser(enabled=True)

        events = parser.feed(event_frame("assistantResponseEvent", {"content": "good"}))
        assert events == [{"type": "content", "data": "good"}]
        assert parser.mode == "framed"

        events = parser.feed(
            corrupt_message_crc(event_frame("assistantResponseEvent", {"content": "bad"}))
        )
        assert parser.mode == "legacy"
        # The bytes were handed to the legacy parser, which scrapes the payload.
        assert {"type": "content", "data": "bad"} in events

    def test_short_stream_flush_falls_back(self):
        """
        What it does: Sends a stream shorter than a prelude, then flushes.
        Goal: The bytes are not lost - flush() drains them through legacy.
        """
        parser = EventStreamRoutingParser(enabled=True)

        assert parser.feed(b'{"content":') == []
        assert parser.mode is None

        events = parser.flush()
        assert parser.mode == "legacy"
        # Incomplete JSON: legacy parser buffers it and emits nothing. The point
        # is that flush() resolves the mode instead of hanging undecided.
        assert events == []

    def test_flush_discards_truncated_trailing_frame(self):
        """
        What it does: Ends a framed stream mid-frame.
        Goal: flush() reports and drops the tail without raising.
        """
        parser = EventStreamRoutingParser(enabled=True)
        good = event_frame("assistantResponseEvent", {"content": "ok"})
        tail = event_frame("assistantResponseEvent", {"content": "cut"})[:18]

        assert parser.feed(good + tail) == [{"type": "content", "data": "ok"}]
        assert parser.flush() == []

    def test_reset_restores_initial_state(self):
        """
        What it does: Resets after a framed stream.
        Goal: Mode is undecided again and tool calls are cleared.
        """
        parser = EventStreamRoutingParser(enabled=True)
        parser.feed(b"".join(event_frame("toolUseEvent", p) for p in TOOL_SEQUENCE_PAYLOADS))
        assert parser.mode == "framed"

        parser.reset()
        assert parser.mode is None
        assert parser.get_tool_calls() == []


# ==================================================================================================
# streaming_core integration
# ==================================================================================================

class TestStreamingCoreIntegration:

    @pytest.mark.asyncio
    async def test_parse_kiro_stream_decodes_framed_response(self, monkeypatch):
        """
        What it does: Drives parse_kiro_stream with a framed byte stream.
        Goal: End-to-end content/usage events without touching converters.
        """
        from unittest.mock import AsyncMock

        import kiro.streaming_core as sc

        monkeypatch.setattr(sc, "FAKE_REASONING_ENABLED", False)

        blob = (
            event_frame("assistantResponseEvent", {"content": "Hello"})
            + event_frame("assistantResponseEvent", {"content": " World"})
            + event_frame("contextUsageEvent", {"contextUsagePercentage": 5.0})
        )

        async def aiter_bytes():
            for i in range(0, len(blob), 11):
                yield blob[i:i + 11]

        response = AsyncMock()
        response.aiter_bytes = aiter_bytes

        events = [e async for e in sc.parse_kiro_stream(response, first_token_timeout=30)]

        assert "".join(e.content for e in events if e.type == "content") == "Hello World"
        assert [e.context_usage_percentage for e in events if e.type == "context_usage"] == [5.0]

    @pytest.mark.asyncio
    async def test_parse_kiro_stream_falls_back_for_raw_json(self, monkeypatch):
        """
        What it does: Drives parse_kiro_stream with a non-framed JSON stream.
        Goal: Fallback path produces the same events it does today.
        """
        from unittest.mock import AsyncMock

        import kiro.streaming_core as sc

        monkeypatch.setattr(sc, "FAKE_REASONING_ENABLED", False)

        async def aiter_bytes():
            yield LEGACY_JSON_STREAM

        response = AsyncMock()
        response.aiter_bytes = aiter_bytes

        events = [e async for e in sc.parse_kiro_stream(response, first_token_timeout=30)]

        assert "".join(e.content for e in events if e.type == "content") == "Hello World"
        assert [e.usage for e in events if e.type == "usage"] == [3]
