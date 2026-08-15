"""
Complete upstream payload-surface tests.

Replays a realistic stream containing ALL SIX observed `:event-type` values
(assistantResponseEvent, reasoningContentEvent, toolUseEvent, metadataEvent,
contextUsageEvent, meteringEvent) and asserts that nothing is silently dropped:
content, tool arguments, context usage, credit metering, native reasoning,
stop reason and model id are all reachable, and an invented future event type
still routes to `unknown_event` (logged once, no raise).
"""

import json
import logging

import pytest

from kiro.eventstream import encode_frame
from kiro.parsers import FRAME_EVENT_TYPE_MAP, EventStreamRoutingParser


def event_frame(event_type: str, payload: dict) -> bytes:
    return encode_frame(
        {
            ":event-type": event_type,
            ":message-type": "event",
            ":content-type": "application/json",
        },
        json.dumps(payload).encode("utf-8"),
    )


TOOL_USE_ID = "tooluse_9RmTVSDLQ7yYQ0kOqz3Zbw"

FULL_STREAM = [
    ("assistantResponseEvent", {"content": "Let me add those.", "modelId": "gpt-5.6-terra"}),
    ("reasoningContentEvent", {"signature": ".KTR~~eyJhbGciOiJ", "text": "The user wants 2+3. "}),
    ("reasoningContentEvent", {"signature": ".KTR~~eyJhbGciOiJ", "text": "I will call add."}),
    ("toolUseEvent", {"name": "add", "toolUseId": TOOL_USE_ID, "input": '{"a"'}),
    ("toolUseEvent", {"name": "add", "toolUseId": TOOL_USE_ID, "input": ': 2, "b"'}),
    ("toolUseEvent", {"name": "add", "toolUseId": TOOL_USE_ID, "input": ': 3}'}),
    ("toolUseEvent", {"name": "add", "toolUseId": TOOL_USE_ID, "stop": True}),
    ("contextUsageEvent", {"contextUsagePercentage": 2.336}),
    ("meteringEvent", {
        "unit": "credit",
        "unitPlural": "credits",
        "usage": 0.045368714228855724,
    }),
    ("metadataEvent", {"stopReason": "END_TURN"}),
]


def replay(frames):
    parser = EventStreamRoutingParser(enabled=True)
    events = []
    for event_type, payload in frames:
        events.extend(parser.feed(event_frame(event_type, payload)))
    events.extend(parser.flush())
    return parser, events


def group(events):
    by_type = {}
    for ev in events:
        by_type.setdefault(ev["type"], []).append(ev)
    return by_type


class TestFullStreamReplay:
    def test_all_six_event_types_are_mapped(self):
        for event_type, _ in FULL_STREAM:
            assert event_type in FRAME_EVENT_TYPE_MAP, event_type

    def test_content_emitted(self):
        _, events = replay(FULL_STREAM)
        assert [e["data"] for e in group(events)["content"]] == ["Let me add those."]

    def test_tool_arguments_accumulate_to_valid_json(self):
        parser, _ = replay(FULL_STREAM)
        calls = parser.get_tool_calls()
        assert len(calls) == 1
        assert calls[0]["id"] == TOOL_USE_ID
        assert calls[0]["function"]["name"] == "add"
        assert json.loads(calls[0]["function"]["arguments"]) == {"a": 2, "b": 3}

    def test_context_usage_and_usage_both_present(self):
        _, events = replay(FULL_STREAM)
        by_type = group(events)
        assert by_type["context_usage"][0]["data"] == 2.336
        assert by_type["usage"][0]["data"] == 0.045368714228855724

    def test_native_reasoning_surfaces_as_thinking(self):
        _, events = replay(FULL_STREAM)
        thinking = group(events)["thinking"]
        assert [e["data"] for e in thinking] == ["The user wants 2+3. ", "I will call add."]
        assert thinking[0]["signature"] == ".KTR~~eyJhbGciOiJ"

    def test_no_unknown_or_exception_events(self):
        _, events = replay(FULL_STREAM)
        by_type = group(events)
        assert "unknown_event" not in by_type
        assert "exception" not in by_type


class TestCreditFields:
    def test_unit_unit_plural_and_usage_retrievable(self):
        parser, _ = replay(FULL_STREAM)
        assert parser.last_metering == {
            "usage": 0.045368714228855724,
            "unit": "credit",
            "unitPlural": "credits",
        }

    def test_numeric_usage_event_shape_unchanged(self):
        _, events = replay(FULL_STREAM)
        usage_events = group(events)["usage"]
        assert len(usage_events) == 1
        assert usage_events[0] == {"type": "usage", "data": 0.045368714228855724}

    def test_reset_clears_metering(self):
        parser, _ = replay(FULL_STREAM)
        parser.reset()
        assert parser.last_metering is None


class TestStopReasonAndModelId:
    def test_retrievable(self):
        parser, _ = replay(FULL_STREAM)
        assert parser.last_stop_reason == "END_TURN"
        assert parser.last_model_id == "gpt-5.6-terra"

    def test_reset_clears_them(self):
        parser, _ = replay(FULL_STREAM)
        parser.reset()
        assert parser.last_stop_reason is None
        assert parser.last_model_id is None


class TestFutureEventTypes:
    @staticmethod
    def _capture(frames, level="DEBUG"):
        from loguru import logger as _logger

        records = []
        sink_id = _logger.add(lambda m: records.append(m.record), level=level)
        try:
            parser, events = replay(frames)
        finally:
            _logger.remove(sink_id)
        return parser, events, records

    def test_unknown_event_passthrough_logged_once(self):
        frames = [
            ("someFutureEvent", {"whatever": 1}),
            ("someFutureEvent", {"whatever": 2}),
        ]
        _, events, records = self._capture(frames)

        unknown = group(events)["unknown_event"]
        assert len(unknown) == 2
        assert all(e["event_type"] == "someFutureEvent" for e in unknown)

        mentions = [r for r in records if "someFutureEvent" in r["message"]]
        assert len(mentions) == 1
        assert mentions[0]["level"].name == "DEBUG"

    def test_payload_never_logged_above_debug(self):
        _, _, records = self._capture([("someFutureEvent", {"secret": "user text"})])
        for record in records:
            if record["level"].no >= 20:  # INFO and above
                assert "user text" not in record["message"]


class TestRegistryHygiene:
    @pytest.mark.parametrize("phantom", [
        "messageMetadata",
        "messageMetadataEvent",
        "message_metadata_event",
    ])
    def test_phantom_names_removed(self, phantom):
        assert phantom not in FRAME_EVENT_TYPE_MAP



class TestNativeReasoningEndToEnd:
    @pytest.mark.asyncio
    async def test_reasoning_reaches_the_thinking_consumer(self, monkeypatch):
        """Native reasoning arrives as a KiroEvent(type='thinking')."""
        from unittest.mock import AsyncMock

        import kiro.streaming_core as sc

        monkeypatch.setattr(sc, "FAKE_REASONING_ENABLED", False)

        blob = b"".join(event_frame(t, p) for t, p in FULL_STREAM)

        async def aiter_bytes():
            for i in range(0, len(blob), 13):
                yield blob[i:i + 13]

        response = AsyncMock()
        response.aiter_bytes = aiter_bytes

        events = [e async for e in sc.parse_kiro_stream(response, first_token_timeout=30)]

        thinking = [e.thinking_content for e in events if e.type == "thinking"]
        assert thinking == ["The user wants 2+3. ", "I will call add."]
        assert "".join(e.content for e in events if e.type == "content") == "Let me add those."
