"""
Regression tests for the real upstream `:event-type` names.

Captured from live upstream frames for the prompt "Reply with exactly: OK":
the wire sends `assistantResponseEvent`, `metadataEvent`, `contextUsageEvent`
and `meteringEvent`. The decoder previously only mapped `messageMetadata*`,
so metering/context-usage events were discarded and the truncation heuristic
in streaming_openai misfired on essentially every tool-free response.
"""

import json

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


CAPTURED_FRAMES = [
    ("assistantResponseEvent", {"content": "OK", "modelId": "claude-sonnet-4.5"}),
    ("metadataEvent", {"stopReason": "END_TURN"}),
    ("contextUsageEvent", {"contextUsagePercentage": 2.245499849319458}),
    ("meteringEvent", {
        "unit": "credit",
        "unitPlural": "credits",
        "usage": 0.018954083648424543,
    }),
]


def feed_all(frames):
    parser = EventStreamRoutingParser(enabled=True)
    events = []
    for event_type, payload in frames:
        events.extend(parser.feed(event_frame(event_type, payload)))
    return parser, events


class TestCapturedFrameReplay:
    def test_replay_emits_content_context_usage_and_usage(self):
        _, events = feed_all(CAPTURED_FRAMES)

        by_type = {}
        for ev in events:
            by_type.setdefault(ev["type"], []).append(ev)

        assert [e["data"] for e in by_type["content"]] == ["OK"]
        assert by_type["context_usage"][0]["data"] == 2.245499849319458
        assert by_type["usage"][0]["data"] == 0.018954083648424543

    def test_no_unknown_or_exception_events(self):
        _, events = feed_all(CAPTURED_FRAMES)
        assert not [e for e in events if e["type"] in ("unknown_event", "exception")]

    def test_metadata_event_surfaces_stop_reason(self):
        _, events = feed_all([CAPTURED_FRAMES[1]])
        assert len(events) == 1
        assert events[0]["type"] == "metadata"
        assert events[0]["data"]["stopReason"] == "END_TURN"


class TestTruncationGuard:
    def test_stream_completion_signals_are_both_present(self):
        """
        streaming_openai computes
        `stream_completed_normally = received_usage or received_context_usage`.
        Both signals must be emitted so truncation cannot misfire.
        """
        _, events = feed_all(CAPTURED_FRAMES)
        types = {e["type"] for e in events}
        received_usage = "usage" in types
        received_context_usage = "context_usage" in types
        assert received_usage and received_context_usage
        assert (received_usage or received_context_usage) is True


class TestPhantomAliases:
    @pytest.mark.parametrize("name", [
        "messageMetadata",
        "messageMetadataEvent",
        "message_metadata_event",
    ])
    def test_phantom_names_are_no_longer_carried(self, name):
        # These Smithy members do not exist on the wire; the registry no longer
        # pretends they do. Unknown names still pass through harmlessly.
        assert name not in FRAME_EVENT_TYPE_MAP
        _, events = feed_all([(name, {"usage": 1.5})])
        assert [e["type"] for e in events] == ["unknown_event"]

    def test_real_names_are_mapped(self):
        for name in ("metadataEvent", "contextUsageEvent", "meteringEvent"):
            assert FRAME_EVENT_TYPE_MAP[name] == "metadata"
