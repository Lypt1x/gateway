# -*- coding: utf-8 -*-

"""
Regression tests for toolUseEvent routing in EventStreamRoutingParser.

Upstream repeats `name` and `toolUseId` in every toolUseEvent frame and streams
`input` as successive partial-JSON string fragments. Routing on the presence of
`name` restarted the tool call on every frame and dropped the accumulated
arguments. These tests replay real captured frames; fully offline.
"""

import json

import pytest

from kiro.eventstream import encode_frame
from kiro.parsers import AwsEventStreamParser, EventStreamRoutingParser


def event_frame(payload: dict, event_type: str = "toolUseEvent") -> bytes:
    return encode_frame(
        {
            ":event-type": event_type,
            ":message-type": "event",
            ":content-type": "application/json",
        },
        json.dumps(payload).encode("utf-8"),
    )


TID = "tooluse_Rx2u53SvACRt9rSCUjw0HA"

# Exact captured wire sequence for add(a=17, b=25).
CAPTURED_FRAMES = [
    {"name": "add", "toolUseId": TID},
    {"input": "", "name": "add", "toolUseId": TID},
    {"input": '{"', "name": "add", "toolUseId": TID},
    {"input": 'a":', "name": "add", "toolUseId": TID},
    {"input": " 17", "name": "add", "toolUseId": TID},
    {"input": ', "b"', "name": "add", "toolUseId": TID},
    {"input": ": 25}", "name": "add", "toolUseId": TID},
    {"name": "add", "stop": True, "toolUseId": TID},
]


class TestRepeatedNameFragments:

    def test_captured_sequence_rebuilds_full_arguments(self):
        """Acceptance criterion: the 8 captured frames rebuild {"a": 17, "b": 25}."""
        parser = EventStreamRoutingParser(enabled=True)
        parser.feed(b"".join(event_frame(p) for p in CAPTURED_FRAMES))

        calls = parser.get_tool_calls()
        assert len(calls) == 1
        assert calls[0]["id"] == TID
        assert calls[0]["function"]["name"] == "add"
        assert json.loads(calls[0]["function"]["arguments"]) == {"a": 17, "b": 25}

    def test_captured_sequence_byte_at_a_time(self):
        """Chunk boundaries must not change the outcome."""
        parser = EventStreamRoutingParser(enabled=True)
        blob = b"".join(event_frame(p) for p in CAPTURED_FRAMES)
        for i in range(0, len(blob), 5):
            parser.feed(blob[i:i + 5])

        calls = parser.get_tool_calls()
        assert len(calls) == 1
        assert json.loads(calls[0]["function"]["arguments"]) == {"a": 17, "b": 25}

    def test_name_input_and_stop_in_one_frame(self):
        """A single frame carrying name+input+stop completes the call."""
        parser = EventStreamRoutingParser(enabled=True)
        parser.feed(event_frame({
            "name": "add", "toolUseId": "tu-solo",
            "input": '{"a": 1}', "stop": True,
        }))

        calls = parser.get_tool_calls()
        assert len(calls) == 1
        assert calls[0]["id"] == "tu-solo"
        assert json.loads(calls[0]["function"]["arguments"]) == {"a": 1}

    def test_empty_string_input_fragment_is_harmless(self):
        """An empty-string `input` must not corrupt the accumulator."""
        parser = EventStreamRoutingParser(enabled=True)
        parser.feed(
            event_frame({"name": "f", "toolUseId": "tu-e", "input": ""})
            + event_frame({"name": "f", "toolUseId": "tu-e", "input": '{"a":'})
            + event_frame({"name": "f", "toolUseId": "tu-e", "input": ""})
            + event_frame({"name": "f", "toolUseId": "tu-e", "input": " 2}"})
            + event_frame({"name": "f", "toolUseId": "tu-e", "input": ""})
            + event_frame({"name": "f", "toolUseId": "tu-e", "stop": True})
        )

        calls = parser.get_tool_calls()
        assert len(calls) == 1
        assert json.loads(calls[0]["function"]["arguments"]) == {"a": 2}

    def test_two_sequential_tool_calls_both_complete(self):
        """A changing toolUseId finalises the previous call and starts a new one."""
        first = [dict(p) for p in CAPTURED_FRAMES]
        second = [
            {"name": "mul", "toolUseId": "tu-second", "input": '{"x'},
            {"name": "mul", "toolUseId": "tu-second", "input": '": 3}'},
            {"name": "mul", "toolUseId": "tu-second", "stop": True},
        ]

        parser = EventStreamRoutingParser(enabled=True)
        parser.feed(b"".join(event_frame(p) for p in first + second))

        calls = parser.get_tool_calls()
        assert len(calls) == 2
        assert calls[0]["id"] == TID
        assert json.loads(calls[0]["function"]["arguments"]) == {"a": 17, "b": 25}
        assert calls[1]["id"] == "tu-second"
        assert calls[1]["function"]["name"] == "mul"
        assert json.loads(calls[1]["function"]["arguments"]) == {"x": 3}

    def test_toolusedid_change_without_stop_finalises_previous(self):
        """Missing stop on the first call still yields both calls."""
        parser = EventStreamRoutingParser(enabled=True)
        parser.feed(
            event_frame({"name": "a", "toolUseId": "t1", "input": '{"p": 1}'})
            + event_frame({"name": "b", "toolUseId": "t2", "input": '{"q": 2}'})
            + event_frame({"name": "b", "toolUseId": "t2", "stop": True})
        )

        calls = parser.get_tool_calls()
        assert [c["id"] for c in calls] == ["t1", "t2"]
        assert json.loads(calls[0]["function"]["arguments"]) == {"p": 1}
        assert json.loads(calls[1]["function"]["arguments"]) == {"q": 2}


class TestLegacyPathUnchanged:

    def test_legacy_prefix_path_result_unchanged(self):
        """Regression guard: the legacy prefix parser still yields the same call."""
        legacy = AwsEventStreamParser()
        legacy.feed(b"".join(
            json.dumps(p).encode()
            for p in (
                {"name": "add", "toolUseId": TID, "input": ""},
                {"input": '{"a":'},
                {"input": " 17"},
                {"input": ', "b": 25}'},
                {"stop": True},
            )
        ))

        calls = legacy.get_tool_calls()
        assert len(calls) == 1
        assert calls[0]["id"] == TID
        assert json.loads(calls[0]["function"]["arguments"]) == {"a": 17, "b": 25}

    def test_framed_matches_legacy_for_repeated_name_sequence(self):
        """Framed path and legacy path agree on the captured payloads."""
        framed = EventStreamRoutingParser(enabled=True)
        framed.feed(b"".join(event_frame(p) for p in CAPTURED_FRAMES))

        legacy = AwsEventStreamParser()
        legacy.feed(b"".join(
            json.dumps(p).encode()
            for p in (
                {"name": "add", "toolUseId": TID, "input": ""},
                {"input": '{"'},
                {"input": 'a":'},
                {"input": " 17"},
                {"input": ', "b"'},
                {"input": ": 25}"},
                {"stop": True},
            )
        ))

        assert framed.get_tool_calls() == legacy.get_tool_calls()
