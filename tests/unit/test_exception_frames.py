# -*- coding: utf-8 -*-

"""
Tests for RECOMMENDATION 2: mid-stream application-level failure frames.

An in-band exception frame (ThrottlingError, ...) or an invalidState event used to
match nothing in the legacy prefix parser: it was silently discarded and the stream
simply ended, so the client saw a clean SHORT ANSWER with no error at all. These
tests pin the new behaviour: a visible error PLUS a well-formed end of stream.

Fully offline - no network, no gateway, framed bytes are built locally.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kiro.eventstream import encode_frame
from kiro.network_errors import (
    UpstreamStreamException,
    classify_stream_failure_frame,
    should_closeout_midstream,
    describe_midstream_failure,
    MIDSTREAM_CLOSEOUT_EXCEPTIONS,
)
from kiro.streaming_anthropic import stream_kiro_to_anthropic
from kiro.streaming_openai import stream_kiro_to_openai_internal
from kiro.streaming_core import parse_kiro_stream, FirstTokenTimeoutError, KiroEvent


# ==================================================================================================
# Fixtures / helpers
# ==================================================================================================

@pytest.fixture
def mock_model_cache():
    cache = MagicMock()
    cache.get_max_input_tokens.return_value = 200000
    return cache


@pytest.fixture
def mock_auth_manager():
    return MagicMock()


@pytest.fixture
def mock_response():
    response = AsyncMock()
    response.status_code = 200
    response.aclose = AsyncMock()
    return response


@pytest.fixture
def mock_http_client():
    return MagicMock()


def content_frame(text: str) -> bytes:
    return encode_frame(
        {
            ":event-type": "assistantResponseEvent",
            ":message-type": "event",
            ":content-type": "application/json",
        },
        json.dumps({"content": text}).encode("utf-8"),
    )


def tool_start_frame(name: str, tool_id: str) -> bytes:
    return encode_frame(
        {":event-type": "toolUseEvent", ":message-type": "event"},
        json.dumps({"name": name, "toolUseId": tool_id}).encode("utf-8"),
    )


def tool_input_frame(fragment: str) -> bytes:
    return encode_frame(
        {":event-type": "toolUseEvent", ":message-type": "event"},
        json.dumps({"input": fragment}).encode("utf-8"),
    )


THROTTLING_MESSAGE = "Too many requests, please wait before trying again."


def throttling_frame(message: str = THROTTLING_MESSAGE) -> bytes:
    return encode_frame(
        {
            ":message-type": "exception",
            ":exception-type": "ThrottlingError",
            ":content-type": "application/json",
        },
        json.dumps({"message": message}).encode("utf-8"),
    )


INVALID_STATE_MESSAGE = "Conversation is in an invalid state."


def invalid_state_frame(message: str = INVALID_STATE_MESSAGE) -> bytes:
    return encode_frame(
        {":event-type": "invalidStateEvent", ":message-type": "event"},
        json.dumps({"reason": "INVALID_STATE", "message": message}).encode("utf-8"),
    )


def aiter_of(chunks):
    """aiter_bytes replacement yielding the given chunks then ending normally."""
    def aiter_bytes():
        async def gen():
            for chunk in chunks:
                yield chunk
        return gen()
    return aiter_bytes


def parse_anthropic_events(chunks):
    parsed = []
    for chunk in chunks:
        event_name = None
        payload = None
        for line in chunk.strip().split("\n"):
            if line.startswith("event: "):
                event_name = line[len("event: "):]
            elif line.startswith("data: "):
                payload = json.loads(line[len("data: "):])
        if event_name:
            parsed.append((event_name, payload))
    return parsed


def parse_openai_chunks(chunks):
    payloads = []
    done_seen = 0
    for chunk in chunks:
        for line in chunk.strip().split("\n"):
            if not line.startswith("data: "):
                continue
            body = line[len("data: "):]
            if body == "[DONE]":
                done_seen += 1
            else:
                payloads.append(json.loads(body))
    return payloads, done_seen


# ==================================================================================================
# Classification
# ==================================================================================================

class TestFailureFrameClassification:
    """classify_stream_failure_frame() maps parser passthrough events to errors."""

    def test_exception_frame_is_classified_with_type_and_message(self):
        print("Action: classifying an exception frame with :exception-type...")
        err = classify_stream_failure_frame({
            "type": "exception",
            "data": {"message": THROTTLING_MESSAGE},
            "headers": {":exception-type": "ThrottlingError", ":message-type": "exception"},
            "message_type": "exception",
            "exception_type": "ThrottlingError",
        })
        assert isinstance(err, UpstreamStreamException)
        assert err.exception_type == "ThrottlingError"
        assert err.message == THROTTLING_MESSAGE
        assert err.is_throttling is True
        assert err.status_code == 429
        print(f"✓ {err.detail} -> {err.status_code}")

    def test_non_throttling_exception_is_not_429(self):
        err = classify_stream_failure_frame({
            "type": "exception",
            "data": {"message": "bad input"},
            "headers": {":exception-type": "ValidationError"},
            "exception_type": "ValidationError",
        })
        assert err.is_throttling is False
        assert err.status_code == 502

    def test_exception_type_recovered_from_payload_when_header_absent(self):
        err = classify_stream_failure_frame({
            "type": "exception",
            "data": {"__type": "com.amazon#ThrottlingException", "message": "slow down"},
            "headers": {},
            "message_type": "error",
            "exception_type": None,
        })
        assert err.exception_type == "ThrottlingException"
        assert err.is_throttling is True

    def test_invalid_state_event_is_classified(self):
        for event_type in ("invalidState", "invalidStateEvent", "invalid_state_event"):
            err = classify_stream_failure_frame({
                "type": "unknown_event",
                "data": {"reason": "INVALID_STATE"},
                "headers": {},
                "event_type": event_type,
            })
            assert isinstance(err, UpstreamStreamException), event_type
            assert err.exception_type == "InvalidStateEvent"
            assert err.is_throttling is False

    def test_benign_unknown_events_are_still_ignored(self):
        print("Action: benign unmapped event types must stay ignored...")
        for event_type in ("reasoningContentEvent", "codeReferenceEvent",
                           "supplementaryWebLinksEvent", ""):
            assert classify_stream_failure_frame({
                "type": "unknown_event",
                "data": {"x": 1},
                "headers": {},
                "event_type": event_type,
            }) is None, event_type
        print("✓ No regression for benign frames")

    def test_missing_message_does_not_crash(self):
        err = classify_stream_failure_frame({
            "type": "exception", "data": None, "headers": {},
            "exception_type": "ThrottlingError",
        })
        assert err.message == ""
        assert "no message" in describe_midstream_failure(err)

    def test_closeout_classifier_and_tuple(self):
        err = UpstreamStreamException("ThrottlingError", "x")
        assert should_closeout_midstream(err) is True
        assert isinstance(err, MIDSTREAM_CLOSEOUT_EXCEPTIONS)
        assert should_closeout_midstream(GeneratorExit()) is False
        assert should_closeout_midstream(asyncio.CancelledError()) is False
        assert should_closeout_midstream(FirstTokenTimeoutError("t")) is False


# ==================================================================================================
# streaming_core: frames become real exceptions
# ==================================================================================================

class TestCoreRaisesOnFailureFrames:

    @pytest.mark.asyncio
    async def test_throttling_frame_raises_after_content(self, mock_response):
        print("Setup: framed content then a ThrottlingError exception frame...")
        mock_response.aiter_bytes = aiter_of([content_frame("Hello"), throttling_frame()])

        seen = []
        with pytest.raises(UpstreamStreamException) as exc_info:
            async for event in parse_kiro_stream(mock_response):
                seen.append(event)

        assert [e.content for e in seen if e.type == "content"] == ["Hello"]
        assert exc_info.value.exception_type == "ThrottlingError"
        assert exc_info.value.message == THROTTLING_MESSAGE
        print(f"✓ raised: {exc_info.value.detail}")

    @pytest.mark.asyncio
    async def test_invalid_state_frame_raises(self, mock_response):
        mock_response.aiter_bytes = aiter_of([content_frame("Hi"), invalid_state_frame()])
        with pytest.raises(UpstreamStreamException) as exc_info:
            async for _ in parse_kiro_stream(mock_response):
                pass
        assert exc_info.value.exception_type == "InvalidStateEvent"
        assert exc_info.value.event_type == "invalidStateEvent"

    @pytest.mark.asyncio
    async def test_benign_unknown_frame_does_not_raise(self, mock_response):
        print("Setup: a reasoningContentEvent frame must not break the stream...")
        reasoning = encode_frame(
            {":event-type": "reasoningContentEvent", ":message-type": "event"},
            json.dumps({"text": "thinking"}).encode("utf-8"),
        )
        mock_response.aiter_bytes = aiter_of([content_frame("A"), reasoning, content_frame("B")])
        contents = [e.content async for e in parse_kiro_stream(mock_response) if e.type == "content"]
        assert contents == ["A", "B"]
        print("✓ Benign frame ignored, stream intact")

    @pytest.mark.asyncio
    async def test_warning_logged_without_secrets(self, mock_response, caplog):
        import logging
        from loguru import logger as _logger

        records = []
        sink_id = _logger.add(lambda m: records.append(m.record), level="WARNING")
        try:
            mock_response.aiter_bytes = aiter_of([content_frame("x"), throttling_frame()])
            with pytest.raises(UpstreamStreamException):
                async for _ in parse_kiro_stream(mock_response):
                    pass
        finally:
            _logger.remove(sink_id)

        warnings = [r for r in records if r["level"].name == "WARNING"]
        assert warnings, "an in-band failure must be logged at WARNING"
        text = " ".join(r["message"] for r in warnings)
        assert "ThrottlingError" in text
        assert THROTTLING_MESSAGE in text
        assert "Bearer" not in text and "token" not in text.lower()
        print(f"✓ WARNING: {text}")


# ==================================================================================================
# Anthropic dialect
# ==================================================================================================

class TestAnthropicFailureFrameCloseOut:

    async def _run(self, chunks, mock_response, mock_model_cache, mock_auth_manager):
        mock_response.aiter_bytes = aiter_of(chunks)
        out = []
        async for chunk in stream_kiro_to_anthropic(
            mock_response, "claude-sonnet-4", mock_model_cache, mock_auth_manager
        ):
            out.append(chunk)
        return parse_anthropic_events(out)

    @pytest.mark.asyncio
    async def test_throttling_frame_yields_error_and_wellformed_end(
        self, mock_response, mock_model_cache, mock_auth_manager
    ):
        print("Setup: two content frames then a mid-stream ThrottlingError frame...")
        events = await self._run(
            [content_frame("Hello"), content_frame(" world"), throttling_frame()],
            mock_response, mock_model_cache, mock_auth_manager,
        )
        names = [n for n, _ in events]
        print(f"events: {names}")

        assert names.count("message_delta") == 1
        assert names.count("message_stop") == 1
        assert names[-1] == "message_stop"
        assert names[-2] == "message_delta"
        assert names.count("message_start") == 1

        # (b) explicit error, mapped to the Anthropic 429 dialect
        errors = [p for n, p in events if n == "error"]
        assert len(errors) == 1
        assert errors[0]["error"]["type"] == "rate_limit_error"
        assert "ThrottlingError" in errors[0]["error"]["message"]
        assert THROTTLING_MESSAGE in errors[0]["error"]["message"]

        # (a) well-formed truncated turn, partial content preserved
        stop_reason = [p for n, p in events if n == "message_delta"][0]["delta"]["stop_reason"]
        assert stop_reason == "max_tokens"
        text = "".join(
            p["delta"]["text"] for n, p in events
            if n == "content_block_delta" and p["delta"].get("type") == "text_delta"
        )
        assert text == "Hello world"
        print("✓ visible error + well-formed truncated turn")

    @pytest.mark.asyncio
    async def test_invalid_state_frame_yields_api_error(
        self, mock_response, mock_model_cache, mock_auth_manager
    ):
        events = await self._run(
            [content_frame("Partial answer"), invalid_state_frame()],
            mock_response, mock_model_cache, mock_auth_manager,
        )
        names = [n for n, _ in events]
        assert names.count("message_delta") == 1 and names.count("message_stop") == 1
        errors = [p for n, p in events if n == "error"]
        assert len(errors) == 1
        assert errors[0]["error"]["type"] == "api_error"
        assert "InvalidStateEvent" in errors[0]["error"]["message"]
        assert INVALID_STATE_MESSAGE in errors[0]["error"]["message"]

    @pytest.mark.asyncio
    async def test_partial_tool_call_dropped_and_stop_reason_not_tool_use(
        self, mock_response, mock_model_cache, mock_auth_manager
    ):
        print("Setup: exception frame arrives while tool argument JSON is incomplete...")
        events = await self._run(
            [
                content_frame("Let me call a tool"),
                tool_start_frame("Write", "tool-r2"),
                tool_input_frame('{"path":"/tmp/a.txt","content":"unfin'),
                throttling_frame(),
            ],
            mock_response, mock_model_cache, mock_auth_manager,
        )
        names = [n for n, _ in events]
        print(f"events: {names}")

        tool_starts = [
            p for n, p in events
            if n == "content_block_start" and p["content_block"]["type"] == "tool_use"
        ]
        assert tool_starts == [], "partial tool call must be dropped"

        stop_reason = [p for n, p in events if n == "message_delta"][0]["delta"]["stop_reason"]
        assert stop_reason != "tool_use"
        assert stop_reason == "max_tokens"
        print("✓ partial tool dropped, stop_reason=max_tokens")

    @pytest.mark.asyncio
    async def test_generator_exit_still_propagates(
        self, mock_response, mock_model_cache, mock_auth_manager
    ):
        print("Setup: GeneratorExit mid-stream must never be swallowed...")

        async def mock_parse(*args, **kwargs):
            yield KiroEvent(type="content", content="Partial")
            raise GeneratorExit()

        with patch('kiro.streaming_anthropic.parse_kiro_stream', mock_parse):
            with pytest.raises(GeneratorExit):
                async for _ in stream_kiro_to_anthropic(
                    mock_response, "claude-sonnet-4", mock_model_cache, mock_auth_manager
                ):
                    pass
        print("✓ GeneratorExit propagated")

    @pytest.mark.asyncio
    async def test_cancelled_error_still_propagates(
        self, mock_response, mock_model_cache, mock_auth_manager
    ):
        async def mock_parse(*args, **kwargs):
            yield KiroEvent(type="content", content="Partial")
            raise asyncio.CancelledError()

        with patch('kiro.streaming_anthropic.parse_kiro_stream', mock_parse):
            with pytest.raises(asyncio.CancelledError):
                async for _ in stream_kiro_to_anthropic(
                    mock_response, "claude-sonnet-4", mock_model_cache, mock_auth_manager
                ):
                    pass

    @pytest.mark.asyncio
    async def test_pre_first_token_failure_propagates(
        self, mock_response, mock_model_cache, mock_auth_manager
    ):
        print("Setup: exception frame BEFORE any content -> route layer must see it...")
        mock_response.aiter_bytes = aiter_of([throttling_frame()])
        with pytest.raises(UpstreamStreamException):
            async for _ in stream_kiro_to_anthropic(
                mock_response, "claude-sonnet-4", mock_model_cache, mock_auth_manager
            ):
                pass
        print("✓ propagated with no half-open turn")

    @pytest.mark.asyncio
    async def test_truncation_recorded_for_next_turn(
        self, mock_response, mock_model_cache, mock_auth_manager
    ):
        print("Setup: recovery enabled -> content truncation must be recorded...")
        mock_response.aiter_bytes = aiter_of([content_frame("Half an answer"), throttling_frame()])

        with patch('kiro.truncation_recovery.should_inject_recovery', return_value=True), \
             patch('kiro.truncation_state.save_content_truncation') as save_mock:
            async for _ in stream_kiro_to_anthropic(
                mock_response, "claude-sonnet-4", mock_model_cache, mock_auth_manager
            ):
                pass

        save_mock.assert_called_once()
        assert save_mock.call_args[0][0] == "Half an answer"
        print("✓ save_content_truncation called")

    @pytest.mark.asyncio
    async def test_normal_framed_stream_unchanged(
        self, mock_response, mock_model_cache, mock_auth_manager
    ):
        """Regression guard: a stream with no failure frame emits no error event."""
        events = await self._run(
            [content_frame("All"), content_frame(" good")],
            mock_response, mock_model_cache, mock_auth_manager,
        )
        names = [n for n, _ in events]
        assert "error" not in names
        assert names.count("message_delta") == 1 and names[-1] == "message_stop"
        text = "".join(
            p["delta"]["text"] for n, p in events
            if n == "content_block_delta" and p["delta"].get("type") == "text_delta"
        )
        assert text == "All good"


# ==================================================================================================
# OpenAI dialect
# ==================================================================================================

class TestOpenAIFailureFrameCloseOut:

    async def _run(self, chunks, mock_http_client, mock_response,
                   mock_model_cache, mock_auth_manager):
        mock_response.aiter_bytes = aiter_of(chunks)
        out = []
        async for chunk in stream_kiro_to_openai_internal(
            mock_http_client, mock_response, "claude-sonnet-4",
            mock_model_cache, mock_auth_manager
        ):
            out.append(chunk)
        return out

    @pytest.mark.asyncio
    async def test_throttling_frame_yields_error_chunk_then_done(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        print("Setup: content frames then a mid-stream ThrottlingError frame...")
        chunks = await self._run(
            [content_frame("Hello"), content_frame(" world"), throttling_frame()],
            mock_http_client, mock_response, mock_model_cache, mock_auth_manager,
        )
        payloads, done_seen = parse_openai_chunks(chunks)

        # (b) in-band error chunk (#268 plumbing), mapped to 429 for throttling
        errors = [p for p in payloads if "error" in p]
        assert len(errors) == 1
        assert errors[0]["error"]["code"] == 429
        assert "ThrottlingError" in errors[0]["error"]["message"]
        assert THROTTLING_MESSAGE in errors[0]["error"]["message"]

        # (a) well-formed end of stream
        finish_reasons = [
            p["choices"][0].get("finish_reason") for p in payloads if "choices" in p
        ]
        non_null = [f for f in finish_reasons if f is not None]
        print(f"finish_reasons={finish_reasons} done={done_seen}")
        assert len(non_null) == 1 and non_null[0] == "length"
        assert done_seen == 1
        assert chunks[-1].strip() == "data: [DONE]"

        text = "".join(
            p["choices"][0]["delta"].get("content", "")
            for p in payloads if "choices" in p
        )
        assert text == "Hello world"
        print("✓ visible 429 error chunk + single finish_reason + [DONE]")

    @pytest.mark.asyncio
    async def test_invalid_state_frame_yields_error_chunk(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        chunks = await self._run(
            [content_frame("Partial"), invalid_state_frame()],
            mock_http_client, mock_response, mock_model_cache, mock_auth_manager,
        )
        payloads, done_seen = parse_openai_chunks(chunks)
        errors = [p for p in payloads if "error" in p]
        assert len(errors) == 1
        assert errors[0]["error"]["code"] == 502
        assert INVALID_STATE_MESSAGE in errors[0]["error"]["message"]
        assert done_seen == 1
        finish_reasons = [
            p["choices"][0].get("finish_reason") for p in payloads if "choices" in p
        ]
        assert len([f for f in finish_reasons if f]) == 1

    @pytest.mark.asyncio
    async def test_partial_tool_call_dropped(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        print("Setup: exception frame during an unfinished tool call...")
        chunks = await self._run(
            [
                content_frame("Calling"),
                tool_start_frame("Write", "tool-r2b"),
                tool_input_frame('{"path":"/tmp/a.txt","content":"unfin'),
                throttling_frame(),
            ],
            mock_http_client, mock_response, mock_model_cache, mock_auth_manager,
        )
        payloads, done_seen = parse_openai_chunks(chunks)

        tool_deltas = [
            p for p in payloads
            if "choices" in p and p["choices"][0]["delta"].get("tool_calls")
        ]
        assert tool_deltas == [], "partial tool call must be dropped"

        finish_reasons = [
            f for f in (
                p["choices"][0].get("finish_reason") for p in payloads if "choices" in p
            ) if f is not None
        ]
        assert finish_reasons == ["length"]
        assert "tool_calls" not in finish_reasons
        assert done_seen == 1
        print("✓ partial tool dropped, finish_reason=length")

    @pytest.mark.asyncio
    async def test_pre_first_token_failure_propagates(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        mock_response.aiter_bytes = aiter_of([throttling_frame()])
        with pytest.raises(UpstreamStreamException):
            async for _ in stream_kiro_to_openai_internal(
                mock_http_client, mock_response, "claude-sonnet-4",
                mock_model_cache, mock_auth_manager
            ):
                pass

    @pytest.mark.asyncio
    async def test_truncation_recorded_for_next_turn(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        mock_response.aiter_bytes = aiter_of([content_frame("Half"), throttling_frame()])
        with patch('kiro.truncation_recovery.should_inject_recovery', return_value=True), \
             patch('kiro.truncation_state.save_content_truncation') as save_mock:
            async for _ in stream_kiro_to_openai_internal(
                mock_http_client, mock_response, "claude-sonnet-4",
                mock_model_cache, mock_auth_manager
            ):
                pass
        save_mock.assert_called_once()
        assert save_mock.call_args[0][0] == "Half"

    @pytest.mark.asyncio
    async def test_normal_framed_stream_unchanged(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        """Regression guard: no failure frame -> no error chunk, one finish_reason."""
        chunks = await self._run(
            [content_frame("All"), content_frame(" good")],
            mock_http_client, mock_response, mock_model_cache, mock_auth_manager,
        )
        payloads, done_seen = parse_openai_chunks(chunks)
        assert [p for p in payloads if "error" in p] == []
        assert done_seen == 1
        finish_reasons = [
            f for f in (
                p["choices"][0].get("finish_reason") for p in payloads if "choices" in p
            ) if f is not None
        ]
        assert len(finish_reasons) == 1
        text = "".join(
            p["choices"][0]["delta"].get("content", "")
            for p in payloads if "choices" in p
        )
        assert text == "All good"


# ==================================================================================================
# Legacy fallback path
# ==================================================================================================

class TestLegacyPathNotRegressed:
    """
    With EVENTSTREAM_DECODER off (or a non-framed stream), an exception payload is
    NOT detectable - there are no headers to classify. The requirement is only that
    behaviour does not regress: the stream still ends well-formed.
    """

    @pytest.mark.asyncio
    async def test_legacy_stream_with_exception_json_does_not_break(
        self, mock_response, mock_model_cache, mock_auth_manager
    ):
        print("Setup: raw (unframed) JSON stream including an exception-looking blob...")
        mock_response.aiter_bytes = aiter_of([
            b'{"content":"Hello"}',
            b'{"message":"Too many requests","__type":"ThrottlingError"}',
        ])

        out = []
        async for chunk in stream_kiro_to_anthropic(
            mock_response, "claude-sonnet-4", mock_model_cache, mock_auth_manager
        ):
            out.append(chunk)

        events = parse_anthropic_events(out)
        names = [n for n, _ in events]
        print(f"events: {names}")
        assert names.count("message_delta") == 1
        assert names[-1] == "message_stop"
        print("✓ legacy path unchanged (exception frames undetectable there)")
