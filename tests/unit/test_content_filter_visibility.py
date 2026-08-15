# -*- coding: utf-8 -*-

"""
Tests for content-policy block visibility.

Upstream ends a blocked turn with `metadataEvent {"stopReason": "CONTENT_FILTERED"}`.
The parser recorded that value and every consumer discarded it, so a flagged
request produced an EMPTY turn with no error whatsoever (reported live in
OpenCode). These tests pin the new behaviour in both dialects.

Fully offline: frames are built locally, nothing is sent anywhere, and no
policy-violating content appears anywhere - the placeholder "[blocked prompt]"
stands in for the request text.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from loguru import logger as _logger

from kiro.eventstream import encode_frame
from kiro.network_errors import (
    CONTENT_FILTER_NOTICE,
    CONTENT_FILTER_STOP_REASONS,
    is_content_filter_stop_reason,
)
from kiro.streaming_anthropic import stream_kiro_to_anthropic, collect_anthropic_response
from kiro.streaming_openai import stream_kiro_to_openai_internal, collect_stream_response


BLOCKED_PROMPT = "[blocked prompt]"


# ==================================================================================================
# Helpers
# ==================================================================================================

def _frame(event_type: str, payload: dict) -> bytes:
    return encode_frame(
        {
            ":event-type": event_type,
            ":message-type": "event",
            ":content-type": "application/json",
        },
        json.dumps(payload).encode("utf-8"),
    )


def content_frame(text: str) -> bytes:
    return _frame("assistantResponseEvent", {"content": text})


def metadata_frame(stop_reason: str) -> bytes:
    return _frame("metadataEvent", {"stopReason": stop_reason})


def context_usage_frame(percentage: float = 2.24) -> bytes:
    return _frame("contextUsageEvent", {"contextUsagePercentage": percentage})


def aiter_of(chunks):
    def aiter_bytes():
        async def gen():
            for chunk in chunks:
                yield chunk
        return gen()
    return aiter_bytes


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


async def run_openai(mock_http_client, mock_response, mock_model_cache, mock_auth_manager, frames):
    mock_response.aiter_bytes = aiter_of(frames)
    out = []
    async for chunk in stream_kiro_to_openai_internal(
        mock_http_client, mock_response, "claude-sonnet-4",
        mock_model_cache, mock_auth_manager,
        request_messages=[{"role": "user", "content": BLOCKED_PROMPT}],
    ):
        out.append(chunk)
    return out


async def run_anthropic(mock_response, mock_model_cache, mock_auth_manager, frames):
    mock_response.aiter_bytes = aiter_of(frames)
    out = []
    async for chunk in stream_kiro_to_anthropic(
        mock_response, "claude-sonnet-4", mock_model_cache, mock_auth_manager,
        request_messages=[{"role": "user", "content": BLOCKED_PROMPT}],
    ):
        out.append(chunk)
    return out


# ==================================================================================================
# The recognised-value set
# ==================================================================================================

class TestStopReasonMatching:

    def test_documented_values_are_recognised(self):
        for value in ("CONTENT_FILTERED", "CONTENT_FILTER"):
            assert value in CONTENT_FILTER_STOP_REASONS
            assert is_content_filter_stop_reason(value) is True

    def test_matching_is_case_and_separator_insensitive(self):
        for value in ("content_filtered", "content_filter", "ContentFiltered",
                      "content-filter", "CONTENT FILTERED"):
            assert is_content_filter_stop_reason(value) is True, value

    def test_normal_stop_reasons_are_not_filtered(self):
        for value in ("END_TURN", "MAX_TOKENS", "TOOL_USE", "", None, 7):
            assert is_content_filter_stop_reason(value) is False, value


# ==================================================================================================
# OpenAI dialect
# ==================================================================================================

class TestOpenAIContentFilter:

    @pytest.mark.asyncio
    async def test_empty_filtered_turn_reports_content_filter_once_then_done(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        print("Setup: a stream with NO content, ending in stopReason=CONTENT_FILTERED...")
        chunks = await run_openai(
            mock_http_client, mock_response, mock_model_cache, mock_auth_manager,
            [metadata_frame("CONTENT_FILTERED")],
        )
        payloads, done_seen = parse_openai_chunks(chunks)

        finish_reasons = [
            p["choices"][0]["finish_reason"] for p in payloads
            if p.get("choices") and p["choices"][0].get("finish_reason")
        ]
        assert finish_reasons == ["content_filter"]
        assert done_seen == 1
        assert chunks[-1] == "data: [DONE]\n\n"

        text = "".join(
            p["choices"][0]["delta"].get("content", "")
            for p in payloads if p.get("choices")
        )
        assert text == CONTENT_FILTER_NOTICE
        assert text.strip()
        print(f"✓ finish_reason=content_filter, notice='{text}'")

    @pytest.mark.asyncio
    async def test_partial_content_preserved_and_still_reported(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        chunks = await run_openai(
            mock_http_client, mock_response, mock_model_cache, mock_auth_manager,
            [content_frame("Partial answer"), metadata_frame("CONTENT_FILTERED")],
        )
        payloads, done_seen = parse_openai_chunks(chunks)

        text = "".join(
            p["choices"][0]["delta"].get("content", "")
            for p in payloads if p.get("choices")
        )
        assert text == "Partial answer"  # no notice injected, nothing dropped
        finish_reasons = [
            p["choices"][0]["finish_reason"] for p in payloads
            if p.get("choices") and p["choices"][0].get("finish_reason")
        ]
        assert finish_reasons == ["content_filter"]
        assert done_seen == 1

    @pytest.mark.asyncio
    async def test_filtered_turn_is_not_truncation(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        print("Setup: filtered turn with partial content and NO completion signals...")
        with patch('kiro.truncation_recovery.should_inject_recovery', return_value=True), \
             patch('kiro.truncation_state.save_content_truncation') as save_mock:
            chunks = await run_openai(
                mock_http_client, mock_response, mock_model_cache, mock_auth_manager,
                [content_frame("Partial"), metadata_frame("CONTENT_FILTERED")],
            )

        payloads, _ = parse_openai_chunks(chunks)
        finish_reasons = [
            p["choices"][0]["finish_reason"] for p in payloads
            if p.get("choices") and p["choices"][0].get("finish_reason")
        ]
        assert finish_reasons == ["content_filter"]
        assert "length" not in finish_reasons
        save_mock.assert_not_called()
        print("✓ Not misreported as truncation, no recovery notice recorded")

    @pytest.mark.asyncio
    async def test_case_variants_are_reported(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        for value in ("CONTENT_FILTER", "content_filtered"):
            chunks = await run_openai(
                mock_http_client, mock_response, mock_model_cache, mock_auth_manager,
                [metadata_frame(value)],
            )
            payloads, _ = parse_openai_chunks(chunks)
            finish_reasons = [
                p["choices"][0]["finish_reason"] for p in payloads
                if p.get("choices") and p["choices"][0].get("finish_reason")
            ]
            assert finish_reasons == ["content_filter"], value

    @pytest.mark.asyncio
    async def test_end_turn_and_max_tokens_unchanged(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        print("Regression guard: normal stop reasons must behave exactly as before...")
        for value in ("END_TURN", "MAX_TOKENS"):
            chunks = await run_openai(
                mock_http_client, mock_response, mock_model_cache, mock_auth_manager,
                [content_frame("Hello"), metadata_frame(value), context_usage_frame()],
            )
            payloads, done_seen = parse_openai_chunks(chunks)
            finish_reasons = [
                p["choices"][0]["finish_reason"] for p in payloads
                if p.get("choices") and p["choices"][0].get("finish_reason")
            ]
            assert finish_reasons == ["stop"], value
            assert done_seen == 1
            text = "".join(
                p["choices"][0]["delta"].get("content", "")
                for p in payloads if p.get("choices")
            )
            assert text == "Hello", value
        print("✓ END_TURN / MAX_TOKENS byte-identical to previous behaviour")

    @pytest.mark.asyncio
    async def test_non_streaming_carries_filtered_outcome(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        mock_response.aiter_bytes = aiter_of([metadata_frame("CONTENT_FILTERED")])
        result = await collect_stream_response(
            mock_http_client, mock_response, "claude-sonnet-4",
            mock_model_cache, mock_auth_manager,
            request_messages=[{"role": "user", "content": BLOCKED_PROMPT}],
        )
        assert result["choices"][0]["finish_reason"] == "content_filter"
        assert result["choices"][0]["message"]["content"] == CONTENT_FILTER_NOTICE

    @pytest.mark.asyncio
    async def test_logs_reason_only_without_content(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        records = []
        sink_id = _logger.add(lambda m: records.append(m.record), level="WARNING")
        try:
            await run_openai(
                mock_http_client, mock_response, mock_model_cache, mock_auth_manager,
                [content_frame("Secret partial text"), metadata_frame("CONTENT_FILTERED")],
            )
        finally:
            _logger.remove(sink_id)

        warnings = [r for r in records if r["level"].name == "WARNING"]
        assert len(warnings) == 1, "the block must be logged exactly once"
        text = warnings[0]["message"]
        assert "CONTENT_FILTERED" in text
        assert "Secret partial text" not in text
        assert BLOCKED_PROMPT not in text
        print(f"✓ WARNING (reason only): {text}")


# ==================================================================================================
# Anthropic dialect
# ==================================================================================================

class TestAnthropicContentFilter:

    @pytest.mark.asyncio
    async def test_empty_filtered_turn_emits_error_and_single_closeout(
        self, mock_response, mock_model_cache, mock_auth_manager
    ):
        print("Setup: Anthropic stream with NO content, ending in CONTENT_FILTERED...")
        chunks = await run_anthropic(
            mock_response, mock_model_cache, mock_auth_manager,
            [metadata_frame("CONTENT_FILTERED")],
        )
        events = parse_anthropic_events(chunks)
        names = [name for name, _ in events]

        assert names.count("message_start") == 1
        assert names.count("error") == 1
        assert names.count("message_delta") == 1
        assert names.count("message_stop") == 1

        deltas = [p for name, p in events if name == "message_delta"]
        assert deltas[0]["delta"]["stop_reason"] == "refusal"

        error_payload = [p for name, p in events if name == "error"][0]
        assert error_payload["error"]["message"] == CONTENT_FILTER_NOTICE

        texts = "".join(
            p["delta"].get("text", "")
            for name, p in events if name == "content_block_delta"
        )
        assert texts == CONTENT_FILTER_NOTICE
        print("✓ error event + exactly one message_delta(refusal) + one message_stop")

    @pytest.mark.asyncio
    async def test_no_tool_use_stop_reason_for_filtered_turn(
        self, mock_response, mock_model_cache, mock_auth_manager
    ):
        chunks = await run_anthropic(
            mock_response, mock_model_cache, mock_auth_manager,
            [metadata_frame("content_filter")],
        )
        events = parse_anthropic_events(chunks)
        deltas = [p for name, p in events if name == "message_delta"]
        assert [d["delta"]["stop_reason"] for d in deltas] == ["refusal"]

    @pytest.mark.asyncio
    async def test_partial_content_preserved(
        self, mock_response, mock_model_cache, mock_auth_manager
    ):
        chunks = await run_anthropic(
            mock_response, mock_model_cache, mock_auth_manager,
            [content_frame("Partial answer"), metadata_frame("CONTENT_FILTERED")],
        )
        events = parse_anthropic_events(chunks)
        texts = "".join(
            p["delta"].get("text", "")
            for name, p in events if name == "content_block_delta"
        )
        assert texts == "Partial answer"
        deltas = [p for name, p in events if name == "message_delta"]
        assert [d["delta"]["stop_reason"] for d in deltas] == ["refusal"]
        assert [name for name, _ in events].count("error") == 1

    @pytest.mark.asyncio
    async def test_filtered_turn_is_not_truncation(
        self, mock_response, mock_model_cache, mock_auth_manager
    ):
        with patch('kiro.truncation_recovery.should_inject_recovery', return_value=True), \
             patch('kiro.truncation_state.save_content_truncation') as save_mock:
            chunks = await run_anthropic(
                mock_response, mock_model_cache, mock_auth_manager,
                [content_frame("Partial"), metadata_frame("CONTENT_FILTERED")],
            )
        events = parse_anthropic_events(chunks)
        deltas = [p for name, p in events if name == "message_delta"]
        assert [d["delta"]["stop_reason"] for d in deltas] == ["refusal"]
        save_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_end_turn_and_max_tokens_unchanged(
        self, mock_response, mock_model_cache, mock_auth_manager
    ):
        for value in ("END_TURN", "MAX_TOKENS"):
            chunks = await run_anthropic(
                mock_response, mock_model_cache, mock_auth_manager,
                [content_frame("Hello"), metadata_frame(value), context_usage_frame()],
            )
            events = parse_anthropic_events(chunks)
            names = [name for name, _ in events]
            assert "error" not in names, value
            assert names.count("message_delta") == 1
            assert names.count("message_stop") == 1
            deltas = [p for name, p in events if name == "message_delta"]
            assert deltas[0]["delta"]["stop_reason"] == "end_turn", value

    @pytest.mark.asyncio
    async def test_non_streaming_carries_filtered_outcome(
        self, mock_response, mock_model_cache, mock_auth_manager
    ):
        mock_response.aiter_bytes = aiter_of([metadata_frame("CONTENT_FILTERED")])
        result = await collect_anthropic_response(
            mock_response, "claude-sonnet-4", mock_model_cache, mock_auth_manager,
            request_messages=[{"role": "user", "content": BLOCKED_PROMPT}],
        )
        assert result["stop_reason"] == "refusal"
        assert result["content"] == [{"type": "text", "text": CONTENT_FILTER_NOTICE}]

    @pytest.mark.asyncio
    async def test_non_streaming_logs_reason_only(
        self, mock_response, mock_model_cache, mock_auth_manager
    ):
        records = []
        sink_id = _logger.add(lambda m: records.append(m.record), level="WARNING")
        try:
            mock_response.aiter_bytes = aiter_of([
                content_frame("Secret partial text"), metadata_frame("CONTENT_FILTERED"),
            ])
            await collect_anthropic_response(
                mock_response, "claude-sonnet-4", mock_model_cache, mock_auth_manager,
                request_messages=[{"role": "user", "content": BLOCKED_PROMPT}],
            )
        finally:
            _logger.remove(sink_id)

        warnings = [r for r in records if r["level"].name == "WARNING"]
        assert len(warnings) == 1
        assert "CONTENT_FILTERED" in warnings[0]["message"]
        assert "Secret partial text" not in warnings[0]["message"]
