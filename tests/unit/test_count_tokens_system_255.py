# -*- coding: utf-8 -*-

"""
Tests for the residual of upstream issue #255:
/v1/messages/count_tokens must apply the same inline-system normalization as
/v1/messages, otherwise it counts a different message set than the gateway sends.
"""

import pytest

COUNT_URL = "/v1/messages/count_tokens"

LONG_SYSTEM = (
    "You are a meticulous assistant. Always answer carefully, cite reasoning, "
    "and never invent facts that the user did not provide."
)


def _count(test_client, key, payload):
    response = test_client.post(COUNT_URL, headers={"x-api-key": key}, json=payload)
    assert response.status_code == 200, response.text
    return response.json()["input_tokens"]


class TestCountTokensInlineSystem:
    """Inline role="system" messages must be counted, and counted identically."""

    def test_inline_system_contributes_to_total(self, test_client, valid_proxy_api_key):
        without = _count(test_client, valid_proxy_api_key, {
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "Hello"}],
        })
        with_inline = _count(test_client, valid_proxy_api_key, {
            "model": "claude-sonnet-4-5",
            "messages": [
                {"role": "system", "content": LONG_SYSTEM},
                {"role": "user", "content": "Hello"},
            ],
        })
        assert with_inline > without

    def test_inline_equals_top_level_system(self, test_client, valid_proxy_api_key):
        inline = _count(test_client, valid_proxy_api_key, {
            "model": "claude-sonnet-4-5",
            "messages": [
                {"role": "system", "content": LONG_SYSTEM},
                {"role": "user", "content": "Hello"},
            ],
        })
        top_level = _count(test_client, valid_proxy_api_key, {
            "model": "claude-sonnet-4-5",
            "system": LONG_SYSTEM,
            "messages": [{"role": "user", "content": "Hello"}],
        })
        assert inline == top_level

    def test_inline_system_block_list_equals_top_level(self, test_client, valid_proxy_api_key):
        inline = _count(test_client, valid_proxy_api_key, {
            "model": "claude-sonnet-4-5",
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": LONG_SYSTEM}]},
                {"role": "user", "content": "Hello"},
            ],
        })
        top_level = _count(test_client, valid_proxy_api_key, {
            "model": "claude-sonnet-4-5",
            "system": LONG_SYSTEM,
            "messages": [{"role": "user", "content": "Hello"}],
        })
        assert inline == top_level

    def test_inline_plus_top_level_are_merged(self, test_client, valid_proxy_api_key):
        merged = _count(test_client, valid_proxy_api_key, {
            "model": "claude-sonnet-4-5",
            "system": "Top level guidance.",
            "messages": [
                {"role": "system", "content": LONG_SYSTEM},
                {"role": "user", "content": "Hello"},
            ],
        })
        only_top = _count(test_client, valid_proxy_api_key, {
            "model": "claude-sonnet-4-5",
            "system": "Top level guidance.",
            "messages": [{"role": "user", "content": "Hello"}],
        })
        assert merged > only_top


class TestCountTokensRegressionGuard:
    """Requests with no inline system must count exactly as before."""

    @pytest.mark.parametrize("payload", [
        {"model": "claude-sonnet-4-5", "messages": [{"role": "user", "content": "Hello, world!"}]},
        {
            "model": "claude-sonnet-4-5",
            "system": "You are helpful.",
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
                {"role": "user", "content": "How are you?"},
            ],
        },
        {
            "model": "claude-sonnet-4-5",
            "system": [{"type": "text", "text": "You are helpful."}],
            "messages": [{"role": "user", "content": "Hi"}],
        },
    ])
    def test_count_is_stable_without_inline_system(self, test_client, valid_proxy_api_key, payload):
        """Normalization is a no-op when no system message is inlined."""
        from kiro.tokenizer import estimate_request_tokens

        expected = estimate_request_tokens(
            messages=payload["messages"],
            tools=None,
            system_prompt=payload.get("system"),
            apply_claude_correction=True,
        )["total_tokens"]

        assert _count(test_client, valid_proxy_api_key, payload) == expected
