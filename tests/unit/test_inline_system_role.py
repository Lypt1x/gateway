# -*- coding: utf-8 -*-
"""
Tests for FIX-01 — inline role="system" messages in the `messages` array.

Covers:
- AnthropicMessage accepts role="system" (previously HTTP 422 literal_error)
- inline system messages are hoisted into the effective system prompt
- multiple inline system messages concatenate in order
- inline system + top-level system merge without loss (inline first)
- unknown roles are coerced to "user" instead of being dropped
- ordering of surviving user/assistant turns is preserved
"""

from kiro.converters_anthropic import (
    extract_system_prompt,
    normalize_inline_system_messages,
)
from kiro.models_anthropic import AnthropicMessage, AnthropicMessagesRequest


def _msg(role, content):
    return AnthropicMessage(role=role, content=content)


class TestRoleValidation:
    def test_system_role_accepted(self):
        assert _msg("system", "You are terse.").role == "system"

    def test_unknown_role_accepted(self):
        assert _msg("tool", "output").role == "tool"

    def test_request_with_inline_system_validates(self):
        req = AnthropicMessagesRequest(
            model="claude-sonnet-4-5",
            max_tokens=16,
            messages=[
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "Hi"},
            ],
        )
        assert [m.role for m in req.messages] == ["system", "user"]


class TestNormalizeInlineSystem:
    def test_single_inline_system_hoisted(self):
        messages, system = normalize_inline_system_messages(
            [_msg("system", "You are terse."), _msg("user", "Hi")], None
        )
        assert [m.role for m in messages] == ["user"]
        assert extract_system_prompt(system) == "You are terse."

    def test_list_of_blocks_content_shape(self):
        messages, system = normalize_inline_system_messages(
            [
                _msg("system", [{"type": "text", "text": "Alpha"}, {"type": "text", "text": "Beta"}]),
                _msg("user", "Hi"),
            ],
            None,
        )
        assert len(messages) == 1
        assert extract_system_prompt(system) == "AlphaBeta"

    def test_multiple_inline_system_concatenate_in_order(self):
        messages, system = normalize_inline_system_messages(
            [
                _msg("system", "First"),
                _msg("user", "Hi"),
                _msg("system", "Second"),
                _msg("assistant", "Hello"),
                _msg("system", "Third"),
            ],
            None,
        )
        assert [m.role for m in messages] == ["user", "assistant"]
        assert extract_system_prompt(system) == "First\nSecond\nThird"

    def test_inline_and_top_level_merge_without_loss(self):
        messages, system = normalize_inline_system_messages(
            [_msg("system", "Inline"), _msg("user", "Hi")], "TopLevel"
        )
        merged = extract_system_prompt(system)
        assert merged == "Inline\nTopLevel"
        assert "Inline" in merged and "TopLevel" in merged

    def test_inline_and_top_level_block_list_merge(self):
        messages, system = normalize_inline_system_messages(
            [_msg("system", "Inline"), _msg("user", "Hi")],
            [{"type": "text", "text": "TopA"}, {"type": "text", "text": "TopB"}],
        )
        assert extract_system_prompt(system) == "Inline\nTopA\nTopB"

    def test_unknown_role_coerced_to_user(self):
        messages, system = normalize_inline_system_messages(
            [_msg("tool", "tool output"), _msg("user", "Hi")], None
        )
        assert [m.role for m in messages] == ["user", "user"]
        assert messages[0].content == "tool output"
        assert system is None

    def test_surviving_turn_order_intact(self):
        messages, _ = normalize_inline_system_messages(
            [
                _msg("user", "u1"),
                _msg("system", "s"),
                _msg("assistant", "a1"),
                _msg("developer", "d1"),
                _msg("user", "u2"),
            ],
            None,
        )
        assert [(m.role, m.content) for m in messages] == [
            ("user", "u1"),
            ("assistant", "a1"),
            ("user", "d1"),
            ("user", "u2"),
        ]

    def test_no_system_is_passthrough(self):
        original = [_msg("user", "Hi"), _msg("assistant", "Yo")]
        messages, system = normalize_inline_system_messages(original, "Top")
        assert [m.role for m in messages] == ["user", "assistant"]
        assert system == "Top"

    def test_empty_inline_system_content_ignored(self):
        messages, system = normalize_inline_system_messages(
            [_msg("system", ""), _msg("user", "Hi")], "Top"
        )
        assert len(messages) == 1
        assert extract_system_prompt(system) == "Top"
