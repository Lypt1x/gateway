# -*- coding: utf-8 -*-

"""
Context / token accounting fixes.

Two measured defects:

* BUG 1 - the context-usage percentage was inverted against a fabricated 200k
  denominator for public aliases and unknown models. "auto-kiro" (alias of the internal
  "auto", 1,000,000) reported ~5x too few tokens.
* BUG 2 - an absent percentage reported prompt_tokens = 0 and labelled the source
  "tiktoken" although tiktoken was never used for the prompt.

Offline only: no gateway process, no live API call.
"""

import asyncio
from unittest.mock import patch

import pytest

from kiro import cache as cache_module
from kiro.cache import ModelInfoCache
from kiro.config import DEFAULT_MAX_INPUT_TOKENS
from kiro.parsers import AwsEventStreamParser, log_context_usage_payload_keys
from kiro.streaming_core import calculate_tokens_from_context_usage


AUTO_ENTRY = {
    "modelId": "auto",
    "modelName": "Auto",
    "tokenLimits": {"maxInputTokens": 1000000, "maxOutputTokens": 64000},
}
SONNET_ENTRY = {
    "modelId": "claude-sonnet-4.5",
    "modelName": "Claude Sonnet 4.5",
    "tokenLimits": {"maxInputTokens": 200000, "maxOutputTokens": 64000},
}
OPUS_ENTRY = {
    "modelId": "claude-opus-5",
    "modelName": "Claude Opus 5",
    "tokenLimits": {"maxInputTokens": 1000000, "maxOutputTokens": 64000},
}
# Static FALLBACK_MODELS shape: id only, nothing measured.
BARE_ENTRY = {"modelId": "bare-model"}


@pytest.fixture
def catalog() -> ModelInfoCache:
    """Cache holding the real catalogue plus the "auto-kiro" public alias."""
    cache = ModelInfoCache()
    asyncio.run(cache.update([AUTO_ENTRY, SONNET_ENTRY, OPUS_ENTRY, BARE_ENTRY]))
    cache.add_hidden_model("auto-kiro", "auto")
    return cache


# ==================================================================================================
# BUG 1 - denominator resolution
# ==================================================================================================

class TestDenominatorResolution:
    """resolve_max_input_tokens(): measured limits only, single-hop inheritance."""

    def test_auto_kiro_inherits_auto_limit(self, catalog):
        """The headline alias must yield auto's 1,000,000, not the fabricated 200k."""
        assert catalog.resolve_max_input_tokens("auto-kiro") == 1000000
        assert catalog.get_max_input_tokens("auto-kiro") == 1000000

    def test_acceptance_auto_kiro_reports_7356_not_1471(self, catalog):
        """ACCEPTANCE: same percentage/completion as the internal model."""
        auto = calculate_tokens_from_context_usage(0.7356, 50, catalog, "auto")
        alias = calculate_tokens_from_context_usage(0.7356, 50, catalog, "auto-kiro")

        assert auto[1] == 7356
        assert alias[1] == 7356, "regression: 1471 means the 200k default came back"
        assert alias[0] == 7306
        assert alias[2:] == ("subtraction", "API Kiro")

    def test_alias_not_in_cache_resolves_via_model_aliases(self, catalog):
        """An alias that is only a MODEL_ALIASES key still gets the target's limits."""
        with patch.dict(cache_module.MODEL_ALIASES, {"opus-shortcut": "claude-opus-5"}):
            assert catalog.get("opus-shortcut") is None  # not a cache entry at all
            assert catalog.resolve_max_input_tokens("opus-shortcut") == 1000000

    def test_self_referential_alias_terminates(self, catalog):
        """A self-referential alias must not loop, and must not invent a limit."""
        with patch.dict(cache_module.MODEL_ALIASES, {"loop": "loop"}):
            assert catalog.resolve_max_input_tokens("loop") is None
        catalog.add_hidden_model("selfie", "selfie")
        assert catalog.resolve_max_input_tokens("selfie") is None

    def test_cyclic_alias_config_terminates(self, catalog):
        """a -> b -> a resolves single-hop and terminates with no measured limit."""
        with patch.dict(cache_module.MODEL_ALIASES, {"a-model": "b-model", "b-model": "a-model"}):
            assert catalog.resolve_max_input_tokens("a-model") is None
            assert catalog.resolve_max_input_tokens("b-model") is None

    def test_alias_chain_is_not_walked(self, catalog):
        """Alias-of-alias is deliberately single-hop: nothing measured is inherited."""
        catalog.add_hidden_model("second-alias", "auto-kiro")
        assert catalog.resolve_max_input_tokens("second-alias") is None

    def test_bare_and_unknown_models_have_no_measured_limit(self, catalog):
        """FALLBACK_MODELS-shaped and absent entries report unknown, not 200k."""
        assert catalog.resolve_max_input_tokens("bare-model") is None
        assert catalog.resolve_max_input_tokens("never-heard-of-it") is None
        assert catalog.has_measured_max_input_tokens("claude-sonnet-4.5") is True

    def test_get_max_input_tokens_still_returns_int_default(self, catalog):
        """Callers needing a usable number keep getting one (no None, no raise)."""
        assert catalog.get_max_input_tokens("bare-model") == DEFAULT_MAX_INPUT_TOKENS
        assert catalog.get_max_input_tokens("never-heard-of-it") == DEFAULT_MAX_INPUT_TOKENS

    def test_add_hidden_model_does_not_fabricate_token_limits(self):
        """A synthetic entry must not carry a limit indistinguishable from a measured one."""
        cache = ModelInfoCache()
        cache.add_hidden_model("ghost-alias", "not-in-catalog")
        entry = cache.get("ghost-alias")

        assert "tokenLimits" not in entry
        assert entry["_default_max_input_tokens"] == DEFAULT_MAX_INPUT_TOKENS
        assert cache.resolve_max_input_tokens("ghost-alias") is None
        # ...but a usable int is still available for callers that need one.
        assert cache.get_max_input_tokens("ghost-alias") == DEFAULT_MAX_INPUT_TOKENS


# ==================================================================================================
# Inversion behaviour
# ==================================================================================================

class TestInversion:
    """calculate_tokens_from_context_usage() against real and unknown limits."""

    @pytest.mark.parametrize("model,limit", [("claude-sonnet-4.5", 200000), ("claude-opus-5", 1000000)])
    def test_real_limits_behave_exactly_as_today(self, catalog, model, limit):
        """Regression guard for models with measured limits."""
        prompt, total, prompt_source, total_source = calculate_tokens_from_context_usage(
            10.0, 100, catalog, model
        )
        assert total == int(0.10 * limit)
        assert prompt == total - 100
        assert (prompt_source, total_source) == ("subtraction", "API Kiro")

    def test_unknown_limit_is_never_presented_as_authoritative(self, catalog):
        """Percentage + unknown limit must not yield a "subtraction"/"API Kiro" figure."""
        prompt, total, prompt_source, total_source = calculate_tokens_from_context_usage(
            0.7356, 50, catalog, "bare-model"
        )
        assert prompt_source != "subtraction"
        assert total_source != "API Kiro"
        # And specifically not the 200k-derived number.
        assert total != int(0.007356 * DEFAULT_MAX_INPUT_TOKENS)

    def test_unknown_limit_with_prompt_material_is_estimated(self, catalog):
        """The unknown-denominator path estimates and says so."""
        messages = [{"role": "user", "content": "Explain the context accounting bug."}]
        prompt, total, prompt_source, total_source = calculate_tokens_from_context_usage(
            0.7356, 50, catalog, "bare-model", prompt_messages=messages
        )
        assert prompt > 0
        assert total == prompt + 50
        assert (prompt_source, total_source) == ("estimate", "estimate")


# ==================================================================================================
# BUG 2 - absent percentage
# ==================================================================================================

class TestAbsentPercentage:
    """The fallback must estimate, not report zero, and must label honestly."""

    def test_absent_percentage_estimates_non_zero_prompt(self, catalog):
        messages = [{"role": "user", "content": "hello " * 200}]
        prompt, total, prompt_source, total_source = calculate_tokens_from_context_usage(
            None, 50, catalog, "claude-sonnet-4.5", prompt_messages=messages
        )
        assert prompt > 0
        assert total == prompt + 50
        assert (prompt_source, total_source) == ("estimate", "estimate")

    def test_zero_percentage_also_estimates(self, catalog):
        prompt, _, prompt_source, _ = calculate_tokens_from_context_usage(
            0.0, 10, catalog, "claude-sonnet-4.5", prompt_text="some prompt text"
        )
        assert prompt > 0
        assert prompt_source == "estimate"

    def test_tools_are_counted_in_the_estimate(self, catalog):
        tools = [{"type": "function", "function": {"name": "get_weather", "description": "x" * 200,
                                                  "parameters": {"type": "object"}}}]
        without = calculate_tokens_from_context_usage(None, 0, catalog, "claude-opus-5",
                                                     prompt_text="hi")[0]
        with_tools = calculate_tokens_from_context_usage(None, 0, catalog, "claude-opus-5",
                                                        prompt_text="hi", prompt_tools=tools)[0]
        assert with_tools > without > 0

    def test_no_prompt_material_keeps_the_sentinel_for_call_sites(self, catalog):
        """streaming_openai/anthropic branch on "unknown" to run their own estimate."""
        result = calculate_tokens_from_context_usage(None, 50, catalog, "claude-sonnet-4.5")
        assert result == (0, 50, "unknown", "unknown")


# ==================================================================================================
# Future-proofing: contextUsageEvent key names
# ==================================================================================================

class TestContextUsageKeyLogging:
    """Names only, once per stream, never a value."""

    def test_logs_names_only_and_never_a_value(self):
        payload = {"contextUsagePercentage": 0.7356, "totalTokens": 123456}
        with patch("kiro.parsers.logger") as mock_logger:
            assert log_context_usage_payload_keys(payload, False) is True
        message = mock_logger.debug.call_args[0][0]

        assert "contextUsagePercentage" in message
        assert "totalTokens" in message
        assert "0.7356" not in message
        assert "123456" not in message

    def test_logs_once_per_stream(self):
        parser = AwsEventStreamParser()
        with patch("kiro.parsers.logger") as mock_logger:
            parser.feed(b'{"contextUsagePercentage":0.7356}')
            parser.feed(b'{"contextUsagePercentage":1.4712}')
            debug_calls = [c for c in mock_logger.debug.call_args_list
                           if "contextUsageEvent payload keys" in c[0][0]]
        assert len(debug_calls) == 1
        assert "0.7356" not in debug_calls[0][0][0]

    def test_reset_re_arms_the_log_for_the_next_stream(self):
        parser = AwsEventStreamParser()
        parser.feed(b'{"contextUsagePercentage":0.7356}')
        parser.reset()
        with patch("kiro.parsers.logger") as mock_logger:
            parser.feed(b'{"contextUsagePercentage":0.7356}')
            debug_calls = [c for c in mock_logger.debug.call_args_list
                           if "contextUsageEvent payload keys" in c[0][0]]
        assert len(debug_calls) == 1
