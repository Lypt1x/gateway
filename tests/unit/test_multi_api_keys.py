# -*- coding: utf-8 -*-

"""
Tests for multiple comma-separated keys in PROXY_API_KEY (upstream issue #236).

Covers:
- backward compatibility with a single key
- multiple keys authenticating on BOTH the OpenAI and Anthropic auth paths
- whitespace tolerance and empty-entry rejection
- unchanged 401 behaviour/shape for a wrong key
- constant-time comparison usage
- no key value ever reaches the logs
"""

import secrets

import pytest
from fastapi import HTTPException

from kiro import config
from kiro.config import (
    parse_proxy_api_keys,
    verify_proxy_api_key,
    verify_proxy_bearer_token,
)
from kiro.routes_openai import verify_api_key
from kiro.routes_anthropic import verify_anthropic_api_key


@pytest.fixture
def multi_keys(monkeypatch):
    """Configure three accepted keys for the duration of a test."""
    keys = ["key-one", "key-two", "key-three"]
    monkeypatch.setattr(config, "PROXY_API_KEYS", keys)
    return keys


# =============================================================================
# Parsing
# =============================================================================

class TestParseProxyApiKeys:
    """Tests for parse_proxy_api_keys()."""

    def test_single_key_backward_compatible(self):
        assert parse_proxy_api_keys("only-key") == ["only-key"]

    def test_comma_separated_list(self):
        assert parse_proxy_api_keys("a,b,c") == ["a", "b", "c"]

    def test_whitespace_is_stripped(self):
        assert parse_proxy_api_keys("  a , b  ,\tc ") == ["a", "b", "c"]

    def test_empty_entries_ignored(self):
        # A trailing comma or a blank entry must NOT create an empty always-matching key.
        assert parse_proxy_api_keys("a,,  ,b,") == ["a", "b"]

    def test_duplicates_collapsed(self):
        assert parse_proxy_api_keys("a,a,b") == ["a", "b"]

    def test_empty_value_yields_no_keys(self):
        assert parse_proxy_api_keys("") == []
        assert parse_proxy_api_keys(None) == []

    def test_default_is_parsed_into_keys(self):
        # The module-level list is always derived from PROXY_API_KEY.
        assert config.PROXY_API_KEYS == parse_proxy_api_keys(config.PROXY_API_KEY)
        assert config.PROXY_API_KEY in config.PROXY_API_KEYS


# =============================================================================
# Comparison helpers
# =============================================================================

class TestVerifyHelpers:
    """Tests for verify_proxy_api_key() / verify_proxy_bearer_token()."""

    def test_each_configured_key_accepted(self, multi_keys):
        for key in multi_keys:
            assert verify_proxy_api_key(key) is True

    def test_wrong_key_rejected(self, multi_keys):
        assert verify_proxy_api_key("nope") is False

    def test_empty_candidate_rejected(self, multi_keys):
        assert verify_proxy_api_key("") is False
        assert verify_proxy_api_key(None) is False

    def test_empty_key_entry_never_matches(self, monkeypatch):
        monkeypatch.setattr(config, "PROXY_API_KEYS", ["", "real"])
        assert verify_proxy_api_key("") is False
        assert verify_proxy_api_key("real") is True

    def test_non_ascii_key_supported(self, monkeypatch):
        monkeypatch.setattr(config, "PROXY_API_KEYS", ["pässwörd"])
        assert verify_proxy_api_key("pässwörd") is True
        assert verify_proxy_api_key("password") is False

    def test_bearer_token_parsing(self, multi_keys):
        assert verify_proxy_bearer_token("Bearer key-two") is True
        assert verify_proxy_bearer_token("Bearer  key-two") is False  # double space
        assert verify_proxy_bearer_token("bearer key-two") is False   # wrong case
        assert verify_proxy_bearer_token("key-two") is False          # missing scheme
        assert verify_proxy_bearer_token(None) is False

    def test_uses_constant_time_comparison(self, multi_keys, monkeypatch):
        """The helper must go through secrets.compare_digest, not == / in."""
        calls = []
        real = secrets.compare_digest

        def spy(a, b):
            calls.append((a, b))
            return real(a, b)

        monkeypatch.setattr(config.secrets, "compare_digest", spy)
        assert verify_proxy_api_key("key-three") is True
        assert calls, "secrets.compare_digest was not used"

    def test_no_early_exit_on_match(self, multi_keys, monkeypatch):
        """All configured keys are compared, so timing does not reveal the position."""
        calls = []
        real = secrets.compare_digest

        def spy(a, b):
            calls.append(b)
            return real(a, b)

        monkeypatch.setattr(config.secrets, "compare_digest", spy)
        verify_proxy_api_key("key-one")
        assert calls == multi_keys


# =============================================================================
# Route-level authentication
# =============================================================================

class TestOpenAIRouteAuth:
    """kiro/routes_openai.py verify_api_key."""

    @pytest.mark.asyncio
    async def test_single_key_still_works(self, monkeypatch):
        monkeypatch.setattr(config, "PROXY_API_KEYS", ["solo-key"])
        assert await verify_api_key(auth_header="Bearer solo-key") is True

    @pytest.mark.asyncio
    async def test_every_key_authenticates(self, multi_keys):
        for key in multi_keys:
            assert await verify_api_key(auth_header=f"Bearer {key}") is True

    @pytest.mark.asyncio
    async def test_wrong_key_gets_unchanged_401(self, multi_keys):
        with pytest.raises(HTTPException) as exc:
            await verify_api_key(auth_header="Bearer wrong")
        assert exc.value.status_code == 401
        assert exc.value.detail == "Invalid or missing API Key"

    @pytest.mark.asyncio
    async def test_missing_header_401(self, multi_keys):
        with pytest.raises(HTTPException) as exc:
            await verify_api_key(auth_header=None)
        assert exc.value.status_code == 401


class TestAnthropicRouteAuth:
    """kiro/routes_anthropic.py verify_anthropic_api_key."""

    @pytest.mark.asyncio
    async def test_single_key_still_works(self, monkeypatch):
        monkeypatch.setattr(config, "PROXY_API_KEYS", ["solo-key"])
        assert await verify_anthropic_api_key(x_api_key="solo-key", authorization=None) is True

    @pytest.mark.asyncio
    async def test_every_key_authenticates_via_x_api_key(self, multi_keys):
        for key in multi_keys:
            assert await verify_anthropic_api_key(x_api_key=key, authorization=None) is True

    @pytest.mark.asyncio
    async def test_every_key_authenticates_via_bearer(self, multi_keys):
        for key in multi_keys:
            assert await verify_anthropic_api_key(
                x_api_key=None, authorization=f"Bearer {key}"
            ) is True

    @pytest.mark.asyncio
    async def test_wrong_key_gets_unchanged_401_shape(self, multi_keys):
        with pytest.raises(HTTPException) as exc:
            await verify_anthropic_api_key(x_api_key="wrong", authorization=None)
        assert exc.value.status_code == 401
        assert exc.value.detail == {
            "type": "error",
            "error": {
                "type": "authentication_error",
                "message": "Invalid or missing API key. Use x-api-key header or Authorization: Bearer.",
            },
        }

    @pytest.mark.asyncio
    async def test_whitespace_around_configured_entries_tolerated(self, monkeypatch):
        monkeypatch.setattr(config, "PROXY_API_KEYS", parse_proxy_api_keys(" pad-a , pad-b "))
        assert await verify_anthropic_api_key(x_api_key="pad-a", authorization=None) is True
        assert await verify_anthropic_api_key(x_api_key="pad-b", authorization=None) is True


# =============================================================================
# Logging hygiene
# =============================================================================

@pytest.fixture
def loguru_sink():
    """Capture loguru output (the gateway logs through loguru, not stdlib logging)."""
    from loguru import logger

    records = []
    sink_id = logger.add(lambda msg: records.append(str(msg)), level="DEBUG")
    yield records
    logger.remove(sink_id)


class TestNoKeyLeakInLogs:
    """Key values must never be logged — at most a count."""

    @pytest.mark.asyncio
    async def test_openai_failure_log_has_no_key_values(self, multi_keys, loguru_sink):
        with pytest.raises(HTTPException):
            await verify_api_key(auth_header="Bearer super-secret-attempt")
        text = "".join(loguru_sink)
        assert text, "expected a warning to be logged"
        for key in multi_keys:
            assert key not in text
        assert "super-secret-attempt" not in text
        assert "3 key(s) configured" in text

    @pytest.mark.asyncio
    async def test_anthropic_failure_log_has_no_key_values(self, multi_keys, loguru_sink):
        with pytest.raises(HTTPException):
            await verify_anthropic_api_key(
                x_api_key="super-secret-attempt", authorization=None
            )
        text = "".join(loguru_sink)
        assert text, "expected a warning to be logged"
        for key in multi_keys:
            assert key not in text
        assert "super-secret-attempt" not in text
