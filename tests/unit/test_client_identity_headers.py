# -*- coding: utf-8 -*-
"""
Tests for client-identity and AWS SDK retry bookkeeping headers.

Covers RECOMMENDATION 3 of issue-plans/kiro-cli-fidelity-analysis.md:
the User-Agent must have the real smithy-rs shape, both UA headers must
agree, the profile ARN header must be present-or-absent (never blank),
the agent mode must be a real wire value, and amz-sdk-* must follow
smithy-rs semantics (stable invocation id, incrementing attempt).

All offline. No network calls.
"""

import importlib
import re

import httpx
import pytest
from unittest.mock import AsyncMock, Mock, patch

from kiro import utils as kiro_utils
from kiro.utils import KIRO_USER_AGENT, get_kiro_headers, new_invocation_id


class _StubAuth:
    """Minimal auth_manager stand-in."""

    def __init__(self, profile_arn=None):
        self.profile_arn = profile_arn
        self.fingerprint = "f" * 64

    async def get_access_token(self):
        return "tok"

    async def force_refresh(self):
        return None


class TestUserAgent:
    """The UA must be the real smithy-rs identity, not the fabricated JS one."""

    def test_contains_real_smithy_tokens(self):
        ua = KIRO_USER_AGENT
        print(f"UA: {ua}")
        for token in (
            "aws-sdk-rust/",
            "ua/2.1",
            "api/codewhispererstreaming/0.1.17975",
            "os/linux",
            "lang/rust",
            "md/appVersion/2.18.0",
            "app/AmazonQ-For-CLI",
        ):
            assert token in ua, token

    def test_no_kiroide_and_no_fingerprint(self):
        ua = KIRO_USER_AGENT
        assert "KiroIDE" not in ua
        assert "aws-sdk-js" not in ua
        # No 64-hex-char per-install fingerprint anywhere in the UA.
        assert re.search(r"[0-9a-f]{64}", ua) is None
        # smithy uses '/' separators, never the JS SDK's '#'.
        assert "#" not in ua

    def test_both_user_agent_headers_identical(self):
        headers = get_kiro_headers(_StubAuth("arn:x"), "tok")
        assert headers["User-Agent"] == headers["x-amz-user-agent"] == KIRO_USER_AGENT

    def test_static_target_and_content_type_preserved(self):
        headers = get_kiro_headers(_StubAuth("arn:x"), "tok")
        assert headers["Content-Type"] == "application/x-amz-json-1.0"
        assert headers["x-amz-target"] == (
            "AmazonCodeWhispererStreamingService.GenerateAssistantResponse"
        )
        assert headers["Authorization"] == "Bearer tok"


class TestProfileArnHeader:
    """x-amzn-kiro-profile-arn: present when known, absent otherwise, never blank."""

    def test_present_from_auth_manager(self):
        headers = get_kiro_headers(_StubAuth("arn:aws:codewhisperer:::profile/A"), "tok")
        assert headers["x-amzn-kiro-profile-arn"] == "arn:aws:codewhisperer:::profile/A"

    def test_falls_back_to_config_profile_arn(self):
        with patch("kiro.utils.PROFILE_ARN", "arn:from:config"):
            headers = get_kiro_headers(_StubAuth(None), "tok")
        assert headers["x-amzn-kiro-profile-arn"] == "arn:from:config"

    def test_absent_when_unknown(self):
        with patch("kiro.utils.PROFILE_ARN", ""):
            headers = get_kiro_headers(_StubAuth(None), "tok")
        assert "x-amzn-kiro-profile-arn" not in headers

    def test_never_sent_blank(self):
        with patch("kiro.utils.PROFILE_ARN", ""):
            headers = get_kiro_headers(_StubAuth(""), "tok")
        assert all(v != "" for v in headers.values())


class TestAgentMode:
    """Only kiro_default / kiro_planner / kiro_spec are real wire values."""

    def test_default_is_kiro_default(self):
        headers = get_kiro_headers(_StubAuth("arn:x"), "tok")
        assert headers["x-amzn-kiro-agent-mode"] == "kiro_default"
        assert headers["x-amzn-kiro-agent-mode"] != "vibe"

    @pytest.mark.parametrize("mode", ["kiro_default", "kiro_planner", "kiro_spec"])
    def test_valid_modes_accepted(self, monkeypatch, mode):
        monkeypatch.setenv("KIRO_AGENT_MODE", mode)
        cfg = importlib.reload(importlib.import_module("kiro.config"))
        try:
            assert cfg.KIRO_AGENT_MODE == mode
        finally:
            monkeypatch.delenv("KIRO_AGENT_MODE", raising=False)
            importlib.reload(cfg)

    @pytest.mark.parametrize("bad", ["vibe", "default", "", "KIRO_DEFAULT", "nonsense"])
    def test_invalid_mode_rejected_and_falls_back(self, monkeypatch, bad):
        monkeypatch.setenv("KIRO_AGENT_MODE", bad)
        cfg = importlib.reload(importlib.import_module("kiro.config"))
        try:
            assert cfg.KIRO_AGENT_MODE == "kiro_default"
        finally:
            monkeypatch.delenv("KIRO_AGENT_MODE", raising=False)
            importlib.reload(cfg)


class TestOptOut:
    """PRIVACY OVERRIDE: opt-out stays true by default."""

    def test_default_true(self):
        headers = get_kiro_headers(_StubAuth("arn:x"), "tok")
        assert headers["x-amzn-codewhisperer-optout"] == "true"

    def test_configurable_false(self):
        with patch("kiro.utils.CODEWHISPERER_OPTOUT", False):
            headers = get_kiro_headers(_StubAuth("arn:x"), "tok")
        assert headers["x-amzn-codewhisperer-optout"] == "false"

    def test_env_var_supported(self, monkeypatch):
        monkeypatch.setenv("CODEWHISPERER_OPTOUT", "false")
        cfg = importlib.reload(importlib.import_module("kiro.config"))
        try:
            assert cfg.CODEWHISPERER_OPTOUT is False
        finally:
            monkeypatch.delenv("CODEWHISPERER_OPTOUT", raising=False)
            importlib.reload(cfg)


class TestSdkRetryBookkeeping:
    """amz-sdk-invocation-id is stable per operation; amz-sdk-request attempt increments."""

    def test_attempt_and_max_rendered(self):
        headers = get_kiro_headers(_StubAuth("a"), "tok", attempt=2, max_attempts=3)
        assert headers["amz-sdk-request"] == "attempt=2; max=3"

    def test_invocation_id_reused_when_passed(self):
        inv = new_invocation_id()
        a = get_kiro_headers(_StubAuth("a"), "tok", invocation_id=inv, attempt=1)
        b = get_kiro_headers(_StubAuth("a"), "tok", invocation_id=inv, attempt=2)
        assert a["amz-sdk-invocation-id"] == b["amz-sdk-invocation-id"] == inv

    def test_invocation_id_fresh_when_omitted(self):
        a = get_kiro_headers(_StubAuth("a"), "tok")
        b = get_kiro_headers(_StubAuth("a"), "tok")
        assert a["amz-sdk-invocation-id"] != b["amz-sdk-invocation-id"]

    @pytest.mark.asyncio
    async def test_retry_loop_stable_id_incrementing_attempt(self):
        """Three 500s: one invocation id, attempts 1..3."""
        from kiro.http_client import KiroHttpClient

        captured = []

        async def fake_request(method, url, **kwargs):
            captured.append(dict(kwargs["headers"]))
            resp = Mock()
            resp.status_code = 500
            return resp

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(side_effect=fake_request)

        http_client = KiroHttpClient(_StubAuth("arn:x"))
        with patch.object(http_client, "_get_client", return_value=mock_client), \
             patch("kiro.http_client.asyncio.sleep", new_callable=AsyncMock):
            await http_client.request_with_retry("POST", "https://example.invalid/")

        assert len(captured) == 3, captured
        ids = {h["amz-sdk-invocation-id"] for h in captured}
        assert len(ids) == 1, f"invocation id must be stable across retries, got {ids}"
        attempts = [h["amz-sdk-request"] for h in captured]
        assert attempts == ["attempt=1; max=3", "attempt=2; max=3", "attempt=3; max=3"]

    @pytest.mark.asyncio
    async def test_streaming_retry_keeps_stable_id(self):
        """Streaming path (build_request + send) shares the same bookkeeping."""
        from kiro.http_client import KiroHttpClient

        captured = []

        def build_request(method, url, **kwargs):
            captured.append(dict(kwargs["headers"]))
            return Mock()

        async def send(req, stream=False):
            resp = Mock()
            resp.status_code = 500
            return resp

        mock_client = AsyncMock()
        mock_client.build_request = Mock(side_effect=build_request)
        mock_client.send = AsyncMock(side_effect=send)

        http_client = KiroHttpClient(_StubAuth("arn:x"))
        with patch.object(http_client, "_get_client", return_value=mock_client), \
             patch("kiro.http_client.asyncio.sleep", new_callable=AsyncMock):
            await http_client.request_with_retry(
                "POST", "https://example.invalid/", json_data={"a": 1}, stream=True
            )

        assert len(captured) == 3
        assert len({h["amz-sdk-invocation-id"] for h in captured}) == 1
        assert [h["amz-sdk-request"] for h in captured] == [
            "attempt=1; max=3",
            "attempt=2; max=3",
            "attempt=3; max=3",
        ]
        # Connection: close (issue #38) must survive.
        assert captured[0]["Connection"] == "close"


class TestConsumersStillWork:
    """Existing header consumers keep working with the new signature."""

    def test_two_positional_args_still_supported(self):
        # account_manager / mcp_tools call get_kiro_headers(auth_manager, token).
        headers = get_kiro_headers(_StubAuth("arn:x"), "tok")
        assert headers["Authorization"] == "Bearer tok"
        assert "amz-sdk-invocation-id" in headers

    def test_all_header_values_are_valid_strings(self):
        headers = get_kiro_headers(_StubAuth("arn:x"), "tok")
        for k, v in headers.items():
            assert isinstance(v, str) and v.strip() == v and v
        # httpx must accept the header set as-is.
        httpx.Headers(headers)

    def test_mcp_adaptation_still_yields_json_without_target(self):
        headers = get_kiro_headers(_StubAuth("arn:x"), "tok")
        # Mirrors kiro/mcp_tools.py's overrides (that file is not modified here).
        headers["Content-Type"] = "application/json"
        headers.pop("x-amz-target", None)
        headers["x-amzn-codewhisperer-optout"] = "false"
        assert headers["Content-Type"] == "application/json"
        assert "x-amz-target" not in headers
        assert headers["User-Agent"] == headers["x-amz-user-agent"]

    def test_module_exposes_updatable_version_constants(self):
        for name in (
            "UA_SDK_VERSION",
            "UA_METADATA_VERSION",
            "UA_API_SERVICE_ID",
            "UA_API_VERSION",
            "UA_APP_VERSION",
            "UA_APP_NAME",
        ):
            assert isinstance(getattr(kiro_utils, name), str)
