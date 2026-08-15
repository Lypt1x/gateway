# -*- coding: utf-8 -*-

"""
Tests for dynamic model discovery (issue-plans/fix-dynamic-model-discovery.md).

All tests are fully offline: KiroHttpClient is patched, so no request ever leaves the
process. They cover the host split (chat stays on runtime.kiro.dev, listing goes to the
control plane), the MODEL_CACHE_TTL window, the FALLBACK_MODELS safety net, the
MODEL_DISCOVERY kill switch, hidden-model filtering, and metadata retention.
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

from kiro.account_manager import (
    Account,
    AccountManager,
    _get_api_region,
    _get_model_listing_host,
    _is_runtime_endpoint,
)
from kiro.auth import AuthType
from kiro.cache import ModelInfoCache
from kiro.config import FALLBACK_MODELS, HIDDEN_FROM_LIST
from kiro.model_resolver import ModelResolver


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

LIVE_MODEL_IDS = [
    "auto",
    "claude-sonnet-4",
    "claude-sonnet-4.5",
    "claude-sonnet-4.6",
    "claude-sonnet-5",
    "claude-haiku-4.5",
    "claude-opus-4.5",
    "claude-opus-4.6",
    "claude-opus-4.7",
    "claude-opus-4.8",
    "claude-opus-5",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "deepseek-3.2",
    "glm-5",
    "minimax-m2.1",
    "minimax-m2.5",
    "qwen3-coder-next",
]


def _model_entry(model_id: str) -> dict:
    """Build one entry shaped like a real ListAvailableModels model."""
    return {
        "modelId": model_id,
        "modelName": model_id.replace("-", " ").title(),
        "description": f"Description for {model_id}",
        "tokenLimits": {"maxInputTokens": 300000, "maxOutputTokens": 64000},
        "promptCaching": {"supported": True},
        "rateMultiplier": 1.5,
        "rateUnit": "REQUEST",
        "supportedInputTypes": ["TEXT", "IMAGE"],
        "availableOrigins": ["AI_EDITOR"],
        "additionalModelRequestFieldsSchema": {"type": "object"},
    }


def live_response_body(model_ids=None) -> dict:
    """Full 19-model response body as measured upstream."""
    ids = LIVE_MODEL_IDS if model_ids is None else model_ids
    return {
        "defaultModel": _model_entry(ids[0]),
        "models": [_model_entry(mid) for mid in ids],
    }


def make_auth_manager(
    api_host: str = "https://runtime.us-east-1.kiro.dev",
    q_host: str = "https://runtime.us-east-1.kiro.dev",
    region: str = "us-east-1",
) -> Mock:
    """Offline stand-in for KiroAuthManager (no credentials, no network)."""
    auth = Mock()
    auth.api_host = api_host
    auth.q_host = q_host
    auth.region = region
    auth.auth_type = AuthType.AWS_SSO_OIDC
    auth.profile_arn = None
    auth.get_access_token = AsyncMock(return_value="test-token")
    return auth


def patch_http(response=None, side_effect=None):
    """Patch account_manager.KiroHttpClient; returns (context manager, mock client)."""
    mock_client = AsyncMock()
    mock_client.request_with_retry = AsyncMock(return_value=response, side_effect=side_effect)
    mock_client.close = AsyncMock()
    ctx = patch("kiro.account_manager.KiroHttpClient", return_value=mock_client)
    return ctx, mock_client


def http_response(status_code: int = 200, body=None, raises=None) -> Mock:
    """Build a non-async httpx-like response double."""
    response = Mock()
    response.status_code = status_code
    if raises is not None:
        response.json.side_effect = raises
    else:
        response.json.return_value = body
    return response


def make_manager_with_account(tmp_path, auth_manager=None) -> tuple:
    """Build an AccountManager holding one initialized, offline account."""
    manager = AccountManager(
        credentials_file=str(tmp_path / "credentials.json"),
        state_file=str(tmp_path / "state.json"),
    )
    account_id = "acct-1"
    cache = ModelInfoCache()
    resolver = ModelResolver(
        cache=cache,
        hidden_models={},
        aliases={},
        hidden_from_list=HIDDEN_FROM_LIST,
    )
    account = Account(
        id=account_id,
        auth_manager=auth_manager or make_auth_manager(),
        model_cache=cache,
        model_resolver=resolver,
    )
    manager._accounts[account_id] = account
    return manager, account


# --------------------------------------------------------------------------------------
# Host split
# --------------------------------------------------------------------------------------

class TestListingHostSplit:
    """Listing goes to the control plane; chat/streaming host is untouched."""

    def test_runtime_host_is_detected(self):
        assert _is_runtime_endpoint(make_auth_manager()) is True

    def test_control_plane_host_is_not_runtime(self):
        auth = make_auth_manager(
            api_host="https://q.us-east-1.amazonaws.com",
            q_host="https://q.us-east-1.amazonaws.com",
        )
        assert _is_runtime_endpoint(auth) is False

    def test_region_recovered_from_runtime_host(self):
        auth = make_auth_manager(
            api_host="https://runtime.eu-central-1.kiro.dev",
            q_host="https://runtime.eu-central-1.kiro.dev",
            region="us-east-1",
        )
        assert _get_api_region(auth) == "eu-central-1"

    def test_runtime_account_lists_against_control_plane(self):
        auth = make_auth_manager()
        assert _get_model_listing_host(auth) == "https://q.us-east-1.amazonaws.com"

    def test_legacy_account_keeps_its_q_host(self):
        auth = make_auth_manager(
            api_host="https://q.eu-central-1.amazonaws.com",
            q_host="https://q.eu-central-1.amazonaws.com",
            region="eu-central-1",
        )
        assert _get_model_listing_host(auth) == "https://q.eu-central-1.amazonaws.com"

    @pytest.mark.asyncio
    async def test_discovery_requests_control_plane_url(self, tmp_path):
        """A runtime-host account must query q.{region}.amazonaws.com, not kiro.dev."""
        manager, account = make_manager_with_account(tmp_path)
        ctx, client = patch_http(response=http_response(200, live_response_body()))

        with ctx:
            models = await manager._discover_models(account.auth_manager, account.id)

        assert models is not None
        called_url = client.request_with_retry.await_args.kwargs["url"]
        assert called_url == "https://q.us-east-1.amazonaws.com/ListAvailableModels"
        assert "kiro.dev" not in called_url
        assert client.request_with_retry.await_args.kwargs["params"] == {"origin": "AI_EDITOR"}
        # Chat/streaming host untouched
        assert account.auth_manager.api_host == "https://runtime.us-east-1.kiro.dev"

    @pytest.mark.asyncio
    async def test_discovery_populates_cache_with_19_models(self, tmp_path):
        manager, account = make_manager_with_account(tmp_path)
        ctx, _ = patch_http(response=http_response(200, live_response_body()))

        with ctx:
            await manager._refresh_account_models(account.id, force=True)

        ids = account.model_cache.get_all_model_ids()
        assert len(ids) == 19
        assert "claude-opus-5" in ids
        for gpt_id in ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"):
            assert gpt_id in ids

    @pytest.mark.asyncio
    async def test_new_upstream_model_appears_automatically(self, tmp_path):
        """A model absent from the static list still shows up when upstream serves it."""
        manager, account = make_manager_with_account(tmp_path)
        static_ids = {m["modelId"] for m in FALLBACK_MODELS}
        assert "claude-opus-9-unreleased" not in static_ids
        body = live_response_body(LIVE_MODEL_IDS + ["claude-opus-9-unreleased"])
        ctx, _ = patch_http(response=http_response(200, body))

        with ctx:
            await manager._refresh_account_models(account.id, force=True)

        assert "claude-opus-9-unreleased" in account.model_cache.get_all_model_ids()


# --------------------------------------------------------------------------------------
# Fallback safety net
# --------------------------------------------------------------------------------------

class TestDiscoveryFallback:
    """FALLBACK_MODELS is the guaranteed floor for every failure mode."""

    @pytest.mark.parametrize(
        "response,side_effect",
        [
            (None, asyncio.TimeoutError()),
            (None, Exception("connection refused")),
            (http_response(500, {}), None),
            (http_response(403, {}), None),
            (http_response(200, {"models": []}), None),
            (http_response(200, {}), None),
            (http_response(200, ["not", "a", "dict"]), None),
            (http_response(200, {"models": "nonsense"}), None),
            (http_response(200, {"models": [{"noModelId": 1}]}), None),
            (http_response(200, None, raises=ValueError("invalid json")), None),
        ],
    )
    @pytest.mark.asyncio
    async def test_failure_returns_none_without_raising(self, tmp_path, response, side_effect):
        manager, account = make_manager_with_account(tmp_path)
        ctx, _ = patch_http(response=response, side_effect=side_effect)

        with ctx:
            result = await manager._discover_models(account.auth_manager, account.id)

        assert result is None

    @pytest.mark.asyncio
    async def test_refresh_installs_fallback_when_cache_empty(self, tmp_path):
        manager, account = make_manager_with_account(tmp_path)
        ctx, _ = patch_http(response=None, side_effect=Exception("host unreachable"))

        with ctx:
            await manager._refresh_account_models(account.id, force=True)

        ids = account.model_cache.get_all_model_ids()
        assert ids == [m["modelId"] for m in FALLBACK_MODELS]
        assert len(ids) == 19

    @pytest.mark.asyncio
    async def test_refresh_failure_keeps_existing_catalog(self, tmp_path):
        manager, account = make_manager_with_account(tmp_path)
        await account.model_cache.update([_model_entry("claude-opus-5")])
        ctx, _ = patch_http(response=http_response(500, {}))

        with ctx:
            await manager._refresh_account_models(account.id, force=True)

        assert account.model_cache.get_all_model_ids() == ["claude-opus-5"]

    @pytest.mark.asyncio
    async def test_timeout_does_not_block_beyond_budget(self, tmp_path):
        """A hanging control plane must be abandoned, not awaited indefinitely."""
        manager, account = make_manager_with_account(tmp_path)

        async def _hang(*args, **kwargs):
            await asyncio.sleep(30)

        mock_client = AsyncMock()
        mock_client.request_with_retry = _hang
        mock_client.close = AsyncMock()

        started = time.monotonic()
        with patch("kiro.account_manager.KiroHttpClient", return_value=mock_client), \
                patch("kiro.account_manager.MODEL_DISCOVERY_TIMEOUT", 0.05):
            result = await manager._discover_models(account.auth_manager, account.id)
        elapsed = time.monotonic() - started

        assert result is None
        assert elapsed < 5

    @pytest.mark.asyncio
    async def test_startup_succeeds_when_control_plane_unreachable(self, tmp_path):
        """Account initialization must succeed on the static list, not fail."""
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z",
        }))
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json"),
        )
        await manager.load_credentials()
        account_id = str(test_json.resolve())

        ctx, _ = patch_http(response=None, side_effect=Exception("Name or service not known"))
        with ctx:
            success = await manager._initialize_account(account_id)

        assert success is True
        account = manager._accounts[account_id]
        assert account.model_cache is not None
        assert account.model_cache.get_all_model_ids() == [m["modelId"] for m in FALLBACK_MODELS]

    @pytest.mark.asyncio
    async def test_initialization_uses_live_list_when_available(self, tmp_path):
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z",
        }))
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json"),
        )
        await manager.load_credentials()
        account_id = str(test_json.resolve())

        body = live_response_body(LIVE_MODEL_IDS + ["brand-new-model"])
        ctx, client = patch_http(response=http_response(200, body))
        with ctx:
            success = await manager._initialize_account(account_id)

        assert success is True
        ids = manager._accounts[account_id].model_cache.get_all_model_ids()
        assert "brand-new-model" in ids
        assert client.request_with_retry.await_count == 1


# --------------------------------------------------------------------------------------
# TTL
# --------------------------------------------------------------------------------------

class TestDiscoveryTTL:
    """models_cached_at + MODEL_CACHE_TTL prevent per-request fetching."""

    @pytest.mark.asyncio
    async def test_no_refetch_inside_ttl_window(self, tmp_path):
        manager, account = make_manager_with_account(tmp_path)
        await account.model_cache.update([_model_entry("claude-opus-5")])
        account.models_cached_at = time.time()

        ctx, client = patch_http(response=http_response(200, live_response_body()))
        with ctx, patch("kiro.account_manager.MODEL_CACHE_TTL", 3600):
            await manager._refresh_account_models(account.id)

        assert client.request_with_retry.await_count == 0
        assert account.model_cache.get_all_model_ids() == ["claude-opus-5"]

    @pytest.mark.asyncio
    async def test_refetch_after_ttl_expiry(self, tmp_path):
        manager, account = make_manager_with_account(tmp_path)
        await account.model_cache.update([_model_entry("claude-opus-5")])
        account.models_cached_at = time.time() - 7200

        ctx, client = patch_http(response=http_response(200, live_response_body()))
        with ctx, patch("kiro.account_manager.MODEL_CACHE_TTL", 3600):
            await manager._refresh_account_models(account.id)

        assert client.request_with_retry.await_count == 1
        assert len(account.model_cache.get_all_model_ids()) == 19

    @pytest.mark.asyncio
    async def test_exactly_one_fetch_across_repeated_calls(self, tmp_path):
        manager, account = make_manager_with_account(tmp_path)
        account.models_cached_at = 0.0

        ctx, client = patch_http(response=http_response(200, live_response_body()))
        with ctx, patch("kiro.account_manager.MODEL_CACHE_TTL", 3600):
            for _ in range(5):
                await manager._refresh_account_models(account.id)

        assert client.request_with_retry.await_count == 1

    @pytest.mark.asyncio
    async def test_timestamp_stamped_after_failed_refresh(self, tmp_path):
        """A failure must also arm the TTL, otherwise every request retries a dead host."""
        manager, account = make_manager_with_account(tmp_path)
        account.models_cached_at = 0.0

        ctx, client = patch_http(response=http_response(503, {}))
        with ctx, patch("kiro.account_manager.MODEL_CACHE_TTL", 3600):
            await manager._refresh_account_models(account.id)
            await manager._refresh_account_models(account.id)

        assert client.request_with_retry.await_count == 1
        assert account.models_cached_at > 0


# --------------------------------------------------------------------------------------
# Kill switch
# --------------------------------------------------------------------------------------

class TestModelDiscoveryDisabled:
    """MODEL_DISCOVERY=false forces the static list and makes no request."""

    @pytest.mark.asyncio
    async def test_disabled_makes_no_request(self, tmp_path):
        manager, account = make_manager_with_account(tmp_path)
        ctx, client = patch_http(response=http_response(200, live_response_body()))

        with ctx, patch("kiro.account_manager.MODEL_DISCOVERY", False):
            result = await manager._discover_models(account.auth_manager, account.id)

        assert result is None
        assert client.request_with_retry.await_count == 0

    @pytest.mark.asyncio
    async def test_disabled_yields_static_catalog(self, tmp_path):
        manager, account = make_manager_with_account(tmp_path)
        ctx, client = patch_http(response=http_response(200, live_response_body()))

        with ctx, patch("kiro.account_manager.MODEL_DISCOVERY", False):
            await manager._refresh_account_models(account.id, force=True)

        assert client.request_with_retry.await_count == 0
        assert account.model_cache.get_all_model_ids() == [m["modelId"] for m in FALLBACK_MODELS]

    def test_env_parsing_accepts_false_spellings(self, monkeypatch):
        import importlib
        import kiro.config as config_module

        for value in ("false", "FALSE", "0", "no", "off", "disabled"):
            monkeypatch.setenv("MODEL_DISCOVERY", value)
            reloaded = importlib.reload(config_module)
            assert reloaded.MODEL_DISCOVERY is False, value

        monkeypatch.delenv("MODEL_DISCOVERY", raising=False)
        reloaded = importlib.reload(config_module)
        assert reloaded.MODEL_DISCOVERY is True


# --------------------------------------------------------------------------------------
# Hidden models + metadata
# --------------------------------------------------------------------------------------

class TestHiddenModelsAndMetadata:
    """Filtering semantics unchanged; richer metadata retained."""

    @pytest.mark.asyncio
    async def test_auto_hidden_from_list_after_discovery(self, tmp_path):
        manager, account = make_manager_with_account(tmp_path)
        ctx, _ = patch_http(response=http_response(200, live_response_body()))

        with ctx:
            await manager._refresh_account_models(account.id, force=True)

        listed = account.model_resolver.get_available_models()
        assert "auto" not in listed
        assert "auto" in account.model_cache.get_all_model_ids()
        assert "claude-opus-5" in listed

    @pytest.mark.asyncio
    async def test_hidden_from_list_default_unchanged(self):
        assert HIDDEN_FROM_LIST == ["auto"]

    @pytest.mark.asyncio
    async def test_metadata_retained_in_cache(self, tmp_path):
        manager, account = make_manager_with_account(tmp_path)
        ctx, _ = patch_http(response=http_response(200, live_response_body()))

        with ctx:
            await manager._refresh_account_models(account.id, force=True)

        entry = account.model_cache.get("claude-opus-5")
        assert entry is not None
        for field in ModelInfoCache.METADATA_FIELDS:
            assert field in entry, field
        assert entry["rateMultiplier"] == 1.5
        assert entry["rateUnit"] == "REQUEST"
        assert entry["promptCaching"] == {"supported": True}
        assert entry["supportedInputTypes"] == ["TEXT", "IMAGE"]
        assert account.model_cache.get_metadata("claude-opus-5", "rateUnit") == "REQUEST"
        assert account.model_cache.get_max_input_tokens("claude-opus-5") == 300000

    @pytest.mark.asyncio
    async def test_malformed_entries_skipped_not_raised(self):
        cache = ModelInfoCache()
        await cache.update([
            _model_entry("claude-opus-5"),
            {"modelId": ""},
            {"noModelId": True},
            "not-a-dict",
            None,
        ])
        assert cache.get_all_model_ids() == ["claude-opus-5"]

    @pytest.mark.asyncio
    async def test_unknown_model_still_passes_through(self, tmp_path):
        """Discovery must not turn the resolver into a gatekeeper."""
        from kiro.model_resolver import to_runtime_model_id

        manager, account = make_manager_with_account(tmp_path)
        ctx, _ = patch_http(response=http_response(200, live_response_body()))
        with ctx:
            await manager._refresh_account_models(account.id, force=True)

        assert to_runtime_model_id("some-model-we-never-heard-of") == "some-model-we-never-heard-of"
