# -*- coding: utf-8 -*-

"""
Tests for upstream model metadata on /v1/models and catalog pagination
(issue-plans/fix-156-expose-model-metadata.md, upstream issue #156).

Fully offline: no request ever leaves the process. Covers
- additive metadata fields on /v1/models (limits, input types, caching, rate, name),
- static FALLBACK_MODELS entries rendering exactly as before (no fabricated values),
- the five pre-existing fields and the list envelope staying unchanged,
- HIDDEN_FROM_LIST filtering ("auto" stays hidden),
- nextToken pagination, the page cap, the model cap, and every fallback path.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from kiro.account_manager import (
    MODEL_DISCOVERY_MAX_MODELS,
    MODEL_DISCOVERY_MAX_PAGES,
)
from kiro.cache import ModelInfoCache
from kiro.config import FALLBACK_MODELS, HIDDEN_FROM_LIST

from tests.unit.test_model_discovery import (
    http_response,
    make_manager_with_account,
    patch_http,
)


BASE_FIELDS = {"id", "object", "created", "owned_by", "description"}

FULL_METADATA_ENTRY = {
    "modelId": "claude-sonnet-4.5",
    "modelName": "Claude Sonnet 4.5",
    "description": "Claude Sonnet 4.5 model",
    "tokenLimits": {"maxInputTokens": 200000, "maxOutputTokens": 64000},
    "supportedInputTypes": ["TEXT", "IMAGE"],
    "promptCaching": {
        "supportsPromptCaching": True,
        "maximumCacheCheckpointsPerRequest": 4,
        "minimumTokensPerCacheCheckpoint": 1024,
    },
    "rateMultiplier": 1.3,
    "rateUnit": "Credit",
    "availableOrigins": None,
    "additionalModelRequestFieldsSchema": None,
}


def models_response(test_client, valid_proxy_api_key, model_ids, cache):
    """Call /v1/models with a stubbed account manager backed by `cache`."""
    account = MagicMock()
    account.auth_manager = object()  # truthy → counts as initialized
    account.model_cache = cache

    manager = MagicMock()
    manager.get_all_available_models.return_value = list(model_ids)
    manager.iter_initialized_accounts.return_value = iter([account])

    original_manager = test_client.app.state.account_manager
    original_system = test_client.app.state.account_system
    test_client.app.state.account_manager = manager
    test_client.app.state.account_system = True
    try:
        return test_client.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
        )
    finally:
        test_client.app.state.account_manager = original_manager
        test_client.app.state.account_system = original_system


# --------------------------------------------------------------------------------------
# Cache-level normalization
# --------------------------------------------------------------------------------------

class TestPublicMetadata:
    """get_public_metadata() maps upstream fields and omits everything unknown."""

    @pytest.mark.asyncio
    async def test_full_metadata_is_mapped(self):
        cache = ModelInfoCache()
        await cache.update([FULL_METADATA_ENTRY])

        assert cache.get_public_metadata("claude-sonnet-4.5") == {
            "display_name": "Claude Sonnet 4.5",
            "context_length": 200000,
            "max_input_tokens": 200000,
            "max_output_tokens": 64000,
            "supported_input_types": ["TEXT", "IMAGE"],
            "supports_prompt_caching": True,
            "max_cache_checkpoints": 4,
            "min_tokens_per_cache_checkpoint": 1024,
            "rate_multiplier": 1.3,
            "rate_unit": "Credit",
        }

    @pytest.mark.asyncio
    async def test_metadata_free_entry_yields_empty_dict(self):
        cache = ModelInfoCache()
        await cache.update([{"modelId": "bare-model"}])

        assert cache.get_public_metadata("bare-model") == {}

    @pytest.mark.asyncio
    async def test_unknown_model_yields_empty_dict(self):
        cache = ModelInfoCache()
        await cache.update([FULL_METADATA_ENTRY])

        assert cache.get_public_metadata("nope") == {}

    @pytest.mark.asyncio
    async def test_partial_and_malformed_values_are_omitted(self):
        cache = ModelInfoCache()
        await cache.update([{
            "modelId": "partial",
            "modelName": "",
            "tokenLimits": {"maxInputTokens": 1000000, "maxOutputTokens": None},
            "supportedInputTypes": [],
            "promptCaching": {"supportsPromptCaching": "yes"},
            "rateUnit": None,
        }])

        assert cache.get_public_metadata("partial") == {
            "context_length": 1000000,
            "max_input_tokens": 1000000,
        }

    @pytest.mark.asyncio
    async def test_text_only_input_types_are_preserved(self):
        cache = ModelInfoCache()
        await cache.update([{"modelId": "text-only", "supportedInputTypes": ["TEXT"]}])

        assert cache.get_public_metadata("text-only")["supported_input_types"] == ["TEXT"]

    @pytest.mark.asyncio
    async def test_hidden_synthetic_entry_reports_no_metadata(self):
        """
        A synthetic alias inherits from its internal model, so when that internal model
        is absent from the cache there is nothing to inherit and the result is empty.
        The alias's own tokenLimits are a local default, never published.
        """
        cache = ModelInfoCache()
        await cache.update([])
        cache.add_hidden_model("claude-3.7-sonnet", "CLAUDE_3_7_SONNET_20250219_V1_0")

        assert cache.get_public_metadata("claude-3.7-sonnet") == {}


# --------------------------------------------------------------------------------------
# /v1/models response
# --------------------------------------------------------------------------------------

class TestModelsEndpointMetadata:
    """New fields are additive; the old contract is byte-identical."""

    def test_full_metadata_model_exposes_all_fields(self, test_client, valid_proxy_api_key):
        cache = ModelInfoCache()
        asyncio.run(cache.update([FULL_METADATA_ENTRY]))

        response = models_response(
            test_client, valid_proxy_api_key, ["claude-sonnet-4.5"], cache
        )

        assert response.status_code == 200
        entry = response.json()["data"][0]
        assert entry["max_input_tokens"] == 200000
        assert entry["context_length"] == 200000
        assert entry["max_output_tokens"] == 64000
        assert entry["supported_input_types"] == ["TEXT", "IMAGE"]
        assert entry["supports_prompt_caching"] is True
        assert entry["max_cache_checkpoints"] == 4
        assert entry["min_tokens_per_cache_checkpoint"] == 1024
        assert entry["rate_multiplier"] == 1.3
        assert entry["rate_unit"] == "Credit"
        assert entry["display_name"] == "Claude Sonnet 4.5"

    def test_static_fallback_model_renders_exactly_as_before(self, test_client, valid_proxy_api_key):
        """A FALLBACK_MODELS entry has only modelId → no invented limits."""
        cache = ModelInfoCache()
        asyncio.run(cache.update(FALLBACK_MODELS))
        fallback_id = FALLBACK_MODELS[0]["modelId"]

        response = models_response(test_client, valid_proxy_api_key, [fallback_id], cache)

        assert response.status_code == 200
        entry = response.json()["data"][0]
        assert set(entry.keys()) == BASE_FIELDS
        for field in (
            "context_length", "max_input_tokens", "max_output_tokens",
            "supported_input_types", "supports_prompt_caching",
            "rate_multiplier", "rate_unit", "display_name",
        ):
            assert field not in entry

    def test_preexisting_fields_unchanged_for_both_cases(self, test_client, valid_proxy_api_key):
        cache = ModelInfoCache()
        asyncio.run(cache.update([FULL_METADATA_ENTRY, {"modelId": "bare-model"}]))

        response = models_response(
            test_client, valid_proxy_api_key, ["claude-sonnet-4.5", "bare-model"], cache
        )

        assert response.status_code == 200
        for entry in response.json()["data"]:
            assert BASE_FIELDS <= set(entry.keys())
            assert entry["object"] == "model"
            assert entry["owned_by"] == "anthropic"
            # description keeps its original, model-independent wording
            assert entry["description"] == "Claude model via Kiro API"
            assert isinstance(entry["created"], int)
        ids = [entry["id"] for entry in response.json()["data"]]
        assert ids == ["claude-sonnet-4.5", "bare-model"]

    def test_envelope_unchanged(self, test_client, valid_proxy_api_key):
        cache = ModelInfoCache()
        asyncio.run(cache.update([FULL_METADATA_ENTRY]))

        response = models_response(
            test_client, valid_proxy_api_key, ["claude-sonnet-4.5"], cache
        )

        body = response.json()
        assert set(body.keys()) == {"object", "data"}
        assert body["object"] == "list"
        assert isinstance(body["data"], list)

    def test_hidden_from_list_models_stay_hidden(self, test_client, valid_proxy_api_key):
        """The endpoint lists exactly what the resolver exposes; 'auto' is filtered."""
        from kiro.model_resolver import ModelResolver

        cache = ModelInfoCache()
        asyncio.run(cache.update([{"modelId": "auto"}, FULL_METADATA_ENTRY]))
        resolver = ModelResolver(
            cache=cache, hidden_models={}, aliases={}, hidden_from_list=HIDDEN_FROM_LIST
        )
        visible = resolver.get_available_models()
        assert "auto" not in visible

        response = models_response(test_client, valid_proxy_api_key, visible, cache)

        ids = [entry["id"] for entry in response.json()["data"]]
        assert "auto" not in ids
        assert "claude-sonnet-4.5" in ids

    def test_metadata_failure_does_not_break_endpoint(self, test_client, valid_proxy_api_key):
        broken = MagicMock()
        broken.get_public_metadata.side_effect = RuntimeError("boom")

        response = models_response(
            test_client, valid_proxy_api_key, ["claude-sonnet-4.5"], broken
        )

        assert response.status_code == 200
        assert set(response.json()["data"][0].keys()) == BASE_FIELDS


# --------------------------------------------------------------------------------------
# Discovery pagination
# --------------------------------------------------------------------------------------

def _page(model_ids, next_token=None):
    body = {"models": [{"modelId": mid, "modelName": mid} for mid in model_ids]}
    if next_token:
        body["nextToken"] = next_token
    return http_response(200, body)


class TestDiscoveryPagination:
    """nextToken is followed, with hard page and model caps."""

    @pytest.mark.asyncio
    async def test_two_pages_are_merged(self, tmp_path):
        manager, account = make_manager_with_account(tmp_path)
        ctx, client = patch_http(side_effect=[
            _page(["model-a", "model-b"], next_token="tok-2"),
            _page(["model-c"]),
        ])

        with ctx:
            models = await manager._discover_models(account.auth_manager, account.id)

        assert [m["modelId"] for m in models] == ["model-a", "model-b", "model-c"]
        assert client.request_with_retry.await_count == 2
        first, second = client.request_with_retry.await_args_list
        # Page 1 request shape is unchanged (no extra params).
        assert first.kwargs["params"] == {"origin": "AI_EDITOR"}
        assert second.kwargs["params"] == {"origin": "AI_EDITOR", "nextToken": "tok-2"}

    @pytest.mark.asyncio
    async def test_single_page_issues_one_request(self, tmp_path):
        manager, account = make_manager_with_account(tmp_path)
        ctx, client = patch_http(response=_page(["model-a"]))

        with ctx:
            models = await manager._discover_models(account.auth_manager, account.id)

        assert len(models) == 1
        assert client.request_with_retry.await_count == 1

    @pytest.mark.asyncio
    async def test_duplicate_ids_across_pages_are_deduplicated(self, tmp_path):
        manager, account = make_manager_with_account(tmp_path)
        ctx, _ = patch_http(side_effect=[
            _page(["model-a"], next_token="tok-2"),
            _page(["model-a", "model-b"]),
        ])

        with ctx:
            models = await manager._discover_models(account.auth_manager, account.id)

        assert [m["modelId"] for m in models] == ["model-a", "model-b"]

    @pytest.mark.asyncio
    async def test_page_cap_stops_infinite_next_token(self, tmp_path):
        """A hostile upstream that always returns nextToken cannot loop forever."""
        manager, account = make_manager_with_account(tmp_path)
        counter = {"n": 0}

        def always_more(*args, **kwargs):
            counter["n"] += 1
            return _page([f"model-{counter['n']}"], next_token="always")

        ctx, client = patch_http(side_effect=always_more)

        with ctx:
            models = await manager._discover_models(account.auth_manager, account.id)

        assert client.request_with_retry.await_count == MODEL_DISCOVERY_MAX_PAGES
        assert len(models) == MODEL_DISCOVERY_MAX_PAGES

    @pytest.mark.asyncio
    async def test_model_cap_stops_oversized_catalog(self, tmp_path):
        manager, account = make_manager_with_account(tmp_path)
        page_size = MODEL_DISCOVERY_MAX_MODELS  # one page already reaches the cap
        ctx, client = patch_http(side_effect=[
            _page([f"model-{i}" for i in range(page_size)], next_token="tok-2"),
            _page(["never-fetched"]),
        ])

        with ctx:
            models = await manager._discover_models(account.auth_manager, account.id)

        assert len(models) == MODEL_DISCOVERY_MAX_MODELS
        assert client.request_with_retry.await_count == 1

    @pytest.mark.asyncio
    async def test_later_page_failure_keeps_earlier_results(self, tmp_path):
        manager, account = make_manager_with_account(tmp_path)
        ctx, _ = patch_http(side_effect=[
            _page(["model-a"], next_token="tok-2"),
            http_response(500, None),
        ])

        with ctx:
            models = await manager._discover_models(account.auth_manager, account.id)

        assert [m["modelId"] for m in models] == ["model-a"]


class TestPaginationPreservesFallbacks:
    """Every pre-existing fallback guarantee still holds after the rewrite."""

    @pytest.mark.asyncio
    async def test_non_200_falls_back(self, tmp_path):
        manager, account = make_manager_with_account(tmp_path)
        ctx, _ = patch_http(response=http_response(403, None))

        with ctx:
            assert await manager._discover_models(account.auth_manager, account.id) is None

    @pytest.mark.asyncio
    async def test_non_dict_body_falls_back(self, tmp_path):
        manager, account = make_manager_with_account(tmp_path)
        ctx, _ = patch_http(response=http_response(200, ["not", "a", "dict"]))

        with ctx:
            assert await manager._discover_models(account.auth_manager, account.id) is None

    @pytest.mark.asyncio
    async def test_unparseable_body_falls_back(self, tmp_path):
        manager, account = make_manager_with_account(tmp_path)
        ctx, _ = patch_http(response=http_response(200, raises=ValueError("bad json")))

        with ctx:
            assert await manager._discover_models(account.auth_manager, account.id) is None

    @pytest.mark.asyncio
    async def test_empty_models_list_falls_back(self, tmp_path):
        manager, account = make_manager_with_account(tmp_path)
        ctx, _ = patch_http(response=http_response(200, {"models": []}))

        with ctx:
            assert await manager._discover_models(account.auth_manager, account.id) is None

    @pytest.mark.asyncio
    async def test_entries_without_model_id_fall_back(self, tmp_path):
        manager, account = make_manager_with_account(tmp_path)
        ctx, _ = patch_http(response=http_response(200, {"models": [{"name": "x"}, 42]}))

        with ctx:
            assert await manager._discover_models(account.auth_manager, account.id) is None

    @pytest.mark.asyncio
    async def test_timeout_falls_back_and_does_not_block(self, tmp_path):
        manager, account = make_manager_with_account(tmp_path)
        ctx, _ = patch_http(side_effect=asyncio.TimeoutError())

        with ctx:
            assert await manager._discover_models(account.auth_manager, account.id) is None

    @pytest.mark.asyncio
    async def test_transport_exception_falls_back(self, tmp_path):
        manager, account = make_manager_with_account(tmp_path)
        ctx, _ = patch_http(side_effect=ConnectionError("no route to host"))

        with ctx:
            assert await manager._discover_models(account.auth_manager, account.id) is None



# --------------------------------------------------------------------------------------
# Alias metadata inheritance
#
# Public aliases created by add_hidden_model() (e.g. "auto-kiro" -> internal "auto")
# carry no upstream measurement of their own: their tokenLimits are a local default.
# They therefore inherit the metadata of the internal model they point at, so an alias
# reports real limits instead of nothing. Only metadata is inherited; the alias always
# keeps its own public id.
# --------------------------------------------------------------------------------------

AUTO_UPSTREAM_ENTRY = {
    "modelId": "auto",
    "modelName": "Auto",
    "description": "Models chosen by task for optimal usage and consistent quality",
    "tokenLimits": {"maxInputTokens": 1000000, "maxOutputTokens": 64000},
    "supportedInputTypes": ["TEXT", "IMAGE"],
    "promptCaching": {
        "supportsPromptCaching": True,
        "maximumCacheCheckpointsPerRequest": 4,
        "minimumTokensPerCacheCheckpoint": 1024,
    },
    "rateMultiplier": 1.0,
    "rateUnit": "Credit",
}


class TestAliasMetadataInheritance:
    """add_hidden_model() aliases inherit their internal model's upstream metadata."""

    @pytest.mark.asyncio
    async def test_alias_inherits_display_name_and_limits(self):
        cache = ModelInfoCache()
        await cache.update([AUTO_UPSTREAM_ENTRY])
        cache.add_hidden_model("auto-kiro", "auto")

        meta = cache.get_public_metadata("auto-kiro")

        assert meta["display_name"] == "Auto"
        assert meta["max_input_tokens"] == 1000000
        assert meta["max_output_tokens"] == 64000
        assert meta["context_length"] == 1000000

    @pytest.mark.asyncio
    async def test_alias_inherits_full_metadata_set(self):
        cache = ModelInfoCache()
        await cache.update([AUTO_UPSTREAM_ENTRY])
        cache.add_hidden_model("auto-kiro", "auto")

        assert cache.get_public_metadata("auto-kiro") == cache.get_public_metadata("auto")

    @pytest.mark.asyncio
    async def test_alias_keeps_its_own_public_id(self):
        """Inheriting metadata must never leak the internal id into the public view."""
        cache = ModelInfoCache()
        await cache.update([AUTO_UPSTREAM_ENTRY])
        cache.add_hidden_model("auto-kiro", "auto")

        meta = cache.get_public_metadata("auto-kiro")

        assert "auto" not in meta.values()
        assert meta.get("display_name") != "auto-kiro"
        assert cache.get("auto-kiro")["modelId"] == "auto-kiro"

    @pytest.mark.asyncio
    async def test_alias_to_metadata_free_model_invents_nothing(self):
        """A static FALLBACK_MODELS target has nothing to offer, so the result is empty."""
        cache = ModelInfoCache()
        await cache.update([{"modelId": "bare"}])
        cache.add_hidden_model("bare-alias", "bare")

        assert cache.get_public_metadata("bare-alias") == {}

    @pytest.mark.asyncio
    async def test_alias_to_unknown_internal_model_invents_nothing(self):
        cache = ModelInfoCache()
        await cache.update([AUTO_UPSTREAM_ENTRY])
        cache.add_hidden_model("ghost-alias", "not-in-catalog")

        assert cache.get_public_metadata("ghost-alias") == {}

    @pytest.mark.asyncio
    async def test_alias_chain_does_not_inherit(self):
        """A synthetic entry pointing at another synthetic entry has nothing measured."""
        cache = ModelInfoCache()
        await cache.update([AUTO_UPSTREAM_ENTRY])
        cache.add_hidden_model("first-alias", "auto")
        cache.add_hidden_model("second-alias", "first-alias")

        assert cache.get_public_metadata("second-alias") == {}

    @pytest.mark.asyncio
    async def test_self_referential_alias_does_not_recurse(self):
        cache = ModelInfoCache()
        await cache.update([AUTO_UPSTREAM_ENTRY])
        cache.add_hidden_model("loop", "loop")

        assert cache.get_public_metadata("loop") == {}

    @pytest.mark.asyncio
    async def test_real_model_metadata_is_unaffected(self):
        """Inheritance must only apply to synthetic entries."""
        cache = ModelInfoCache()
        await cache.update(
            [AUTO_UPSTREAM_ENTRY, {"modelId": "real", "modelName": "Real", "rateMultiplier": 2.2}]
        )
        cache.add_hidden_model("auto-kiro", "auto")

        meta = cache.get_public_metadata("real")

        assert meta["display_name"] == "Real"
        assert meta["rate_multiplier"] == 2.2
        assert "context_length" not in meta
