# -*- coding: utf-8 -*-

"""
Tests for GET /integrations/opencode.json
(see issue-plans/fix-opencode-integration.md).

Fully offline: no request ever leaves the process, and no file is ever written.
Covers auth parity with the other endpoints, the documented OpenCode schema shape,
the guarantee that the configured PROXY_API_KEY never appears in the body, metadata
-> limit mapping (and omission when unknown), hidden-model exclusion / parity with
/v1/models, every query parameter, and request-derived baseURL.
"""

import asyncio
import json

from unittest.mock import MagicMock

import pytest

from kiro.cache import ModelInfoCache
from kiro.config import PROXY_API_KEY


ENDPOINT = "/integrations/opencode.json"

FULL_ENTRY = {
    "modelId": "claude-sonnet-4.5",
    "modelName": "Claude Sonnet 4.5",
    "tokenLimits": {"maxInputTokens": 200000, "maxOutputTokens": 64000},
}
BARE_ENTRY = {"modelId": "bare-model"}


@pytest.fixture
def cache():
    """A ModelInfoCache holding one fully-described and one metadata-free model."""
    c = ModelInfoCache()
    asyncio.run(c.update([FULL_ENTRY, BARE_ENTRY]))
    return c


def _stub_manager(test_client, model_ids, cache):
    """Point app.state at a fake account manager exposing `model_ids` and `cache`."""
    account = MagicMock()
    account.auth_manager = object()  # truthy -> counts as initialized
    account.model_cache = cache

    manager = MagicMock()
    manager.get_all_available_models.return_value = list(model_ids)
    manager.iter_initialized_accounts.side_effect = lambda: iter([account])

    test_client.app.state.account_manager = manager
    test_client.app.state.account_system = True


def call(test_client, valid_proxy_api_key, model_ids, cache, params=None):
    """Call the endpoint with a stubbed catalog, restoring app.state afterwards."""
    original_manager = test_client.app.state.account_manager
    original_system = test_client.app.state.account_system
    _stub_manager(test_client, model_ids, cache)
    try:
        return test_client.get(
            ENDPOINT,
            params=params or {},
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
        )
    finally:
        test_client.app.state.account_manager = original_manager
        test_client.app.state.account_system = original_system


def provider_of(response, provider_id="kiro"):
    return response.json()["provider"][provider_id]


# --------------------------------------------------------------------------------------
# Authentication
# --------------------------------------------------------------------------------------

class TestAuthentication:
    """The endpoint is protected exactly like the other gateway endpoints."""

    def test_missing_key_is_401(self, test_client):
        response = test_client.get(ENDPOINT)
        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid or missing API Key"

    def test_wrong_key_is_401(self, test_client, invalid_proxy_api_key):
        response = test_client.get(
            ENDPOINT, headers={"Authorization": f"Bearer {invalid_proxy_api_key}"}
        )
        assert response.status_code == 401

    def test_401_shape_matches_v1_models(self, test_client):
        assert (
            test_client.get(ENDPOINT).json()
            == test_client.get("/v1/models").json()
        )

    def test_valid_key_is_accepted(self, test_client, valid_proxy_api_key, cache):
        response = call(test_client, valid_proxy_api_key, ["claude-sonnet-4.5"], cache)
        assert response.status_code == 200


# --------------------------------------------------------------------------------------
# Document shape
# --------------------------------------------------------------------------------------

class TestDocumentShape:
    """The emitted document matches the documented OpenCode provider schema."""

    def test_top_level_shape(self, test_client, valid_proxy_api_key, cache):
        body = call(test_client, valid_proxy_api_key, ["claude-sonnet-4.5"], cache).json()
        assert set(body) == {"$schema", "provider"}
        assert body["$schema"] == "https://opencode.ai/config.json"
        assert list(body["provider"]) == ["kiro"]

    def test_npm_is_openai_compatible(self, test_client, valid_proxy_api_key, cache):
        response = call(test_client, valid_proxy_api_key, ["claude-sonnet-4.5"], cache)
        # @ai-sdk/openai targets /v1/responses, which this gateway does not implement.
        assert provider_of(response)["npm"] == "@ai-sdk/openai-compatible"
        assert "@ai-sdk/openai\"" not in response.text

    def test_only_documented_provider_fields(self, test_client, valid_proxy_api_key, cache):
        provider = provider_of(call(test_client, valid_proxy_api_key, ["claude-sonnet-4.5"], cache))
        assert set(provider) == {"npm", "name", "options", "models"}
        assert set(provider["options"]) == {"baseURL", "apiKey"}
        assert provider["name"] == "Kiro Gateway"

    def test_only_documented_model_fields(self, test_client, valid_proxy_api_key, cache):
        models = provider_of(
            call(test_client, valid_proxy_api_key, ["claude-sonnet-4.5"], cache)
        )["models"]
        assert set(models["claude-sonnet-4.5"]) == {"name", "limit"}
        assert set(models["claude-sonnet-4.5"]["limit"]) == {"context", "output"}

    def test_no_invented_metadata_fields(self, test_client, valid_proxy_api_key, cache):
        text = call(test_client, valid_proxy_api_key, ["claude-sonnet-4.5"], cache).text
        for invented in (
            "supported_input_types",
            "supports_prompt_caching",
            "rate_multiplier",
            "attachment",
            "cost",
        ):
            assert invented not in text

    def test_content_type_and_pretty_printed(self, test_client, valid_proxy_api_key, cache):
        response = call(test_client, valid_proxy_api_key, ["claude-sonnet-4.5"], cache)
        assert response.headers["content-type"].startswith("application/json")
        assert "\n  " in response.text  # indented, directly pasteable
        json.loads(response.text)  # still valid JSON


# --------------------------------------------------------------------------------------
# Secret safety
# --------------------------------------------------------------------------------------

class TestApiKeySafety:
    """The server's real key must never reach the document."""

    def test_configured_key_never_in_body(self, test_client, valid_proxy_api_key, cache):
        response = call(test_client, valid_proxy_api_key, ["claude-sonnet-4.5"], cache)
        assert PROXY_API_KEY
        assert PROXY_API_KEY not in response.text

    def test_default_api_key_is_env_placeholder(self, test_client, valid_proxy_api_key, cache):
        provider = provider_of(call(test_client, valid_proxy_api_key, ["claude-sonnet-4.5"], cache))
        assert provider["options"]["apiKey"] == "{env:KIRO_GATEWAY_KEY}"

    def test_configured_key_absent_even_with_api_key_override(
        self, test_client, valid_proxy_api_key, cache
    ):
        response = call(
            test_client, valid_proxy_api_key, ["claude-sonnet-4.5"], cache,
            params={"api_key": "{env:OTHER}"},
        )
        assert PROXY_API_KEY not in response.text
        assert provider_of(response)["options"]["apiKey"] == "{env:OTHER}"

    def test_caller_supplied_secret_is_echoed_verbatim_not_resolved(
        self, test_client, valid_proxy_api_key, cache
    ):
        # A caller passing something secret-looking gets exactly that string back; the
        # server never substitutes its own configured key.
        response = call(
            test_client, valid_proxy_api_key, ["claude-sonnet-4.5"], cache,
            params={"api_key": "sk-looks-real-123"},
        )
        assert provider_of(response)["options"]["apiKey"] == "sk-looks-real-123"
        assert PROXY_API_KEY not in response.text


# --------------------------------------------------------------------------------------
# Limits from metadata
# --------------------------------------------------------------------------------------

class TestLimits:
    """limit comes from upstream metadata and is omitted when unknown."""

    def test_limits_from_metadata(self, test_client, valid_proxy_api_key, cache):
        entry = provider_of(
            call(test_client, valid_proxy_api_key, ["claude-sonnet-4.5"], cache)
        )["models"]["claude-sonnet-4.5"]
        assert entry["limit"] == {"context": 200000, "output": 64000}

    def test_display_name_used_as_model_name(self, test_client, valid_proxy_api_key, cache):
        entry = provider_of(
            call(test_client, valid_proxy_api_key, ["claude-sonnet-4.5"], cache)
        )["models"]["claude-sonnet-4.5"]
        assert entry["name"] == "Claude Sonnet 4.5"

    def test_limit_omitted_when_metadata_unknown(self, test_client, valid_proxy_api_key, cache):
        entry = provider_of(
            call(test_client, valid_proxy_api_key, ["bare-model"], cache)
        )["models"]["bare-model"]
        assert "limit" not in entry  # no guessed value
        assert entry == {"name": "bare-model"}  # falls back to the id


# --------------------------------------------------------------------------------------
# Per-account catalog / parity with /v1/models
# --------------------------------------------------------------------------------------

class TestCatalogParity:
    """The model set is the live per-account set and matches /v1/models exactly."""

    def test_model_set_matches_v1_models(self, test_client, valid_proxy_api_key, cache):
        ids = ["claude-sonnet-4.5", "bare-model"]
        original_manager = test_client.app.state.account_manager
        original_system = test_client.app.state.account_system
        _stub_manager(test_client, ids, cache)
        try:
            headers = {"Authorization": f"Bearer {valid_proxy_api_key}"}
            listed = [m["id"] for m in test_client.get("/v1/models", headers=headers).json()["data"]]
            emitted = list(
                test_client.get(ENDPOINT, headers=headers).json()["provider"]["kiro"]["models"]
            )
        finally:
            test_client.app.state.account_manager = original_manager
            test_client.app.state.account_system = original_system
        assert emitted == listed == ids

    def test_hidden_models_excluded(self, test_client, valid_proxy_api_key, cache):
        # The resolver/manager already filters HIDDEN_FROM_LIST, so "auto" never reaches
        # the visible set the endpoint consumes.
        models = provider_of(
            call(test_client, valid_proxy_api_key, ["claude-sonnet-4.5"], cache)
        )["models"]
        assert "auto" not in models

    def test_smaller_entitlement_yields_fewer_models(self, test_client, valid_proxy_api_key, cache):
        free = provider_of(call(test_client, valid_proxy_api_key, ["bare-model"], cache))["models"]
        paid = provider_of(
            call(test_client, valid_proxy_api_key, ["bare-model", "claude-sonnet-4.5"], cache)
        )["models"]
        assert list(free) == ["bare-model"]
        assert len(paid) == 2


# --------------------------------------------------------------------------------------
# Query parameters
# --------------------------------------------------------------------------------------

class TestQueryParams:

    def test_provider_override(self, test_client, valid_proxy_api_key, cache):
        body = call(
            test_client, valid_proxy_api_key, ["bare-model"], cache,
            params={"provider": "kiro-gw"},
        ).json()
        assert list(body["provider"]) == ["kiro-gw"]

    def test_base_url_derived_from_request_host(self, test_client, valid_proxy_api_key, cache):
        base_url = provider_of(
            call(test_client, valid_proxy_api_key, ["bare-model"], cache)
        )["options"]["baseURL"]
        assert base_url.endswith("/v1")
        assert not base_url.endswith("//v1")
        assert "testserver" in base_url

    def test_base_url_override(self, test_client, valid_proxy_api_key, cache):
        provider = provider_of(call(
            test_client, valid_proxy_api_key, ["bare-model"], cache,
            params={"base_url": "http://gateway:8000/v1"},
        ))
        assert provider["options"]["baseURL"] == "http://gateway:8000/v1"

    def test_reasoning_absent_by_default(self, test_client, valid_proxy_api_key, cache):
        response = call(test_client, valid_proxy_api_key, ["claude-sonnet-4.5", "bare-model"], cache)
        for entry in provider_of(response)["models"].values():
            assert "reasoning" not in entry
            assert "interleaved" not in entry
        assert "reasoning_content" not in response.text

    def test_reasoning_false_explicitly(self, test_client, valid_proxy_api_key, cache):
        entry = provider_of(call(
            test_client, valid_proxy_api_key, ["bare-model"], cache,
            params={"reasoning": "false"},
        ))["models"]["bare-model"]
        assert "reasoning" not in entry

    def test_reasoning_true_shape(self, test_client, valid_proxy_api_key, cache):
        entry = provider_of(call(
            test_client, valid_proxy_api_key, ["claude-sonnet-4.5"], cache,
            params={"reasoning": "true"},
        ))["models"]["claude-sonnet-4.5"]
        assert entry["reasoning"] is True
        assert entry["interleaved"] == {"field": "reasoning_content"}
        assert set(entry) == {"name", "limit", "reasoning", "interleaved"}

    def test_all_params_together(self, test_client, valid_proxy_api_key, cache):
        body = call(
            test_client, valid_proxy_api_key, ["claude-sonnet-4.5"], cache,
            params={
                "provider": "mygw",
                "base_url": "https://host/v1",
                "api_key": "{env:MY_KEY}",
                "reasoning": "true",
            },
        ).json()
        provider = body["provider"]["mygw"]
        assert provider["options"] == {
            "baseURL": "https://host/v1",
            "apiKey": "{env:MY_KEY}",
        }
        assert provider["models"]["claude-sonnet-4.5"]["reasoning"] is True



# --------------------------------------------------------------------------------------
# Alias metadata inheritance
#
# A public alias (e.g. "auto-kiro" -> internal "auto") has no upstream measurement of
# its own, so it inherits the internal model's metadata. The emitted document must
# therefore carry real limits for the alias, under the alias's public id.
# --------------------------------------------------------------------------------------

AUTO_UPSTREAM_ENTRY = {
    "modelId": "auto",
    "modelName": "Auto",
    "tokenLimits": {"maxInputTokens": 1000000, "maxOutputTokens": 64000},
}


class TestAliasInheritanceInDocument:
    def _cache_with_alias(self):
        c = ModelInfoCache()
        asyncio.run(c.update([AUTO_UPSTREAM_ENTRY, FULL_ENTRY]))
        c.add_hidden_model("auto-kiro", "auto")
        return c

    def test_alias_carries_inherited_limits(self, test_client, valid_proxy_api_key):
        cache = self._cache_with_alias()
        response = call(
            test_client, valid_proxy_api_key, ["auto-kiro", FULL_ENTRY["modelId"]], cache
        )

        models = provider_of(response)["models"]

        assert models["auto-kiro"]["name"] == "Auto"
        assert models["auto-kiro"]["limit"] == {"context": 1000000, "output": 64000}

    def test_alias_keyed_by_public_id_not_internal(self, test_client, valid_proxy_api_key):
        cache = self._cache_with_alias()
        response = call(
            test_client, valid_proxy_api_key, ["auto-kiro", FULL_ENTRY["modelId"]], cache
        )

        models = provider_of(response)["models"]

        assert "auto-kiro" in models
        assert "auto" not in models

    def test_alias_without_inheritable_metadata_omits_limit(
        self, test_client, valid_proxy_api_key
    ):
        """Nothing is fabricated when the internal model carries no metadata."""
        cache = ModelInfoCache()
        asyncio.run(cache.update([BARE_ENTRY]))
        cache.add_hidden_model("bare-alias", BARE_ENTRY["modelId"])

        response = call(test_client, valid_proxy_api_key, ["bare-alias"], cache)

        assert "limit" not in provider_of(response)["models"]["bare-alias"]

    def test_document_and_models_endpoint_agree_on_alias(
        self, test_client, valid_proxy_api_key
    ):
        """The opencode document and /v1/models must not disagree about the alias."""
        cache = self._cache_with_alias()
        ids = ["auto-kiro", FULL_ENTRY["modelId"]]

        doc = call(test_client, valid_proxy_api_key, ids, cache)
        doc_models = provider_of(doc)["models"]

        original_manager = test_client.app.state.account_manager
        original_system = test_client.app.state.account_system
        _stub_manager(test_client, ids, cache)
        try:
            listed = test_client.get(
                "/v1/models",
                headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            ).json()["data"]
        finally:
            test_client.app.state.account_manager = original_manager
            test_client.app.state.account_system = original_system

        assert set(doc_models) == {m["id"] for m in listed}

        alias = next(m for m in listed if m["id"] == "auto-kiro")
        assert alias["max_input_tokens"] == doc_models["auto-kiro"]["limit"]["context"]
        assert alias["max_output_tokens"] == doc_models["auto-kiro"]["limit"]["output"]



class TestAliasKeyMetadataResolution:
    """
    MODEL_ALIASES keys are NOT cache entries.

    ModelResolver.get_available_models() adds `self.aliases.keys()` to the visible set,
    so an id like "auto-kiro" is advertised without ever existing in the model cache.
    A direct cache lookup therefore misses, and the metadata must be resolved through
    the alias target. This is a different mechanism from add_hidden_model() and was the
    cause of "auto-kiro" shipping with no limits.
    """

    def test_alias_key_resolves_metadata_via_target(
        self, test_client, valid_proxy_api_key, monkeypatch
    ):
        monkeypatch.setattr("kiro.routes_openai.MODEL_ALIASES", {"auto-kiro": "auto"})

        cache = ModelInfoCache()
        asyncio.run(cache.update([AUTO_UPSTREAM_ENTRY]))  # note: no "auto-kiro" entry

        assert cache.get_public_metadata("auto-kiro") == {}, "precondition: alias not cached"

        response = call(test_client, valid_proxy_api_key, ["auto-kiro"], cache)
        entry = provider_of(response)["models"]["auto-kiro"]

        assert entry["name"] == "Auto"
        assert entry["limit"] == {"context": 1000000, "output": 64000}

    def test_alias_resolution_is_single_hop(
        self, test_client, valid_proxy_api_key, monkeypatch
    ):
        """A chained alias must not be walked, so cyclic config cannot loop."""
        monkeypatch.setattr(
            "kiro.routes_openai.MODEL_ALIASES", {"a": "b", "b": "auto"}
        )

        cache = ModelInfoCache()
        asyncio.run(cache.update([AUTO_UPSTREAM_ENTRY]))

        response = call(test_client, valid_proxy_api_key, ["a"], cache)

        assert "limit" not in provider_of(response)["models"]["a"]

    def test_self_referential_alias_terminates(
        self, test_client, valid_proxy_api_key, monkeypatch
    ):
        monkeypatch.setattr("kiro.routes_openai.MODEL_ALIASES", {"loop": "loop"})

        cache = ModelInfoCache()
        asyncio.run(cache.update([AUTO_UPSTREAM_ENTRY]))

        response = call(test_client, valid_proxy_api_key, ["loop"], cache)

        assert "limit" not in provider_of(response)["models"]["loop"]

    def test_alias_to_unknown_target_invents_nothing(
        self, test_client, valid_proxy_api_key, monkeypatch
    ):
        monkeypatch.setattr("kiro.routes_openai.MODEL_ALIASES", {"ghost": "not-in-catalog"})

        cache = ModelInfoCache()
        asyncio.run(cache.update([AUTO_UPSTREAM_ENTRY]))

        response = call(test_client, valid_proxy_api_key, ["ghost"], cache)

        assert "limit" not in provider_of(response)["models"]["ghost"]

    def test_non_alias_model_is_unaffected(
        self, test_client, valid_proxy_api_key, monkeypatch
    ):
        monkeypatch.setattr("kiro.routes_openai.MODEL_ALIASES", {"auto-kiro": "auto"})

        cache = ModelInfoCache()
        asyncio.run(cache.update([FULL_ENTRY, BARE_ENTRY]))

        response = call(
            test_client,
            valid_proxy_api_key,
            [FULL_ENTRY["modelId"], BARE_ENTRY["modelId"]],
            cache,
        )
        models = provider_of(response)["models"]

        assert models[FULL_ENTRY["modelId"]]["limit"] == {"context": 200000, "output": 64000}
        assert "limit" not in models[BARE_ENTRY["modelId"]]
