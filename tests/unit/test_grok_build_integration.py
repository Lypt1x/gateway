# -*- coding: utf-8 -*-

"""
Tests for GET /integrations/grok-build.toml and the additive `context_window` field on
/v1/models (see issue-plans/fix-grok-build-integration.md).

Fully offline: no request leaves the process, nothing is written to disk, and no Grok
or xAI binary is invoked.
"""

import asyncio
import tomllib

from unittest.mock import MagicMock

import pytest

from kiro.cache import ModelInfoCache
from kiro.config import PROXY_API_KEY


ENDPOINT = "/integrations/grok-build.toml"

FULL_ENTRY = {
    "modelId": "claude-sonnet-4.5",
    "modelName": "Claude Sonnet 4.5",
    "tokenLimits": {"maxInputTokens": 200000, "maxOutputTokens": 64000},
}
BARE_ENTRY = {"modelId": "bare-model"}


@pytest.fixture
def cache():
    c = ModelInfoCache()
    asyncio.run(c.update([FULL_ENTRY, BARE_ENTRY]))
    return c


def _stub_manager(test_client, model_ids, cache):
    account = MagicMock()
    account.auth_manager = object()
    account.model_cache = cache

    manager = MagicMock()
    manager.get_all_available_models.return_value = list(model_ids)
    manager.iter_initialized_accounts.side_effect = lambda: iter([account])

    test_client.app.state.account_manager = manager
    test_client.app.state.account_system = True


def call(test_client, valid_proxy_api_key, model_ids, cache, params=None, path=ENDPOINT):
    original_manager = test_client.app.state.account_manager
    original_system = test_client.app.state.account_system
    _stub_manager(test_client, model_ids, cache)
    try:
        return test_client.get(
            path,
            params=params or {},
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
        )
    finally:
        test_client.app.state.account_manager = original_manager
        test_client.app.state.account_system = original_system


def doc(response) -> dict:
    return tomllib.loads(response.text)


# --------------------------------------------------------------------------------------
# /v1/models: additive context_window
# --------------------------------------------------------------------------------------

class TestModelsContextWindow:
    """Grok's prefetch reads context_window; the five legacy fields stay untouched."""

    def _entry(self, test_client, key, model_ids, cache):
        body = call(test_client, key, model_ids, cache, path="/v1/models").json()
        return {m["id"]: m for m in body["data"]}

    def test_context_window_present_when_known(self, test_client, valid_proxy_api_key, cache):
        entry = self._entry(
            test_client, valid_proxy_api_key, ["claude-sonnet-4.5"], cache
        )["claude-sonnet-4.5"]
        assert entry["context_window"] == 200000

    def test_context_window_matches_legacy_input_limit(
        self, test_client, valid_proxy_api_key, cache
    ):
        entry = self._entry(
            test_client, valid_proxy_api_key, ["claude-sonnet-4.5"], cache
        )["claude-sonnet-4.5"]
        assert entry["context_window"] == entry["max_input_tokens"] == entry["context_length"]

    def test_context_window_omitted_when_unknown(self, test_client, valid_proxy_api_key, cache):
        entry = self._entry(test_client, valid_proxy_api_key, ["bare-model"], cache)["bare-model"]
        assert "context_window" not in entry  # never fabricated

    def test_legacy_fields_and_envelope_unchanged(self, test_client, valid_proxy_api_key, cache):
        body = call(
            test_client, valid_proxy_api_key, ["claude-sonnet-4.5"], cache, path="/v1/models"
        ).json()
        assert set(body) == {"object", "data"}
        assert body["object"] == "list"
        entry = body["data"][0]
        assert entry["id"] == "claude-sonnet-4.5"
        assert entry["object"] == "model"
        assert entry["owned_by"] == "anthropic"
        assert entry["display_name"] == "Claude Sonnet 4.5"
        assert entry["context_length"] == 200000
        assert entry["max_input_tokens"] == 200000
        assert entry["max_output_tokens"] == 64000

    def test_bare_model_entry_gains_nothing(self, test_client, valid_proxy_api_key, cache):
        entry = self._entry(test_client, valid_proxy_api_key, ["bare-model"], cache)["bare-model"]
        assert set(entry) == {"id", "object", "created", "owned_by", "description"}


# --------------------------------------------------------------------------------------
# Authentication
# --------------------------------------------------------------------------------------

class TestAuthentication:

    def test_missing_key_is_401(self, test_client):
        response = test_client.get(ENDPOINT)
        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid or missing API Key"

    def test_401_shape_matches_v1_models(self, test_client):
        assert test_client.get(ENDPOINT).json() == test_client.get("/v1/models").json()

    def test_valid_key_is_accepted(self, test_client, valid_proxy_api_key, cache):
        response = call(test_client, valid_proxy_api_key, ["claude-sonnet-4.5"], cache)
        assert response.status_code == 200


# --------------------------------------------------------------------------------------
# Document shape
# --------------------------------------------------------------------------------------

class TestDocumentShape:

    def test_content_type_is_toml(self, test_client, valid_proxy_api_key, cache):
        response = call(test_client, valid_proxy_api_key, ["bare-model"], cache)
        content_type = response.headers["content-type"]
        assert content_type.startswith("application/toml")
        assert "utf-8" in content_type

    def test_parses_as_toml(self, test_client, valid_proxy_api_key, cache):
        body = doc(call(test_client, valid_proxy_api_key, ["claude-sonnet-4.5", "bare-model"], cache))
        assert set(body) == {"endpoints", "models", "model"}

    def test_models_base_url_ends_in_v1(self, test_client, valid_proxy_api_key, cache):
        base = doc(call(test_client, valid_proxy_api_key, ["bare-model"], cache))["endpoints"][
            "models_base_url"
        ]
        assert base.endswith("/v1")
        assert not base.endswith("//v1")
        assert "testserver" in base

    def test_api_backend_is_chat_completions(self, test_client, valid_proxy_api_key, cache):
        body = doc(call(test_client, valid_proxy_api_key, ["claude-sonnet-4.5", "bare-model"], cache))
        for table in body["model"].values():
            assert table["api_backend"] == "chat_completions"

    def test_default_model_is_declared(self, test_client, valid_proxy_api_key, cache):
        body = doc(call(test_client, valid_proxy_api_key, ["claude-sonnet-4.5", "bare-model"], cache))
        assert body["models"]["default"] == "claude-sonnet-4.5"

    def test_one_table_per_visible_model_keyed_by_full_id(
        self, test_client, valid_proxy_api_key, cache
    ):
        body = doc(call(test_client, valid_proxy_api_key, ["claude-sonnet-4.5", "bare-model"], cache))
        # The dotted id must be quoted in the header, else TOML nests it.
        assert set(body["model"]) == {"claude-sonnet-4.5", "bare-model"}
        assert body["model"]["claude-sonnet-4.5"]["model"] == "claude-sonnet-4.5"
        assert body["model"]["claude-sonnet-4.5"]["name"] == "Claude Sonnet 4.5"

    def test_context_window_present_only_when_known(
        self, test_client, valid_proxy_api_key, cache
    ):
        tables = doc(
            call(test_client, valid_proxy_api_key, ["claude-sonnet-4.5", "bare-model"], cache)
        )["model"]
        assert tables["claude-sonnet-4.5"]["context_window"] == 200000
        assert tables["claude-sonnet-4.5"]["max_completion_tokens"] == 64000
        assert "context_window" not in tables["bare-model"]
        assert "max_completion_tokens" not in tables["bare-model"]

    def test_no_stream_tool_calls_key(self, test_client, valid_proxy_api_key, cache):
        response = call(test_client, valid_proxy_api_key, ["claude-sonnet-4.5", "bare-model"], cache)
        assert "stream_tool_calls" not in response.text

    def test_empty_catalog_still_valid_toml(self, test_client, valid_proxy_api_key, cache):
        body = doc(call(test_client, valid_proxy_api_key, [], cache))
        assert body["endpoints"]["models_base_url"].endswith("/v1")
        assert "models" not in body  # no fabricated default


# --------------------------------------------------------------------------------------
# Secret safety
# --------------------------------------------------------------------------------------

class TestSecretSafety:

    def test_configured_key_never_in_body(self, test_client, valid_proxy_api_key, cache):
        response = call(test_client, valid_proxy_api_key, ["claude-sonnet-4.5"], cache)
        assert PROXY_API_KEY
        assert PROXY_API_KEY not in response.text

    def test_only_env_var_name_is_emitted(self, test_client, valid_proxy_api_key, cache):
        tables = doc(call(test_client, valid_proxy_api_key, ["bare-model"], cache))["model"]
        assert tables["bare-model"]["env_key"] == "KIRO_GATEWAY_KEY"
        assert "api_key" not in tables["bare-model"]

    def test_env_key_override(self, test_client, valid_proxy_api_key, cache):
        response = call(
            test_client, valid_proxy_api_key, ["bare-model"], cache,
            params={"env_key": "MY_GW_KEY"},
        )
        assert doc(response)["model"]["bare-model"]["env_key"] == "MY_GW_KEY"
        assert PROXY_API_KEY not in response.text

    def test_base_url_override(self, test_client, valid_proxy_api_key, cache):
        body = doc(call(
            test_client, valid_proxy_api_key, ["bare-model"], cache,
            params={"base_url": "http://gateway:8000/v1"},
        ))
        assert body["endpoints"]["models_base_url"] == "http://gateway:8000/v1"


# --------------------------------------------------------------------------------------
# Parity with /v1/models
# --------------------------------------------------------------------------------------

class TestCatalogParity:

    def test_model_set_matches_v1_models(self, test_client, valid_proxy_api_key, cache):
        ids = ["claude-sonnet-4.5", "bare-model"]
        original_manager = test_client.app.state.account_manager
        original_system = test_client.app.state.account_system
        _stub_manager(test_client, ids, cache)
        try:
            headers = {"Authorization": f"Bearer {valid_proxy_api_key}"}
            listed = [m["id"] for m in test_client.get("/v1/models", headers=headers).json()["data"]]
            emitted = list(tomllib.loads(test_client.get(ENDPOINT, headers=headers).text)["model"])
        finally:
            test_client.app.state.account_manager = original_manager
            test_client.app.state.account_system = original_system
        assert emitted == listed == ids

    def test_context_window_agrees_with_v1_models(self, test_client, valid_proxy_api_key, cache):
        ids = ["claude-sonnet-4.5"]
        original_manager = test_client.app.state.account_manager
        original_system = test_client.app.state.account_system
        _stub_manager(test_client, ids, cache)
        try:
            headers = {"Authorization": f"Bearer {valid_proxy_api_key}"}
            listed = test_client.get("/v1/models", headers=headers).json()["data"][0]
            table = tomllib.loads(test_client.get(ENDPOINT, headers=headers).text)["model"]
        finally:
            test_client.app.state.account_manager = original_manager
            test_client.app.state.account_system = original_system
        assert listed["context_window"] == table["claude-sonnet-4.5"]["context_window"]



# --------------------------------------------------------------------------------------
# Auxiliary [models] roles (session_summary / image_description / web_search)
#
# Measured defect: with only `default` set, Grok Build resolved its auxiliary roles
# against its OWN built-in model (aux_model=grok-4.6) and fired one request per session
# at this gateway that upstream rejects with INVALID_MODEL_ID, silently degrading session
# titles to truncated user text. Every role must therefore name a visible model.
# --------------------------------------------------------------------------------------

from kiro.routes_openai import build_grok_build_config  # noqa: E402


def build(ids, meta=None, base_url="http://localhost:8000/v1", env_key="KIRO_GATEWAY_KEY"):
    meta = meta or {}
    return build_grok_build_config(
        model_ids=ids,
        metadata_for=lambda model_id: meta.get(model_id, {}),
        base_url=base_url,
        env_key=env_key,
    )


class TestAuxiliaryModelRoles:

    def test_session_summary_picks_cheapest_rate_multiplier(self):
        meta = {
            "expensive": {"rate_multiplier": 4.0},
            "cheap": {"rate_multiplier": 0.25},
            "mid": {"rate_multiplier": 1.0},
        }
        body = tomllib.loads(build(["expensive", "mid", "cheap"], meta))["models"]
        assert body["default"] == "expensive"  # catalog order still drives default
        assert body["session_summary"] == "cheap"

    def test_session_summary_tie_breaks_on_sorted_id(self):
        meta = {
            "b-model": {"rate_multiplier": 0.5},
            "a-model": {"rate_multiplier": 0.5},
            "z-model": {"rate_multiplier": 2.0},
        }
        body = tomllib.loads(build(["b-model", "z-model", "a-model"], meta))["models"]
        assert body["session_summary"] == "a-model"

    def test_session_summary_ignores_models_without_rate_metadata(self):
        meta = {"priced": {"rate_multiplier": 3.0}}
        body = tomllib.loads(build(["unpriced", "priced"], meta))["models"]
        assert body["session_summary"] == "priced"

    def test_session_summary_falls_back_to_default_without_any_rate_metadata(self):
        body = tomllib.loads(build(["claude-sonnet-4.5", "bare-model"]))["models"]
        assert body["session_summary"] == body["default"] == "claude-sonnet-4.5"

    def test_image_description_emitted_for_image_capable_model(self):
        meta = {
            "text-only": {"supported_input_types": ["TEXT"]},
            "vision": {"supported_input_types": ["TEXT", "IMAGE"]},
        }
        body = tomllib.loads(build(["text-only", "vision"], meta))["models"]
        assert body["image_description"] == "vision"

    def test_image_description_omitted_when_no_model_supports_images(self):
        meta = {"text-only": {"supported_input_types": ["TEXT"]}}
        text = build(["text-only"], meta)
        assert "image_description" not in text
        assert "image_description" not in tomllib.loads(text)["models"]

    def test_image_description_omitted_when_input_types_unknown(self):
        assert "image_description" not in tomllib.loads(build(["bare"]))["models"]

    def test_web_search_equals_default_and_backend_search_never_emitted(self):
        text = build(["first", "second"])
        body = tomllib.loads(text)["models"]
        assert body["web_search"] == body["default"] == "first"
        assert "supports_backend_search" not in text

    def test_every_role_names_a_visible_model(self):
        ids = ["alpha", "beta", "gamma"]
        meta = {
            "beta": {"rate_multiplier": 0.1, "supported_input_types": ["IMAGE"]},
            "gamma": {"rate_multiplier": 9.0},
        }
        body = tomllib.loads(build(ids, meta))["models"]
        assert set(body) == {
            "default", "session_summary", "prompt_suggestions",
            "image_description", "web_search",
        }
        for role, value in body.items():
            assert value in ids, role

    def test_empty_visible_set_omits_every_role(self):
        body = tomllib.loads(build([]))
        assert "models" not in body
        assert body["endpoints"]["models_base_url"].endswith("/v1")

    def test_document_still_parses_and_keeps_the_rest_of_the_shape(self):
        meta = {"claude-sonnet-4.5": {
            "display_name": "Claude Sonnet 4.5",
            "max_output_tokens": 64000,
            "max_input_tokens": 200000,
            "rate_multiplier": 1.0,
            "supported_input_types": ["TEXT", "IMAGE"],
        }}
        text = build(["claude-sonnet-4.5", "bare-model"], meta)
        body = tomllib.loads(text)
        assert set(body) == {"endpoints", "models", "model"}
        assert '[model."claude-sonnet-4.5"]' in text
        table = body["model"]["claude-sonnet-4.5"]
        assert table["api_backend"] == "chat_completions"
        assert table["context_window"] == 200000
        assert table["max_completion_tokens"] == 64000
        assert table["env_key"] == "KIRO_GATEWAY_KEY"
        assert "context_window" not in body["model"]["bare-model"]
        assert "stream_tool_calls" not in text

    def test_no_secret_in_the_document_with_roles_present(
        self, test_client, valid_proxy_api_key, cache
    ):
        response = call(test_client, valid_proxy_api_key, ["claude-sonnet-4.5"], cache)
        assert PROXY_API_KEY
        assert PROXY_API_KEY not in response.text
        roles = doc(response)["models"]
        assert roles["session_summary"] == "claude-sonnet-4.5"
        assert roles["web_search"] == "claude-sonnet-4.5"



# --------------------------------------------------------------------------------------
# prompt_suggestions: the FIFTH auxiliary role (GROK_PROMPT_SUGGESTIONS_MODEL). A TUI-only
# feature, so headless testing never exercised it. Throwaway work, so it uses the same
# cheapest-by-rate_multiplier rule as session_summary.
# --------------------------------------------------------------------------------------
class TestPromptSuggestionsRole:

    def test_emitted_and_equals_cheapest_by_rate(self):
        meta = {
            "expensive": {"rate_multiplier": 4.0},
            "cheap": {"rate_multiplier": 0.25},
            "mid": {"rate_multiplier": 1.0},
        }
        body = tomllib.loads(build(["expensive", "mid", "cheap"], meta))["models"]
        assert body["prompt_suggestions"] == "cheap"
        assert body["prompt_suggestions"] == body["session_summary"]

    def test_rate_ties_break_on_sorted_id(self):
        meta = {"zeta": {"rate_multiplier": 0.5}, "alpha": {"rate_multiplier": 0.5}}
        body = tomllib.loads(build(["zeta", "alpha"], meta))["models"]
        assert body["prompt_suggestions"] == "alpha"

    def test_degrades_to_default_without_rate_metadata(self):
        body = tomllib.loads(build(["first", "second"]))["models"]
        assert body["prompt_suggestions"] == body["default"] == "first"

    def test_omitted_when_visible_set_is_empty(self):
        text = build([])
        assert "prompt_suggestions" not in text
        assert "models" not in tomllib.loads(text)

    def test_never_names_a_model_outside_the_visible_set(self):
        ids = ["alpha", "beta"]
        meta = {"beta": {"rate_multiplier": 0.1}}
        body = tomllib.loads(build(ids, meta))["models"]
        assert body["prompt_suggestions"] in ids

    def test_document_still_parses_and_carries_no_secret(self):
        """The generator must never read the configured gateway key."""
        text = build(["alpha", "beta"], {"beta": {"rate_multiplier": 0.1}})
        parsed = tomllib.loads(text)  # explicit: still valid TOML with the new key
        assert parsed["models"]["prompt_suggestions"] == "beta"
        assert PROXY_API_KEY
        assert PROXY_API_KEY not in text
        assert "KIRO_GATEWAY_KEY" in text
