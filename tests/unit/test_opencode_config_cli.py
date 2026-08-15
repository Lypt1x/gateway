"""Tests for the opencode/ config helper CLI. Fully offline.

The helper is a standalone single-file script, so it is loaded by path rather
than imported as a package.
"""

import importlib.util
import json
import os
import urllib.error
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "opencode" / "opencode_config.py"
_spec = importlib.util.spec_from_file_location("opencode_config_helper", _SCRIPT)
oc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oc)


LIVE = {
    "$schema": "https://opencode.ai/config.json",
    "provider": {
        "kiro": {
            "npm": "@ai-sdk/openai-compatible",
            "name": "Kiro Gateway",
            "options": {
                "baseURL": "http://localhost:8000/v1",
                "apiKey": "{env:KIRO_GATEWAY_KEY}",
            },
            "models": {
                "claude-sonnet-4.5": {
                    "name": "Claude Sonnet 4.5",
                    "limit": {"context": 200000, "output": 64000},
                },
                "gpt-5.6-terra": {
                    "name": "GPT 5.6 Terra",
                    "limit": {"context": 300000, "output": 80000},
                },
            },
        }
    },
}

SECRET = "sk-live-abcdef1234567890"


@pytest.fixture
def stub_fetch(monkeypatch):
    """Replace the HTTP fetch; no request ever leaves the process."""
    calls = []

    def fake(**kwargs):
        calls.append(kwargs)
        return json.loads(json.dumps(LIVE))

    monkeypatch.setattr(oc, "fetch_document", fake)
    return calls


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ("KIRO_GATEWAY_URL", "KIRO_GATEWAY_KEY", "PROXY_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# merge safety
# --------------------------------------------------------------------------- #
def test_merge_preserves_unrelated_keys_and_other_providers():
    existing = {
        "$schema": "https://opencode.ai/config.json",
        "agent": {"build": {"model": "kiro/claude-sonnet-4.5"}},
        "mcp": {"fs": {"type": "local"}},
        "permission": {"edit": "ask"},
        "provider": {"openai": {"npm": "@ai-sdk/openai", "models": {"gpt-4o": {}}}},
    }
    merged = oc.merge_provider(existing, LIVE, "kiro")

    assert merged["agent"] == existing["agent"]
    assert merged["mcp"] == existing["mcp"]
    assert merged["permission"] == existing["permission"]
    assert merged["provider"]["openai"] == existing["provider"]["openai"]
    assert set(merged["provider"]["kiro"]["models"]) == {
        "claude-sonnet-4.5", "gpt-5.6-terra"
    }
    # input untouched
    assert "kiro" not in existing["provider"]


def test_merge_preserves_key_order_and_user_extras():
    existing = {
        "provider": {
            "kiro": {
                "npm": "wrong",
                "options": {"baseURL": "http://old/v1", "extraOption": 1},
                "models": {"claude-sonnet-4.5": {"customFlag": True}},
            }
        },
        "keybinds": {"leader": "ctrl+x"},
    }
    merged = oc.merge_provider(existing, LIVE, "kiro")
    entry = merged["provider"]["kiro"]

    assert list(merged) == ["provider", "keybinds", "$schema"]
    assert entry["npm"] == "@ai-sdk/openai-compatible"
    assert entry["options"]["extraOption"] == 1
    assert entry["options"]["baseURL"] == "http://localhost:8000/v1"
    assert entry["models"]["claude-sonnet-4.5"]["customFlag"] is True
    assert entry["models"]["claude-sonnet-4.5"]["limit"]["context"] == 200000


def test_update_refreshes_limits_only():
    existing = {
        "provider": {
            "kiro": {
                "npm": "@ai-sdk/openai-compatible",
                "options": {"baseURL": "http://custom:9/v1", "apiKey": "{env:X}"},
                "models": {"claude-sonnet-4.5": {"limit": {"context": 1, "output": 2}}},
            }
        }
    }
    updated = oc.refresh_models(existing, LIVE, "kiro")
    entry = updated["provider"]["kiro"]
    assert entry["options"] == {"baseURL": "http://custom:9/v1", "apiKey": "{env:X}"}
    assert entry["models"]["claude-sonnet-4.5"]["limit"] == {
        "context": 200000, "output": 64000
    }
    assert "gpt-5.6-terra" in entry["models"]


def test_update_without_setup_is_an_error(tmp_path, stub_fetch, capsys):
    cfg = write_json(tmp_path / "opencode.json", {"agent": {}})
    assert oc.main(["update", "--config", str(cfg)]) == oc.EXIT_ERROR
    assert "run 'setup' first" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# dry run / write safety
# --------------------------------------------------------------------------- #
def test_dry_run_writes_nothing(tmp_path, stub_fetch, capsys):
    cfg = write_json(tmp_path / "opencode.json", {"agent": {"build": {}}})
    before = cfg.read_bytes()

    assert oc.main(["setup", "--config", str(cfg)]) == oc.EXIT_DRIFT
    assert cfg.read_bytes() == before
    out = capsys.readouterr().out
    assert "Dry run" in out and "+" in out
    assert list(tmp_path.iterdir()) == [cfg]


def test_write_creates_backup_and_applies(tmp_path, stub_fetch):
    cfg = write_json(tmp_path / "opencode.json", {"agent": {"build": {"x": 1}}})

    assert oc.main(["setup", "--config", str(cfg), "--write"]) == oc.EXIT_OK
    data = json.loads(cfg.read_text())
    assert data["agent"] == {"build": {"x": 1}}
    assert "kiro" in data["provider"]

    backups = [p for p in tmp_path.iterdir() if ".bak-" in p.name]
    assert len(backups) == 1
    assert json.loads(backups[0].read_text()) == {"agent": {"build": {"x": 1}}}
    # no temp files left behind
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".opencode-")]


def test_write_is_atomic_via_os_replace(tmp_path, monkeypatch):
    cfg = write_json(tmp_path / "opencode.json", {"agent": {}})
    seen = {}
    real_replace = os.replace

    def spy(src, dst):
        seen["src"] = src
        seen["dst"] = dst
        assert os.path.dirname(src) == os.path.dirname(dst)
        real_replace(src, dst)

    monkeypatch.setattr(oc.os, "replace", spy)
    oc.write_config(str(cfg), {"provider": {}})
    assert seen["dst"] == str(cfg)


def test_write_failure_leaves_original_intact(tmp_path, monkeypatch):
    cfg = write_json(tmp_path / "opencode.json", {"agent": {"keep": True}})

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(oc.os, "replace", boom)
    with pytest.raises(OSError):
        oc.write_config(str(cfg), {"provider": {"kiro": {}}})

    assert json.loads(cfg.read_text()) == {"agent": {"keep": True}}
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".opencode-")]


def test_fresh_install_with_no_existing_config(tmp_path, stub_fetch):
    cfg = tmp_path / "nested" / "opencode.json"
    assert oc.main(["setup", "--config", str(cfg), "--write"]) == oc.EXIT_OK
    data = json.loads(cfg.read_text())
    assert data["$schema"] == "https://opencode.ai/config.json"
    assert len(data["provider"]["kiro"]["models"]) == 2
    assert not [p for p in cfg.parent.iterdir() if ".bak-" in p.name]


def test_second_setup_is_a_noop(tmp_path, stub_fetch, capsys):
    cfg = tmp_path / "opencode.json"
    oc.main(["setup", "--config", str(cfg), "--write"])
    assert oc.main(["setup", "--config", str(cfg)]) == oc.EXIT_OK
    assert "No changes needed" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# doctor
# --------------------------------------------------------------------------- #
def _config(**provider_overrides):
    entry = {
        "npm": "@ai-sdk/openai-compatible",
        "options": {
            "baseURL": "http://localhost:8000/v1",
            "apiKey": "{env:KIRO_GATEWAY_KEY}",
        },
        "models": json.loads(json.dumps(LIVE["provider"]["kiro"]["models"])),
    }
    entry.update(provider_overrides)
    return {"provider": {"kiro": entry}}


def test_doctor_clean_config_is_exit_zero(tmp_path, stub_fetch, capsys):
    cfg = write_json(tmp_path / "opencode.json", _config())
    assert oc.main(["doctor", "--config", str(cfg)]) == oc.EXIT_OK
    assert "up to date" in capsys.readouterr().out


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda c: c["provider"]["kiro"]["models"].update({"ghost-model": {}}),
         "not available upstream"),
        (lambda c: c["provider"]["kiro"]["models"].pop("gpt-5.6-terra"),
         "available upstream but missing"),
        (lambda c: c["provider"]["kiro"]["models"]["claude-sonnet-4.5"]["limit"]
         .update({"context": 8192}),
         "limit.context is 8192"),
        (lambda c: c["provider"]["kiro"].update({"npm": "@ai-sdk/openai"}),
         "expected '@ai-sdk/openai-compatible'"),
        (lambda c: c["provider"]["kiro"]["options"].update(
            {"baseURL": "http://localhost:8000"}),
         "does not end in /v1"),
        (lambda c: c["provider"]["kiro"]["options"].update({"apiKey": SECRET}),
         "looks like a literal secret"),
    ],
    ids=["unavailable", "missing", "stale-limit", "npm", "base-url", "literal-secret"],
)
def test_doctor_detects_each_drift_class(tmp_path, stub_fetch, capsys, mutate, expected):
    config = _config()
    mutate(config)
    cfg = write_json(tmp_path / "opencode.json", config)

    assert oc.main(["doctor", "--config", str(cfg)]) == oc.EXIT_DRIFT
    captured = capsys.readouterr()
    assert expected in captured.out
    assert SECRET not in captured.out and SECRET not in captured.err


def test_doctor_reports_unconfigured_provider(tmp_path, stub_fetch, capsys):
    cfg = write_json(tmp_path / "opencode.json", {"agent": {}})
    assert oc.main(["doctor", "--config", str(cfg)]) == oc.EXIT_DRIFT
    assert "not configured" in capsys.readouterr().out


def test_doctor_changes_nothing_on_disk(tmp_path, stub_fetch):
    config = _config()
    config["provider"]["kiro"]["models"].pop("gpt-5.6-terra")
    cfg = write_json(tmp_path / "opencode.json", config)
    before = cfg.read_bytes()
    oc.main(["doctor", "--config", str(cfg)])
    assert cfg.read_bytes() == before
    assert list(tmp_path.iterdir()) == [cfg]


# --------------------------------------------------------------------------- #
# secret hygiene
# --------------------------------------------------------------------------- #
def test_secret_never_appears_in_setup_diff_and_is_not_rewritten(
    tmp_path, stub_fetch, capsys
):
    config = _config()
    config["provider"]["kiro"]["options"]["apiKey"] = SECRET
    config["provider"]["kiro"]["models"].pop("gpt-5.6-terra")
    cfg = write_json(tmp_path / "opencode.json", config)

    assert oc.main(["setup", "--config", str(cfg), "--write"]) == oc.EXIT_OK
    captured = capsys.readouterr()
    assert SECRET not in captured.out and SECRET not in captured.err
    assert oc.REDACTED in json.dumps(oc.redact(json.loads(cfg.read_text())))
    # the user's own value is left exactly as it was; never replaced, never printed
    assert json.loads(cfg.read_text())["provider"]["kiro"]["options"]["apiKey"] == SECRET
    assert not oc.looks_like_secret("{env:KIRO_GATEWAY_KEY}")
    assert not oc.looks_like_secret("{file:~/.kiro/key}")


# --------------------------------------------------------------------------- #
# error paths
# --------------------------------------------------------------------------- #
def _stub_urlopen(monkeypatch, raiser):
    monkeypatch.setattr(oc.urllib.request, "urlopen", raiser)


def test_gateway_down_gives_clean_message(tmp_path, monkeypatch, capsys):
    def raiser(*_a, **_k):
        raise urllib.error.URLError("Connection refused")

    _stub_urlopen(monkeypatch, raiser)
    code = oc.main(["doctor", "--config", str(tmp_path / "c.json")])
    captured = capsys.readouterr()
    assert code == oc.EXIT_ERROR
    assert "Could not reach the gateway" in captured.err
    assert "Traceback" not in captured.err


def test_401_gives_clean_message(tmp_path, monkeypatch, capsys):
    def raiser(*_a, **_k):
        raise urllib.error.HTTPError("u", 401, "Unauthorized", {}, None)

    _stub_urlopen(monkeypatch, raiser)
    code = oc.main(["doctor", "--config", str(tmp_path / "c.json"),
                    "--api-key", SECRET])
    captured = capsys.readouterr()
    assert code == oc.EXIT_ERROR
    assert "rejected the API key (401)" in captured.err
    assert SECRET not in captured.err and SECRET not in captured.out


def test_malformed_json_response_gives_clean_message(tmp_path, monkeypatch, capsys):
    class FakeResponse:
        def read(self):
            return b"<html>login</html>"

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    _stub_urlopen(monkeypatch, lambda *_a, **_k: FakeResponse())
    code = oc.main(["doctor", "--config", str(tmp_path / "c.json")])
    assert code == oc.EXIT_ERROR
    assert "not valid JSON" in capsys.readouterr().err


def test_malformed_local_config_gives_clean_message(tmp_path, stub_fetch, capsys):
    cfg = tmp_path / "opencode.json"
    cfg.write_text("{ broken", encoding="utf-8")
    assert oc.main(["doctor", "--config", str(cfg)]) == oc.EXIT_ERROR
    assert "not valid JSON" in capsys.readouterr().err


def test_unknown_provider_in_document_is_an_error(tmp_path, stub_fetch, capsys):
    cfg = write_json(tmp_path / "opencode.json", _config())
    assert oc.main(["doctor", "--config", str(cfg), "--provider", "nope"]) == oc.EXIT_ERROR
    assert "no provider 'nope'" in capsys.readouterr().err


def test_flags_and_env_reach_the_fetch(tmp_path, stub_fetch, monkeypatch):
    cfg = write_json(tmp_path / "opencode.json", _config())
    oc.main(["doctor", "--config", str(cfg), "--url", "http://gw:9000/",
             "--api-key", SECRET, "--base-url", "http://gw:9000/v1"])
    assert stub_fetch[0]["url"] == "http://gw:9000/"
    assert stub_fetch[0]["api_key"] == SECRET
    assert stub_fetch[0]["base_url"] == "http://gw:9000/v1"

    monkeypatch.setenv("KIRO_GATEWAY_KEY", SECRET)
    monkeypatch.setenv("KIRO_GATEWAY_URL", "http://env-gw:1/")
    oc.main(["doctor", "--config", str(cfg)])
    assert stub_fetch[1]["api_key"] == SECRET
    assert stub_fetch[1]["url"] == "http://env-gw:1/"



# --------------------------------------------------------------------------- #
# --prune
#
# Stale entries are kept by default: an account may legitimately reach a model that
# is absent from this catalog, and deleting someone's config silently is worse than
# leaving a stale line. --prune is what lets `doctor` reach a clean state.
# --------------------------------------------------------------------------- #

def _config_with_stale_model():
    return {
        "provider": {
            "kiro": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Kiro Gateway",
                "options": {
                    "baseURL": "http://localhost:8000/v1",
                    "apiKey": "{env:KIRO_GATEWAY_KEY}",
                },
                "models": {
                    "claude-sonnet-4.5": {"name": "Claude Sonnet 4.5"},
                    "ghost-model-9000": {"name": "Ghost"},
                },
            }
        },
        "agent": {"build": {"model": "kiro/claude-sonnet-4.5"}},
    }


def test_stale_model_is_kept_without_prune():
    merged = oc.merge_provider(_config_with_stale_model(), LIVE, "kiro")

    assert "ghost-model-9000" in merged["provider"]["kiro"]["models"]


def test_prune_removes_models_upstream_does_not_serve():
    merged = oc.merge_provider(_config_with_stale_model(), LIVE, "kiro", prune=True)
    models = merged["provider"]["kiro"]["models"]

    assert "ghost-model-9000" not in models
    assert set(models) == set(LIVE["provider"]["kiro"]["models"])


def test_prune_preserves_unrelated_config():
    merged = oc.merge_provider(_config_with_stale_model(), LIVE, "kiro", prune=True)

    assert merged["agent"] == {"build": {"model": "kiro/claude-sonnet-4.5"}}


def test_update_prunes_when_asked():
    updated = oc.refresh_models(_config_with_stale_model(), LIVE, "kiro", prune=True)

    assert "ghost-model-9000" not in updated["provider"]["kiro"]["models"]


def test_prune_makes_doctor_clean(tmp_path, stub_fetch, capsys):
    """setup --write --prune should resolve every drift class doctor can auto-fix."""
    path = write_json(tmp_path / "opencode.json", _config_with_stale_model())

    assert oc.main(["setup", "--config", str(path), "--write", "--prune"]) == oc.EXIT_OK
    capsys.readouterr()

    assert oc.main(["doctor", "--config", str(path)]) == oc.EXIT_OK
