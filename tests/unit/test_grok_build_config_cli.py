"""Tests for the grok-build/ config helper CLI. Fully offline.

The helper is a standalone single-file script, so it is loaded by path rather than
imported as a package. No HTTP request leaves the process (the fetch is stubbed) and
every config lives in tmp_path — the real ~/.grok/config.toml is never touched.
"""

import importlib.util
import tomllib
import urllib.error
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "grok-build" / "grok_build_config.py"
_spec = importlib.util.spec_from_file_location("grok_build_config_helper", _SCRIPT)
gb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gb)


LIVE = '''# Kiro Gateway -> Grok Build config
[endpoints]
models_base_url = "http://localhost:8000/v1"

[models]
default = "claude-sonnet-4.5"
session_summary = "gpt-5.6-terra"
image_description = "claude-sonnet-4.5"
web_search = "claude-sonnet-4.5"

[model."claude-sonnet-4.5"]
model = "claude-sonnet-4.5"
name = "Claude Sonnet 4.5"
api_backend = "chat_completions"
context_window = 200000
max_completion_tokens = 64000
env_key = "KIRO_GATEWAY_KEY"

[model."gpt-5.6-terra"]
model = "gpt-5.6-terra"
name = "GPT 5.6 Terra"
api_backend = "chat_completions"
context_window = 300000
env_key = "KIRO_GATEWAY_KEY"
'''

# What the user's file actually contains today: an npm-managed installer marker plus a
# comment and an unrelated table that must survive untouched.
USER_FILE = '''# my hand-written notes
[cli]
installer = "npm"

# keep my servers
[mcp_servers.fs]
command = "npx"
'''

SECRET = "sk-live-abcdef1234567890"


@pytest.fixture
def stub_fetch(monkeypatch):
    calls = []

    def fake(**kwargs):
        calls.append(kwargs)
        return LIVE

    monkeypatch.setattr(gb, "fetch_document", fake)
    return calls


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ("KIRO_GATEWAY_URL", "KIRO_GATEWAY_KEY", "PROXY_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def healthy_config() -> str:
    return gb.merge_document(USER_FILE, LIVE)


# --------------------------------------------------------------------------- #
# merge safety: the user's text survives byte-for-byte
# --------------------------------------------------------------------------- #
def test_merge_preserves_user_text_byte_for_byte():
    merged = gb.merge_document(USER_FILE, LIVE)
    assert USER_FILE.rstrip("\n") in merged
    assert '[cli]\ninstaller = "npm"' in merged
    assert "# my hand-written notes" in merged
    assert "# keep my servers" in merged


def test_merge_adds_our_sections():
    parsed = tomllib.loads(gb.merge_document(USER_FILE, LIVE))
    assert parsed["cli"]["installer"] == "npm"
    assert parsed["mcp_servers"]["fs"]["command"] == "npx"
    assert parsed["endpoints"]["models_base_url"] == "http://localhost:8000/v1"
    assert set(parsed["model"]) == {"claude-sonnet-4.5", "gpt-5.6-terra"}


def test_merge_replaces_only_our_sections_and_is_idempotent():
    stale = gb.merge_document(USER_FILE, LIVE.replace("200000", "111111"))
    fixed = gb.merge_document(stale, LIVE)
    assert "111111" not in fixed
    assert gb.merge_document(fixed, LIVE) == fixed


def test_merge_into_empty_file_emits_just_the_document():
    assert gb.merge_document("", LIVE) == LIVE


def test_dotted_model_ids_are_not_nested():
    parsed = tomllib.loads(LIVE)
    assert "claude-sonnet-4.5" in parsed["model"]
    assert "claude-sonnet-4" not in parsed["model"]


def test_unrelated_model_providers_table_is_not_owned():
    text = '[model_providers.acme]\nbase_url = "http://x/v1"\n'
    assert gb.merge_document(text, LIVE).startswith(text)


# --------------------------------------------------------------------------- #
# setup: dry run vs --write
# --------------------------------------------------------------------------- #
def test_setup_dry_run_writes_nothing(tmp_path, stub_fetch, capsys):
    path = tmp_path / "config.toml"
    path.write_text(USER_FILE, encoding="utf-8")

    code = gb.main(["setup", "--config", str(path)])

    assert code == gb.EXIT_DRIFT
    assert path.read_text(encoding="utf-8") == USER_FILE
    assert "Dry run" in capsys.readouterr().out
    assert not list(tmp_path.glob("*.bak-*"))


def test_setup_write_creates_backup_and_applies(tmp_path, stub_fetch):
    path = tmp_path / "config.toml"
    path.write_text(USER_FILE, encoding="utf-8")

    code = gb.main(["setup", "--config", str(path), "--write"])

    assert code == gb.EXIT_OK
    backups = list(tmp_path.glob("config.toml.bak-*"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == USER_FILE
    assert '[cli]\ninstaller = "npm"' in path.read_text(encoding="utf-8")


def test_setup_twice_is_a_no_op(tmp_path, stub_fetch, capsys):
    path = tmp_path / "config.toml"
    path.write_text(USER_FILE, encoding="utf-8")
    gb.main(["setup", "--config", str(path), "--write"])
    capsys.readouterr()

    assert gb.main(["setup", "--config", str(path)]) == gb.EXIT_OK
    assert "No changes needed" in capsys.readouterr().out


def test_write_failure_leaves_original_intact(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text(USER_FILE, encoding="utf-8")

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(gb.os, "replace", boom)
    with pytest.raises(OSError):
        gb.write_text(str(path), "clobbered")

    assert path.read_text(encoding="utf-8") == USER_FILE
    assert not list(tmp_path.glob(".grok-config-*"))


def test_update_requires_setup_first(tmp_path, stub_fetch, capsys):
    path = tmp_path / "config.toml"
    path.write_text(USER_FILE, encoding="utf-8")

    assert gb.main(["update", "--config", str(path)]) == gb.EXIT_ERROR
    assert "run 'setup' first" in capsys.readouterr().err


def test_update_refreshes_stale_context_window(tmp_path, stub_fetch):
    path = tmp_path / "config.toml"
    path.write_text(gb.merge_document(USER_FILE, LIVE.replace("200000", "111111")),
                    encoding="utf-8")

    assert gb.main(["update", "--config", str(path), "--write"]) == gb.EXIT_OK
    parsed = tomllib.loads(path.read_text(encoding="utf-8"))
    assert parsed["model"]["claude-sonnet-4.5"]["context_window"] == 200000
    assert parsed["cli"]["installer"] == "npm"


# --------------------------------------------------------------------------- #
# doctor
# --------------------------------------------------------------------------- #
def test_doctor_healthy_exits_zero(tmp_path, stub_fetch, capsys):
    path = tmp_path / "config.toml"
    path.write_text(healthy_config(), encoding="utf-8")

    assert gb.main(["doctor", "--config", str(path)]) == gb.EXIT_OK
    assert "up to date" in capsys.readouterr().out


def _doctor(tmp_path, text):
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return gb.diagnose(text, LIVE, str(path))


def test_doctor_detects_missing_models_base_url(tmp_path):
    findings = _doctor(tmp_path, USER_FILE)
    assert any("models_base_url is missing" in f for f in findings)


def test_doctor_detects_base_url_not_our_gateway(tmp_path):
    text = healthy_config().replace(
        '"http://localhost:8000/v1"', '"https://api.x.ai/v1"'
    )
    findings = _doctor(tmp_path, text)
    assert any("this gateway reports" in f for f in findings)


def test_doctor_detects_base_url_without_v1(tmp_path):
    text = healthy_config().replace(
        '"http://localhost:8000/v1"', '"http://localhost:8000"'
    )
    findings = _doctor(tmp_path, text)
    assert any("does not end in /v1" in f for f in findings)


def test_doctor_detects_model_not_available_upstream(tmp_path):
    text = healthy_config() + '\n[model.ghost]\nmodel = "ghost"\nenv_key = "KIRO_GATEWAY_KEY"\n'
    findings = _doctor(tmp_path, text)
    assert any("'ghost' is configured but not available upstream" in f for f in findings)


def test_doctor_detects_missing_model(tmp_path):
    text = healthy_config().replace('[model."gpt-5.6-terra"]', "[unrelated_leftover]")
    findings = _doctor(tmp_path, text)
    assert any("'gpt-5.6-terra' is available upstream but missing" in f for f in findings)


def test_doctor_detects_stale_context_window(tmp_path):
    text = healthy_config().replace("context_window = 200000", "context_window = 111111")
    findings = _doctor(tmp_path, text)
    assert any("context_window is 111111" in f for f in findings)


def test_doctor_detects_wrong_api_backend(tmp_path):
    text = healthy_config().replace(
        'api_backend = "chat_completions"', 'api_backend = "responses"', 1
    )
    findings = _doctor(tmp_path, text)
    assert any("api_backend is 'responses'" in f for f in findings)


def test_doctor_detects_literal_secret_in_env_key_without_printing_it(tmp_path):
    text = healthy_config().replace(
        'env_key = "KIRO_GATEWAY_KEY"', f'env_key = "{SECRET}"', 1
    )
    findings = _doctor(tmp_path, text)
    assert any("looks like a literal secret" in f for f in findings)
    assert all(SECRET not in f for f in findings)


def test_doctor_exit_code_is_one_on_drift(tmp_path, stub_fetch):
    path = tmp_path / "config.toml"
    path.write_text(USER_FILE, encoding="utf-8")
    assert gb.main(["doctor", "--config", str(path)]) == gb.EXIT_DRIFT


def test_doctor_detects_missing_session_summary(tmp_path):
    text = healthy_config().replace('session_summary = "gpt-5.6-terra"\n', "")
    findings = _doctor(tmp_path, text)
    assert any("[models] session_summary is not set" in f for f in findings)
    assert any("built-in xAI model" in f for f in findings)


def test_doctor_detects_missing_web_search(tmp_path):
    text = healthy_config().replace('web_search = "claude-sonnet-4.5"\n', "")
    findings = _doctor(tmp_path, text)
    assert any("[models] web_search is not set" in f for f in findings)


def test_doctor_detects_session_summary_naming_unavailable_model(tmp_path):
    text = healthy_config().replace(
        'session_summary = "gpt-5.6-terra"', 'session_summary = "grok-4.6"'
    )
    findings = _doctor(tmp_path, text)
    assert any(
        "session_summary names 'grok-4.6', which is not available upstream" in f
        for f in findings
    )


def test_doctor_detects_image_description_naming_unavailable_model(tmp_path):
    text = healthy_config().replace(
        'image_description = "claude-sonnet-4.5"', 'image_description = "grok-4.6"'
    )
    findings = _doctor(tmp_path, text)
    assert any("image_description names 'grok-4.6'" in f for f in findings)


def test_doctor_tolerates_absent_image_description(tmp_path):
    """The gateway omits it when no visible model takes images; that is not drift."""
    text = healthy_config().replace('image_description = "claude-sonnet-4.5"\n', "")
    findings = _doctor(tmp_path, text)
    assert not any("image_description" in f for f in findings)


def test_setup_installs_aux_roles_without_touching_user_text(tmp_path, stub_fetch):
    path = tmp_path / "config.toml"
    path.write_text(USER_FILE, encoding="utf-8")

    assert gb.main(["setup", "--config", str(path), "--write"]) == gb.EXIT_OK
    text = path.read_text(encoding="utf-8")
    parsed = tomllib.loads(text)
    assert parsed["models"]["session_summary"] == "gpt-5.6-terra"
    assert parsed["models"]["web_search"] == "claude-sonnet-4.5"
    assert parsed["cli"]["installer"] == "npm"
    assert USER_FILE.rstrip("\n") in text


def test_update_repairs_an_aux_role_pointing_at_xai(tmp_path, stub_fetch):
    path = tmp_path / "config.toml"
    path.write_text(
        healthy_config().replace(
            'session_summary = "gpt-5.6-terra"', 'session_summary = "grok-4.6"'
        ),
        encoding="utf-8",
    )

    assert gb.main(["update", "--config", str(path), "--write"]) == gb.EXIT_OK
    text = path.read_text(encoding="utf-8")
    assert "grok-4.6" not in text
    assert tomllib.loads(text)["models"]["session_summary"] == "gpt-5.6-terra"
    assert '[cli]\ninstaller = "npm"' in text


def test_diff_redacts_secret_values(tmp_path):
    before = healthy_config().replace(
        'env_key = "KIRO_GATEWAY_KEY"', f'env_key = "{SECRET}"', 1
    )
    diff = gb.render_diff(before, healthy_config(), str(tmp_path / "config.toml"))
    assert SECRET not in diff
    assert gb.REDACTED in diff


# --------------------------------------------------------------------------- #
# clean failures
# --------------------------------------------------------------------------- #
def test_gateway_down_is_clean_error(tmp_path, monkeypatch, capsys):
    path = tmp_path / "config.toml"
    path.write_text(USER_FILE, encoding="utf-8")

    def boom(request, timeout=None):
        raise urllib.error.URLError("Connection refused")

    monkeypatch.setattr(gb.urllib.request, "urlopen", boom)

    assert gb.main(["setup", "--config", str(path)]) == gb.EXIT_ERROR
    err = capsys.readouterr().err
    assert "Could not reach the gateway" in err
    assert "Traceback" not in err


def test_401_is_clean_error(tmp_path, monkeypatch, capsys):
    path = tmp_path / "config.toml"
    path.write_text(USER_FILE, encoding="utf-8")

    def boom(request, timeout=None):
        raise urllib.error.HTTPError("http://x", 401, "Unauthorized", {}, None)

    monkeypatch.setattr(gb.urllib.request, "urlopen", boom)

    assert gb.main(["setup", "--config", str(path)]) == gb.EXIT_ERROR
    assert "rejected the API key" in capsys.readouterr().err


def test_malformed_gateway_toml_is_clean_error(tmp_path, monkeypatch, capsys):
    path = tmp_path / "config.toml"
    path.write_text(USER_FILE, encoding="utf-8")
    monkeypatch.setattr(gb, "fetch_document",
                        lambda **kw: (_ for _ in ()).throw(
                            gb.HelperError("gateway document is not valid TOML: boom")))

    assert gb.main(["setup", "--config", str(path)]) == gb.EXIT_ERROR
    assert "not valid TOML" in capsys.readouterr().err


def test_unparseable_local_config_is_clean_error(tmp_path, stub_fetch, capsys):
    path = tmp_path / "config.toml"
    path.write_text("[cli\ninstaller = npm", encoding="utf-8")

    assert gb.main(["doctor", "--config", str(path)]) == gb.EXIT_ERROR
    err = capsys.readouterr().err
    assert "is not valid TOML" in err
    assert "Traceback" not in err
    assert path.read_text(encoding="utf-8") == "[cli\ninstaller = npm"


def test_default_config_path_is_dot_grok():
    assert gb.config_path(None).endswith("/.grok/config.toml")
