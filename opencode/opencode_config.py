#!/usr/bin/env python3
"""OpenCode config helper for the Kiro gateway.

Thin client: it speaks HTTP to the gateway's own
``GET /integrations/opencode.json`` endpoint and edits a local JSON file.
It contains no auth, token-refresh, region, header or catalog logic — the
gateway is the single source of truth for all of that.

Subcommands
-----------
setup    fetch the live document and merge ``provider.<id>`` into your config
doctor   diff your config against the live document and report drift only
update   refresh the managed provider's models/limits in place

Exit codes
----------
0  healthy / nothing to change
1  drift found or changes pending (dry run)
2  usage error, connection failure, bad key, malformed response
"""

from __future__ import annotations

import argparse
import copy
import datetime as _dt
import difflib
import json
import os
import re
import shutil
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request

EXIT_OK = 0
EXIT_DRIFT = 1
EXIT_ERROR = 2

DEFAULT_URL = "http://localhost:8000"
DEFAULT_PROVIDER = "kiro"
DEFAULT_CONFIG = "~/.config/opencode/opencode.json"
EXPECTED_NPM = "@ai-sdk/openai-compatible"
REDACTED = "<redacted>"

_PLACEHOLDER_RE = re.compile(r"^\{(env|file):[^}]+\}$")


class HelperError(Exception):
    """Any expected, user-facing failure. Never surfaces as a traceback."""


# --------------------------------------------------------------------------- #
# fetching
# --------------------------------------------------------------------------- #
def fetch_document(
    url: str,
    api_key: str | None,
    provider: str = DEFAULT_PROVIDER,
    base_url: str | None = None,
    api_key_placeholder: str | None = None,
    reasoning: bool = False,
    timeout: float = 30.0,
) -> dict:
    """GET the gateway's OpenCode document. Raises HelperError on any failure."""
    params: dict[str, str] = {"provider": provider}
    if base_url:
        params["base_url"] = base_url
    if api_key_placeholder:
        params["api_key"] = api_key_placeholder
    if reasoning:
        params["reasoning"] = "true"

    endpoint = url.rstrip("/") + "/integrations/opencode.json"
    full = endpoint + "?" + urllib.parse.urlencode(params)

    request = urllib.request.Request(full, method="GET")
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
        request.add_header("x-api-key", api_key)

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise HelperError(
                f"Gateway rejected the API key ({exc.code}). Pass --api-key or set "
                "KIRO_GATEWAY_KEY to the gateway's proxy API key."
            ) from None
        if exc.code == 404:
            raise HelperError(
                f"{endpoint} returned 404 — this gateway is too old to expose the "
                "OpenCode integration endpoint."
            ) from None
        raise HelperError(f"Gateway returned HTTP {exc.code} for {endpoint}.") from None
    except urllib.error.URLError as exc:
        raise HelperError(
            f"Could not reach the gateway at {url} ({exc.reason}). Is it running, "
            "and is --url correct?"
        ) from None
    except TimeoutError:
        raise HelperError(f"Timed out talking to the gateway at {url}.") from None

    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise HelperError(
            f"Gateway response from {endpoint} was not valid JSON. Check that --url "
            "points at the gateway and not at a proxy or login page."
        ) from None

    if not isinstance(document, dict) or not isinstance(document.get("provider"), dict):
        raise HelperError(
            "Gateway response has no 'provider' object — unexpected document shape."
        )
    return document


# --------------------------------------------------------------------------- #
# local config I/O
# --------------------------------------------------------------------------- #
def config_path(raw: str | None) -> str:
    return os.path.abspath(os.path.expanduser(raw or DEFAULT_CONFIG))


def load_config(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise HelperError(f"{path} is not valid JSON (line {exc.lineno}).") from None
    except OSError as exc:
        raise HelperError(f"Could not read {path}: {exc.strerror}.") from None
    if not isinstance(data, dict):
        raise HelperError(f"{path} does not contain a JSON object.")
    return data


def write_config(path: str, document: dict) -> str | None:
    """Back up any existing file, then write atomically. Returns backup path."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)

    backup = None
    if os.path.exists(path):
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = f"{path}.bak-{stamp}"
        shutil.copy2(path, backup)

    text = json.dumps(document, indent=2, ensure_ascii=False) + "\n"
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=directory, prefix=".opencode-", suffix=".tmp",
        delete=False,
    )
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, path)
    except BaseException:
        try:
            os.unlink(handle.name)
        except OSError:
            pass
        raise
    return backup


# --------------------------------------------------------------------------- #
# secret hygiene
# --------------------------------------------------------------------------- #
def is_placeholder(value: object) -> bool:
    return isinstance(value, str) and bool(_PLACEHOLDER_RE.match(value.strip()))


def looks_like_secret(value: object) -> bool:
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    if not stripped or is_placeholder(stripped):
        return False
    return True


def redact(document: dict) -> dict:
    """Copy with every provider ``options.apiKey`` literal replaced."""
    safe = copy.deepcopy(document)
    providers = safe.get("provider")
    if isinstance(providers, dict):
        for entry in providers.values():
            if not isinstance(entry, dict):
                continue
            options = entry.get("options")
            if isinstance(options, dict) and looks_like_secret(options.get("apiKey")):
                options["apiKey"] = REDACTED
    return safe


def render_diff(before: dict, after: dict, path: str) -> str:
    old = json.dumps(redact(before), indent=2, ensure_ascii=False).splitlines()
    new = json.dumps(redact(after), indent=2, ensure_ascii=False).splitlines()
    return "\n".join(
        difflib.unified_diff(old, new, fromfile=f"{path} (current)",
                             tofile=f"{path} (proposed)", lineterm="")
    )


# --------------------------------------------------------------------------- #
# merging
# --------------------------------------------------------------------------- #
def _live_provider(document: dict, provider: str) -> dict:
    providers = document.get("provider", {})
    entry = providers.get(provider)
    if not isinstance(entry, dict):
        available = ", ".join(sorted(providers)) or "none"
        raise HelperError(
            f"Gateway document has no provider '{provider}' (found: {available}). "
            "Use --provider to match."
        )
    return entry


def merge_provider(
    existing: dict, document: dict, provider: str, prune: bool = False
) -> dict:
    """Merge only ``provider.<provider>``; everything else is untouched."""
    live = _live_provider(document, provider)
    merged = copy.deepcopy(existing)

    if "$schema" not in merged and "$schema" in document:
        merged["$schema"] = document["$schema"]

    providers = merged.setdefault("provider", {})
    if not isinstance(providers, dict):
        raise HelperError("Local config's 'provider' key is not an object.")

    entry = providers.get(provider)
    if not isinstance(entry, dict):
        entry = {}
    entry = copy.deepcopy(entry)

    for key, value in live.items():
        if key in ("options", "models"):
            continue
        entry[key] = value

    options = entry.get("options")
    options = copy.deepcopy(options) if isinstance(options, dict) else {}
    for key, value in live.get("options", {}).items():
        if key == "apiKey" and looks_like_secret(options.get("apiKey")):
            # Never rewrite (or echo) a user-managed literal; doctor warns instead.
            continue
        options[key] = value
    entry["options"] = options

    entry["models"] = _merge_models(entry.get("models"), live.get("models", {}), prune)
    providers[provider] = entry
    return merged


def _merge_models(existing_models: object, live_models: dict, prune: bool = False) -> dict:
    """
    Merge live model definitions over the user's, preserving any extra keys they set.

    Entries the user configured that upstream does not serve are KEPT by default: an
    account may legitimately reach a model that is absent from this catalog, and silently
    deleting someone's configuration is worse than leaving a stale line. `prune=True`
    removes them, which is what makes `doctor` able to reach a clean state.
    """
    models = copy.deepcopy(existing_models) if isinstance(existing_models, dict) else {}
    if prune:
        models = {
            model_id: entry
            for model_id, entry in models.items()
            if model_id in live_models
        }
    for model_id, live_model in live_models.items():
        current = models.get(model_id)
        if not isinstance(current, dict):
            models[model_id] = copy.deepcopy(live_model)
            continue
        merged = copy.deepcopy(current)
        for key, value in live_model.items():
            merged[key] = copy.deepcopy(value)
        models[model_id] = merged
    return models


def refresh_models(
    existing: dict, document: dict, provider: str, prune: bool = False
) -> dict:
    """`update`: only the models map and limits change."""
    live = _live_provider(document, provider)
    updated = copy.deepcopy(existing)
    providers = updated.get("provider")
    if not isinstance(providers, dict) or not isinstance(providers.get(provider), dict):
        raise HelperError(
            f"Provider '{provider}' is not configured yet — run 'setup' first."
        )
    entry = providers[provider]
    entry["models"] = _merge_models(entry.get("models"), live.get("models", {}), prune)
    return updated


# --------------------------------------------------------------------------- #
# doctor
# --------------------------------------------------------------------------- #
def diagnose(existing: dict, document: dict, provider: str) -> list[str]:
    live = _live_provider(document, provider)
    findings: list[str] = []

    providers = existing.get("provider")
    entry = providers.get(provider) if isinstance(providers, dict) else None
    if not isinstance(entry, dict):
        return [f"provider.{provider} is not configured — run 'setup'."]

    npm = entry.get("npm")
    if npm != EXPECTED_NPM:
        shown = "missing" if npm is None else f"'{npm}'"
        findings.append(
            f"provider.{provider}.npm is {shown}; expected '{EXPECTED_NPM}'."
        )

    options = entry.get("options") if isinstance(entry.get("options"), dict) else {}
    base_url = options.get("baseURL")
    if not isinstance(base_url, str) or not base_url:
        findings.append(f"provider.{provider}.options.baseURL is missing.")
    elif not base_url.rstrip("/").endswith("/v1"):
        findings.append(
            f"provider.{provider}.options.baseURL ('{base_url}') does not end in /v1."
        )

    api_key = options.get("apiKey")
    if api_key is None:
        findings.append(f"provider.{provider}.options.apiKey is missing.")
    elif looks_like_secret(api_key):
        findings.append(
            f"provider.{provider}.options.apiKey looks like a literal secret (value "
            "not shown). Replace it with a {env:NAME} or {file:PATH} reference."
        )

    live_models = live.get("models", {})
    user_models = entry.get("models") if isinstance(entry.get("models"), dict) else {}

    for model_id in user_models:
        if model_id not in live_models:
            findings.append(
                f"model '{model_id}' is configured but not available upstream — "
                "calls to it would fail."
            )
    for model_id in live_models:
        if model_id not in user_models:
            findings.append(f"model '{model_id}' is available upstream but missing.")

    for model_id, live_model in live_models.items():
        current = user_models.get(model_id)
        if not isinstance(current, dict):
            continue
        live_limit = live_model.get("limit") or {}
        current_limit = current.get("limit") or {}
        for field in ("context", "output"):
            if field not in live_limit:
                continue
            if current_limit.get(field) != live_limit[field]:
                findings.append(
                    f"model '{model_id}' limit.{field} is "
                    f"{current_limit.get(field, 'missing')}; live value is "
                    f"{live_limit[field]}."
                )
    return findings


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="opencode_config",
        description="Merge, check and refresh the Kiro gateway provider block in an "
                    "OpenCode config file.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--config", help=f"OpenCode config file (default {DEFAULT_CONFIG})")
        sp.add_argument("--url", help=f"Gateway base URL (default {DEFAULT_URL}, "
                                      "or $KIRO_GATEWAY_URL)")
        sp.add_argument("--api-key", help="Gateway proxy API key (or $KIRO_GATEWAY_KEY)")
        sp.add_argument("--provider", default=DEFAULT_PROVIDER,
                        help=f"Provider id (default {DEFAULT_PROVIDER})")
        sp.add_argument("--base-url", help="Override options.baseURL in the document")
        sp.add_argument("--timeout", type=float, default=30.0)

    for name, help_text in (
        ("setup", "merge the gateway provider block into your config"),
        ("doctor", "report drift between your config and the live catalog"),
        ("update", "refresh models and limits of the managed provider"),
    ):
        sp = sub.add_parser(name, help=help_text)
        common(sp)
        if name in ("setup", "update"):
            sp.add_argument("--write", action="store_true",
                            help="apply the change (default is a dry run)")
            sp.add_argument("--prune", action="store_true",
                            help="also remove configured models that upstream does not "
                                 "serve (off by default, since an account may reach a "
                                 "model absent from the catalog)")
        if name == "setup":
            sp.add_argument("--api-key-placeholder",
                            help="text for options.apiKey, e.g. {env:MY_KEY}")
            sp.add_argument("--reasoning", action="store_true",
                            help="request reasoning fields (known ineffective in "
                                 "OpenCode 1.18.x)")
    return parser


def _fetch_for(args: argparse.Namespace) -> dict:
    return fetch_document(
        url=args.url or os.environ.get("KIRO_GATEWAY_URL") or DEFAULT_URL,
        api_key=args.api_key or os.environ.get("KIRO_GATEWAY_KEY")
        or os.environ.get("PROXY_API_KEY"),
        provider=args.provider,
        base_url=args.base_url,
        api_key_placeholder=getattr(args, "api_key_placeholder", None),
        reasoning=getattr(args, "reasoning", False),
        timeout=args.timeout,
    )


def _apply(args: argparse.Namespace, merged: dict, existing: dict, path: str) -> int:
    if merged == existing:
        print(f"No changes needed for provider '{args.provider}' in {path}.")
        return EXIT_OK

    diff = render_diff(existing, merged, path)
    print(diff)

    if not args.write:
        print("\nDry run — nothing written. Re-run with --write to apply.")
        return EXIT_DRIFT

    backup = write_config(path, merged)
    if backup:
        print(f"\nBackup: {backup}")
    print(f"Wrote {path}.")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    path = config_path(args.config)

    try:
        existing = load_config(path)
        document = _fetch_for(args)

        if args.command == "doctor":
            findings = diagnose(existing, document, args.provider)
            if not findings:
                print(f"provider '{args.provider}' in {path} is up to date.")
                return EXIT_OK
            print(f"{len(findings)} issue(s) found in {path}:")
            for finding in findings:
                print(f"  - {finding}")
            return EXIT_DRIFT

        prune = bool(getattr(args, "prune", False))
        if args.command == "setup":
            merged = merge_provider(existing, document, args.provider, prune)
        else:
            merged = refresh_models(existing, document, args.provider, prune)
        return _apply(args, merged, existing, path)

    except HelperError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
