# Grok Build config helper

`grok_build_config.py` — a stdlib-only, single-file CLI that wires the
[Grok Build](https://docs.x.ai) CLI (`grok`, 1.0.4) up to this gateway. It does **one**
HTTP GET against the gateway's own `GET /integrations/grok-build.toml` and edits
`~/.grok/config.toml`. No Kiro auth, token-refresh, region or catalog logic lives here —
the gateway owns all of that.

```bash
python grok-build/grok_build_config.py setup  --url http://localhost:8000 --api-key "$KIRO_GATEWAY_KEY"
python grok-build/grok_build_config.py doctor
python grok-build/grok_build_config.py update --write
```

Then export the key Grok will read and start it:

```bash
export KIRO_GATEWAY_KEY="<the gateway's proxy API key>"
grok
```

No `grok login` is needed: setting `models_base_url` switches Grok to API-key auth
(`Authorization: Bearer`), per its own documentation.

## What gets written

Only these tables, appended to your config:

```toml
[endpoints]
models_base_url = "http://localhost:8000/v1"

[models]
default = "claude-sonnet-4.5"
session_summary = "claude-haiku-4.5"   # cheapest visible model by rate_multiplier
image_description = "claude-sonnet-4.5" # omitted when no visible model accepts IMAGE
web_search = "claude-sonnet-4.5"        # same as default

[model."claude-sonnet-4.5"]
model = "claude-sonnet-4.5"
name = "Claude Sonnet 4.5"
api_backend = "chat_completions"
context_window = 200000          # omitted when upstream does not report it
max_completion_tokens = 64000    # omitted when upstream does not report it
env_key = "KIRO_GATEWAY_KEY"     # a variable NAME, never a secret value
```

`stream_tool_calls` is deliberately never emitted: Grok's docs note it changes request
*shape*, and some BYOK endpoints expect it unset. `supports_backend_search` is not emitted
either — we have not verified this gateway satisfies Grok's backend-search contract.

### Why the auxiliary roles matter

`[models]` accepts exactly `default`, `web_search`, `image_description` and
`session_summary`. Any role left unset is resolved by Grok against its **own built-in xAI
model**, so with only `default` set every session logged
`no credentials for auxiliary model; falling back to active model aux_model=grok-4.6` and
fired one request at this gateway for `grok-4.6`, which upstream rejects
(`INVALID_MODEL_ID`, HTTP 400) — session titles then silently degraded to truncated user
text. Pinning the roles takes that to zero failing requests per session. `session_summary`
uses the cheapest visible model because titles are throwaway work; `image_description` is
omitted rather than pointed at a text-only model.


## Subcommands

| Command  | Does |
|----------|------|
| `setup`  | fetches the live document and **merges** our tables into your config |
| `doctor` | fetches the live document, diffs it against your config, reports only |
| `update` | refreshes the managed tables in place (requires a prior `setup`) |

`doctor` reports: a missing `[endpoints] models_base_url`, one that does not end in `/v1`
or does not match this gateway, models you list that upstream does not offer, models
available upstream that you are missing, a stale `context_window`, an `api_backend` other
than `chat_completions`, a missing `env_key`, an inline `api_key`, and an `env_key` that
holds a literal secret instead of a variable name. It also reports a missing
`[models] session_summary` or `web_search`, and any of `session_summary`,
`web_search` or `image_description` naming a model this gateway does not serve —
explaining that Grok would otherwise fall back to its built-in xAI model and every session
would emit a failing request. Secret values are never printed.

## Options

| Flag | Default |
|------|---------|
| `--config PATH` | `~/.grok/config.toml` |
| `--url URL` | `$KIRO_GATEWAY_URL`, else `http://localhost:8000` |
| `--api-key KEY` | `$KIRO_GATEWAY_KEY`, else `$PROXY_API_KEY` |
| `--base-url URL` | gateway-derived `endpoints.models_base_url` |
| `--env-key NAME` | `KIRO_GATEWAY_KEY` |
| `--write` | off — `setup` / `update` are **dry run** by default |

## Safety

- **Your file is preserved as text.** The stdlib has no TOML writer, and round-tripping
  the document through a parser would destroy comments, blank lines and key order. So the
  helper drops only the table blocks it owns (`[endpoints]`, `[models]`, `[model.*]`) and
  appends fresh ones. `[cli] installer = "npm"`, `[mcp_servers]`, `[ui]`, `[permission]`
  and every comment survive byte-for-byte. `tomllib` is used for reading, validation and
  comparison only.
- `setup` and `update` print a diff and touch nothing unless `--write`.
- With `--write` the file is copied to `<config>.bak-YYYYmmdd-HHMMSS`, then the new content
  is written to a temp file in the same directory and `os.replace`d in, so an interrupted
  run cannot truncate your config.
- No credential is ever written or printed: only the *name* of an environment variable.
  Diffs show `<redacted>` for anything secret-looking.
- A gateway that is down, returns 401, or answers with non-TOML — or a local config that
  does not parse — produces a one-line message and exit code 2, never a traceback.

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | healthy / nothing to change (or `--write` succeeded) |
| `1` | drift found, or changes pending in a dry run |
| `2` | usage error, connection failure, bad key, malformed TOML |

Tests: `tests/unit/test_grok_build_config_cli.py` and
`tests/unit/test_grok_build_integration.py` — fully offline; the HTTP fetch is stubbed,
configs live in `tmp_path`, and no `grok` binary is ever invoked.
