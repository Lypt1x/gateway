# OpenCode config helper

`opencode_config.py` — a stdlib-only, single-file CLI that wires OpenCode up to this
gateway. It is deliberately thin: it does **one** HTTP GET against the gateway's own
`GET /integrations/opencode.json` and edits a local JSON file. It contains no Kiro auth,
token-refresh, region, header or catalog logic — the gateway owns all of that.

```bash
python opencode/opencode_config.py setup   --url http://localhost:8000 --api-key "$KIRO_GATEWAY_KEY"
python opencode/opencode_config.py doctor
python opencode/opencode_config.py update  --write
```

## Subcommands

| Command  | Does |
|----------|------|
| `setup`  | fetches the live document and **merges** `provider.<id>` into your config |
| `doctor` | fetches the live document, diffs it against your config, reports only |
| `update` | refreshes the managed provider's models map and limits in place |

`doctor` reports: models you list that upstream does not offer (calls would fail), models
available upstream that you are missing, `limit.context` / `limit.output` that are stale,
a wrong or missing `npm` package, a `baseURL` that does not end in `/v1`, and an `apiKey`
that looks like a literal secret instead of a `{env:…}` / `{file:…}` reference.

## Options

| Flag | Default |
|------|---------|
| `--config PATH` | `~/.config/opencode/opencode.json` |
| `--url URL` | `$KIRO_GATEWAY_URL`, else `http://localhost:8000` |
| `--api-key KEY` | `$KIRO_GATEWAY_KEY`, else `$PROXY_API_KEY` |
| `--provider ID` | `kiro` |
| `--base-url URL` | gateway-derived `options.baseURL` |
| `--write` | off — `setup` / `update` are **dry run** by default |
| `--api-key-placeholder` (`setup`) | `{env:KIRO_GATEWAY_KEY}` |
| `--reasoning` (`setup`) | off; known ineffective in OpenCode 1.18.x |

OpenCode loads `~/.config/opencode/config.json`, then `opencode.json`, then
`opencode.jsonc`; use `--config` if yours is not the default.

## Safety

- Only the `provider.<id>` subtree is ever touched. `agent`, `plugin`, `mcp`,
  `permission`, keybinds and other providers are copied through unchanged, in their
  original key order. The file is never overwritten wholesale.
- `setup` and `update` print a diff and exit without touching disk unless `--write`.
- With `--write` the existing file is copied to `<config>.bak-YYYYmmdd-HHMMSS` first, then
  the new content is written to a temp file in the same directory and `os.replace`d in, so
  an interrupted run cannot truncate your config.
- No real credential is ever written: the endpoint's `{env:…}` placeholder is kept, and an
  existing literal value in your config is left untouched. Secret values are never printed
  — `doctor` refers to them by key name and diffs show `<redacted>`.
- A gateway that is down, returns 401, or answers with non-JSON produces a one-line
  message and exit code 2 — never a traceback.

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | healthy / nothing to change (or `--write` succeeded) |
| `1` | drift found, or changes pending in a dry run — useful for CI |
| `2` | usage error, connection failure, bad key, malformed JSON |

Tests: `tests/unit/test_opencode_config_cli.py` (26 tests, fully offline — the HTTP fetch
is stubbed and configs live in `tmp_path`).
