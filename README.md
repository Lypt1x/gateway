<div align="center">

# 👻 Kiro Gateway

**Use your Kiro subscription from any OpenAI or Anthropic compatible tool**

[English](README.md) • [Русский](docs/ru/README.md) • [中文](docs/zh/README.md) • [Español](docs/es/README.md) • [Indonesia](docs/id/README.md) • [Português](docs/pt/README.md) • [日本語](docs/ja/README.md) • [한국어](docs/ko/README.md)

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)

</div>

A local proxy that speaks the OpenAI and Anthropic APIs and translates them to the Kiro API
(Amazon Q Developer / AWS CodeWhisperer). Point any compatible client at it and use the models
your Kiro account already provides.

Works with Claude Code, OpenCode, Codex, Cursor, Cline, Roo Code, Kilo Code, Continue, Obsidian,
the OpenAI SDK, LangChain, and anything else that can target a custom base URL.

## Quick start

**Requirements:** Python 3.10 or newer, and an authenticated Kiro IDE or `kiro-cli` installation.

```bash
git clone https://github.com/jwadow/kiro-gateway.git
cd kiro-gateway
pip install -r requirements.txt
```

Point the gateway at your existing credentials by creating `credentials.json`:

```json
[
  { "type": "json", "path": "~/.aws/sso/cache/kiro-auth-token.json" }
]
```

Then start it:

```bash
PROXY_API_KEY=pick-your-own-secret python main.py
```

Send it a request:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer pick-your-own-secret" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-sonnet-4.5","messages":[{"role":"user","content":"Hello"}]}'
```

> [!IMPORTANT]
> `PROXY_API_KEY` is the password protecting *your* gateway, not an Amazon key. It has a
> well-known default, and the server binds `0.0.0.0` by default. Always set your own key, and
> set `SERVER_HOST=127.0.0.1` unless you deliberately want it reachable from your network.

## Features

- **Two dialects, one endpoint.** OpenAI `/v1/chat/completions` and Anthropic `/v1/messages`,
  including streaming, tool calling, and token counting.
- **Four credential sources.** Kiro IDE JSON, AWS SSO OIDC cache, the `kiro-cli` SQLite
  database, or a raw refresh token. Tokens refresh automatically.
- **Account pooling.** Run several accounts with automatic failover when one is rate limited or
  out of credits.
- **Native event-stream decoding.** Upstream responses are parsed as real AWS event-stream
  frames with CRC validation, so tool arguments, usage metadata, and upstream errors survive
  intact.
- **Streams that end cleanly.** Transport drops and upstream exception frames are reported
  in-band and close the stream as a well-formed turn, so agent harnesses recover instead of
  aborting.
- **Reasoning support.** Thinking blocks are surfaced as Anthropic `thinking` blocks or OpenAI
  `reasoning_content`.
- **Runs anywhere.** Single process, or `docker compose up`.

## Models

Fifteen models are exposed. `GET /v1/models` returns whatever your account can actually reach.

| Family | Models |
| --- | --- |
| Claude Sonnet | `claude-sonnet-4`, `claude-sonnet-4.5`, `claude-sonnet-4.6`, `claude-sonnet-5` |
| Claude Opus | `claude-opus-4.5`, `claude-opus-4.6`, `claude-opus-4.7`, `claude-opus-4.8` |
| Claude Haiku | `claude-haiku-4.5` |
| Other | `deepseek-3.2`, `glm-5`, `minimax-m2.1`, `minimax-m2.5`, `qwen3-coder-next` |
| Automatic | `auto-kiro` (lets Kiro choose) |

Unlisted model IDs are passed through to the upstream API unchanged, so a newly released model
usually works before it appears here.

## Configuration

### Credentials

`credentials.json` holds a list of accounts. Each entry names a source:

```json
[
  { "type": "json",   "path": "~/.aws/sso/cache/kiro-auth-token.json" },
  { "type": "sqlite", "path": "~/.local/share/kiro-cli/data.sqlite3" },
  { "type": "refresh_token", "refresh_token": "eyJ...", "region": "us-east-1" }
]
```

| Type | Source | Notes |
| --- | --- | --- |
| `json` | Kiro IDE / AWS SSO cache file | Enterprise installs are detected automatically |
| `sqlite` | `kiro-cli` database | Also supplies the profile ARN |
| `refresh_token` | Token supplied directly | Useful in containers and CI |

A `path` may point at a directory to scan every credential file inside it. Per-account
`profile_arn`, `region`, `api_region`, and `enabled` overrides are supported.

> [!TIP]
> If a stored refresh token has been revoked, the gateway now says so explicitly and tells you
> to log in again rather than reporting a generic configuration error.

### Environment variables

| Variable | Default | Description |
| --- | --- | --- |
| `PROXY_API_KEY` | *(insecure default)* | Key clients must present. Comma-separate for multiple keys |
| `SERVER_HOST` / `SERVER_PORT` | `0.0.0.0` / `8000` | Listen address |
| `ACCOUNT_SYSTEM` | `false` | Enable the multi-account pool and failover |
| `PROFILE_ARN` | *(auto)* | CodeWhisperer profile ARN override |
| `KIRO_REGION` / `KIRO_API_REGION` | `us-east-1` | SSO region and API region |
| `FIRST_TOKEN_TIMEOUT` | `15` | Seconds to wait for the first token before retrying |
| `STREAMING_READ_TIMEOUT` | `300` | Idle timeout while streaming |
| `EVENTSTREAM_DECODER` | `true` | Decode real event-stream frames; `false` uses the legacy parser |
| `MIDSTREAM_RESUME` | `false` | Attempt one continuation after a mid-stream drop |
| `TRUNCATION_RECOVERY` | `true` | Tell the model when a previous turn was cut short |
| `VPN_PROXY_URL` | *(none)* | Route upstream traffic through an HTTP or SOCKS5 proxy |
| `HIDDEN_MODELS` | *(none)* | Hide models from `/v1/models` |
| `LOG_LEVEL` | `INFO` | `DEBUG` prints full request and stream detail |

Run `python main.py --help` for command line overrides.

## Usage

### OpenAI clients

Set the base URL to `http://localhost:8000/v1` and the API key to your `PROXY_API_KEY`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="pick-your-own-secret")

stream = client.chat.completions.create(
    model="claude-sonnet-4.5",
    messages=[{"role": "user", "content": "Explain event-stream framing briefly"}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

### Anthropic clients

```bash
export ANTHROPIC_BASE_URL=http://localhost:8000
export ANTHROPIC_API_KEY=pick-your-own-secret
claude
```

Inline `system` messages, `document` blocks, and non-standard roles are accepted and normalised,
so clients that deviate from the published Anthropic schema still work.

### OpenCode

OpenCode does not auto-discover models for custom providers, so every model has to be listed
in its config. The gateway generates that list for you from your account's live catalog:

```bash
export KIRO_GATEWAY_KEY=pick-your-own-secret

python opencode/opencode_config.py setup --url http://localhost:8000 --write
```

That merges a `kiro` provider block into `~/.config/opencode/opencode.json` with every model
your account can reach, including correct context and output limits. Then pick a model:

```bash
opencode run --model kiro/claude-sonnet-4.5 "Explain this repo" < /dev/null
```

> [!TIP]
> `opencode run` hangs if stdin is neither a terminal nor redirected. Add `< /dev/null` when
> scripting it.

The helper has three subcommands, and `setup`/`update` are dry runs unless you pass `--write`:

| Command | Purpose |
| --- | --- |
| `setup` | Merge the provider block into your config |
| `doctor` | Report drift against the live catalog, changing nothing |
| `update` | Refresh models and limits only |

`doctor` reports models you list that your account cannot reach, models you are missing, stale
context/output limits, a wrong `npm` package, a `baseURL` missing `/v1`, and an `apiKey` holding
a literal secret instead of an `{env:NAME}` reference. It exits `1` when it finds drift, so it
works in CI. Add `--prune` to `setup` or `update` to also drop models the catalog no longer
serves.

Useful flags: `--config` for a non-default path, `--provider` to rename the provider block,
`--base-url` if OpenCode reaches the gateway on a different address (Docker, LAN), and
`--reasoning` to opt into reasoning fields.

> [!IMPORTANT]
> Only the `provider.<id>` block is ever touched, and `--write` takes a timestamped backup
> first, so your agents, plugins, MCP servers and other providers are left alone. Your API key
> is never written into the config: the generated `{env:KIRO_GATEWAY_KEY}` reference is resolved
> by OpenCode at runtime.

If you prefer to wire it up by hand, fetch the same document directly:

```bash
curl -H "Authorization: Bearer $KIRO_GATEWAY_KEY" \
  http://localhost:8000/integrations/opencode.json
```

Reasoning is off by default. Enabling it sets `reasoning` plus
`interleaved.field: reasoning_content`, which the gateway does emit, but OpenCode was observed
to report zero reasoning tokens for it. It is harmless and currently has no effect.

### Grok Build

xAI's [Grok Build](https://x.ai/cli) CLI supports a custom OpenAI-compatible backend, so it can
run against the gateway with no xAI subscription. Setting a custom models endpoint switches it to
API-key auth, so `grok login` is not needed.

```bash
export KIRO_GATEWAY_KEY=pick-your-own-secret
export XAI_API_KEY="$KIRO_GATEWAY_KEY"

python grok-build/grok_build_config.py setup --url http://localhost:8000 --write
grok -p "Explain this repo"
```

> [!IMPORTANT]
> Grok Build reads the key from the **environment**; it has no file-reference syntax. Both
> variables are needed: the per-model `env_key` covers inference, while `XAI_API_KEY`
> authenticates the model-list fetch. Without them Grok falls back to its grok.com login and
> sends an xAI token, which the gateway rejects with 401 — the symptom is a login prompt
> followed by every model failing. Put both in your shell profile so interactive `grok` sessions
> inherit them, and run `doctor`, which checks for them explicitly.

That merges the gateway's sections into `~/.grok/config.toml`: an `[endpoints] models_base_url`
pointing at the gateway, and a `[model."<id>"]` table per model with its real context window.
`setup`, `doctor` and `update` behave the same as the OpenCode helper, including the dry-run
default and `--write` backups, and only the sections the gateway owns are rewritten.

> [!NOTE]
> Grok Build uses separate models for auxiliary work: session titles, prompt suggestions, image
> descriptions and web search. Left unset, each falls back to its own xAI model and every session
> fires a request the gateway cannot serve. The generated config pins all of them, choosing the
> cheapest available model by rate multiplier for throwaway work.

Check what Grok actually resolved with `grok models` and `grok inspect`.

### Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/v1/chat/completions` | OpenAI chat completions, streaming and non-streaming |
| `POST` | `/v1/messages` | Anthropic messages, streaming and non-streaming |
| `POST` | `/v1/messages/count_tokens` | Anthropic token estimation |
| `GET` | `/v1/models` | Models available to the active account, with limits and rate info |
| `GET` | `/integrations/opencode.json` | Ready-to-merge OpenCode provider config |
| `GET` | `/integrations/grok-build.toml` | Ready-to-merge Grok Build config |
| `GET` | `/health` | Liveness check |
| `GET` | `/docs` | Interactive OpenAPI documentation |

## Docker

```bash
docker compose up -d
docker compose logs -f
```

Configuration comes from the `environment` block or an `env_file` in `docker-compose.yml`.

> [!NOTE]
> The credential volume mounts in `docker-compose.yml` are commented out by default. Uncomment
> the mount for your credential source before starting, for example
> `~/.aws/sso/cache:/home/kiro/.aws/sso/cache:ro`. If the container cannot read the file, check
> that it is readable by the container user.

## Running as a background service

On Linux (including WSL with `systemd=true` in `/etc/wsl.conf`) the gateway can run as a
systemd **user** service, so it starts at boot and is always there for your editor.

Keep the key in a file rather than the unit, so it never lands in your shell history:

```bash
mkdir -p ~/.config/kiro-gateway ~/.local/share/kiro-gateway
python3 -c "import secrets,pathlib;p=pathlib.Path.home()/'.config/kiro-gateway/key';p.write_text(secrets.token_urlsafe(32));p.chmod(0o600)"

cat > ~/.config/kiro-gateway/env <<EOF
PROXY_API_KEY=$(cat ~/.config/kiro-gateway/key)
SERVER_HOST=127.0.0.1
SERVER_PORT=8000
ACCOUNT_SYSTEM=1
EOF
chmod 600 ~/.config/kiro-gateway/env
```

Install and enable the unit:

```bash
cp service/kiro-gateway.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now kiro-gateway.service

# start at boot without having to log in first
sudo loginctl enable-linger "$USER"
```

Check it and read its log:

```bash
systemctl --user status kiro-gateway.service
tail -f ~/.local/share/kiro-gateway/service.log
```

Then point OpenCode at the service, letting it read the key from the same file:

```bash
python opencode/opencode_config.py setup \
  --url http://127.0.0.1:8000 \
  --api-key "$(cat ~/.config/kiro-gateway/key)" \
  --api-key-placeholder '{file:'"$HOME"'/.config/kiro-gateway/key}' \
  --write
```

No environment variable is needed after that: OpenCode resolves the `{file:...}` reference
itself.

> [!WARNING]
> Run only **one** gateway instance per credential source. Two instances sharing the same
> `kiro-cli` SQLite database will both try to rotate the same refresh token, and because AWS
> SSO OIDC refresh tokens are single-use, one can invalidate the other's credentials. Stop any
> manually started `python main.py` before enabling the service.

The unit sets `Restart=always`, so a crash is recovered automatically. It binds `127.0.0.1` via
the env file, which keeps it off your network.

## Account pooling

Set `ACCOUNT_SYSTEM=true` and list several accounts in `credentials.json`. The gateway keeps one
active and moves to the next when it hits a rate limit, exhausted credits, or an auth failure,
persisting its position across restarts. Failures are logged with the upstream reason so you can
tell a revoked token from a throttled one.

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| `401` from the gateway | `PROXY_API_KEY` mismatch between client and server |
| `Failed to initialize any account` | Credentials expired or revoked; the log names the reason |
| `profileArn is required for this request` | No profile ARN available; set `PROFILE_ARN` |
| Streams end early with no error | Run with `LOG_LEVEL=DEBUG` and check for upstream exception frames |
| Web search returns an error | The account needs a profile ARN for MCP calls |

Set `LOG_LEVEL=DEBUG` to see which stream parse path is in use, per-attempt retries, and full
upstream payloads.

## Documentation

- [Architecture](docs/en/ARCHITECTURE.md) — request flow, converters, and streaming internals
- Translations live under [`docs/`](docs/)

> [!WARNING]
> This project is not affiliated with, endorsed by, or sponsored by Amazon Web Services,
> Anthropic, or Kiro IDE. Use it at your own risk and in compliance with the terms of service of
> the underlying APIs.

---

<div align="center">

Created by [@Jwadow](https://github.com/jwadow) — [support the project](https://app.lava.top/products/b4e34d12-3b6b-49b7-be50-50b6a20ed262/f3ea941f-de73-4ad1-bbb6-f82042ef8132)

</div>
