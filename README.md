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

### Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/v1/chat/completions` | OpenAI chat completions, streaming and non-streaming |
| `POST` | `/v1/messages` | Anthropic messages, streaming and non-streaming |
| `POST` | `/v1/messages/count_tokens` | Anthropic token estimation |
| `GET` | `/v1/models` | Models available to the active account |
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
