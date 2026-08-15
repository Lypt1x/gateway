# TODO

Verification backlog for this fork. Everything below is either untested or tested only in a
narrow configuration. Items are ordered by risk, highest first.

All live testing so far was done against exactly **one** configuration: a `kiro-cli` SQLite
credential, AWS SSO OIDC auth, `us-east-1`, with a valid profile ARN. Anything depending on a
different account type, region, or platform is unverified.

## 1. Dynamic model discovery on other account types and regions

Discovery fetches the catalog from `https://q.{api_region}.amazonaws.com/ListAvailableModels`
while chat stays on `runtime.{region}.kiro.dev`. Verified live on SSO OIDC / `us-east-1` only.

- [ ] Builder ID account (no profile ARN) — does listing authorize at all?
- [ ] Kiro Desktop (`AuthType.KIRO_DESKTOP`) — the legacy path sends `profileArn`; confirm parity
- [ ] Enterprise IdC account with a working (non-revoked) token
- [ ] Non-`us-east-1` region, e.g. `eu-central-1` — does the control-plane host even exist there?
- [ ] Account whose SSO region differs from its API region (`KIRO_REGION` vs `KIRO_API_REGION`)
- [ ] Confirm the discovered catalog differs per account tier, and that a smaller tier does not
      end up advertising models it cannot call

> [!NOTE]
> `KIRO_CONTROL_PLANE_HOST_TEMPLATE` (`kiro/config.py:376`) is a hardcoded constant, not an env
> var. Making it overridable would help anyone on a restricted network, and would make the
> failure path testable without forcing a timeout.

Also unresolved: `codewhisperer.{region}.amazonaws.com` answers the same operation and could
serve as a failover host, but no failover is implemented.

## 2. Reliability paths that have never seen a real failure

Each of these is covered by unit tests with injected faults, but never observed against a real
upstream fault.

- [ ] `#129` mid-stream transport drop — only ever injected, never a genuine upstream drop
- [ ] `#268` in-band OpenAI error — could not force first-token exhaustion; upstream answered
      quickly even at `FIRST_TOKEN_TIMEOUT=0.05`, on Claude and non-Claude models alike
- [ ] Upstream exception / throttling frames — no real `ThrottlingError` frame was ever seen, so
      the classification is unproven against a live payload. The issue reporter correlated these
      with ~80% credit usage, which suggests a way to reproduce
- [ ] `invalidState` frame handling
- [ ] `#203` token-refresh race — needs a genuinely concurrent host `kiro-cli` refreshing the
      same SQLite database. Plausibly explains this machine's revoked Enterprise token
- [ ] `MIDSTREAM_RESUME=true` — off by default; the overlap-trimming logic is the fragile part
      and has never run live

## 3. Event-stream decoder coverage

The decoder is on by default and confirmed to take the `framed` path in production, but only
against Claude models on the happy path plus one tool call.

- [ ] Non-Claude families under the decoder: `gpt-5.6-sol` / `-terra` / `-luna`, `deepseek-3.2`,
      `glm-5`, `minimax-*`, `qwen3-coder-next` — these emit reasoning differently
- [ ] `claude-opus-5` streaming and tool calls
- [ ] Parallel/multiple tool calls in one stream (only sequential `toolUseId` handling is tested)
- [ ] Very large tool arguments spanning many frames
- [ ] Legacy fallback path (`EVENTSTREAM_DECODER=false`) beyond the single A/B tool-call check
- [ ] Mid-stream framing corruption triggering the live fallback
- [ ] Whether dropping the legacy adjacent-duplicate content suppression changes any real output

> [!WARNING]
> The `toolUseEvent` regression is the cautionary tale here: unit tests passed while live tool
> calls returned `7` and `{}`. Frame-shape assumptions need live confirmation, not just fixtures.

## 4. Deployment and platform

- [ ] Docker deployment end to end — never run in this session. The credential volume mounts in
      `docker-compose.yml` are commented out by default, so first-run UX is unverified
- [ ] `#209` container user permissions on mounted credentials
- [ ] `#151` Windows path handling (upstream PR 152 exists)
- [ ] `VPN_PROXY_URL` routing, and the `FIX-08` proxy-precedence change, on a real proxy
- [ ] `FIX-08` offline tokenizer cache on a host with no network access to the tiktoken CDN
- [ ] Multi-account pooling and failover — only ever run with a single account, so rotation,
      state persistence across restarts, and error-driven failover are untested

## 5. Feature gaps and accuracy

- [ ] `#176` document blocks for textual media types (only a PDF was tested, which correctly
      degrades to a "cannot read" placeholder since Kiro has no document channel)
- [ ] `count_tokens` accuracy against real upstream usage numbers, not just internal consistency
- [ ] Expose the richer catalog metadata now retained in the cache (`tokenLimits`,
      `promptCaching`, `rateMultiplier`, `rateUnit`, `supportedInputTypes`) — satisfies `#156`
- [ ] `#168` chat-path `profileArn` failures — unreproducible here; needs an affected account.
      The MCP half is fixed and verified
- [ ] `#170` web search on other harnesses, e.g. opencode — may already work now that the MCP
      call is authorized
- [ ] Client identity: two User-Agent version tokens (`ua/2.1` and the client crate version)
      could not be isolated in the binary and are inferred

## 6. Known deliberate divergences from real kiro-cli

Not bugs. Recorded so they are not "fixed" by accident.

- `x-amzn-codewhisperer-optout` stays `true`; the real client sends `false`. Keeping `true` opts
  out of content being used upstream. Override with `CODEWHISPERER_OPTOUT` if desired.
- No mid-stream resume by default. The real client simply fails and surfaces the error, so
  matching it is correct rather than lazy.

## 7. Housekeeping

- [ ] The seven translated READMEs under `docs/` still describe the old README structure
- [ ] `README.md` now diverges from upstream, so merges from `jwadow/kiro-gateway` will conflict
- [ ] `tests/conftest.py` contains a JWT-shaped dummy token; harmless, but worth confirming it is
      not mistaken for a real credential by secret scanners
