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

The decoder is on by default and confirmed to take the `framed` path in production.

- [x] Non-Claude families under the decoder: verified `gpt-5.6-terra`, `qwen3-coder-next`,
      `glm-5`, `claude-opus-5`, `auto-kiro` all answer correctly through OpenCode
- [x] Tool calling on a non-Claude family (`gpt-5.6-terra` wrote a file via the tool loop)
- [x] All six upstream event types mapped and documented
      (`assistantResponseEvent`, `toolUseEvent`, `metadataEvent`, `contextUsageEvent`,
      `meteringEvent`, `reasoningContentEvent`)
- [ ] Parallel/multiple tool calls in one stream (only sequential `toolUseId` is tested)
- [ ] Very large tool arguments spanning many frames
- [ ] Legacy fallback path (`EVENTSTREAM_DECODER=false`) beyond the A/B checks
- [ ] Mid-stream framing corruption triggering the live fallback
- [ ] Whether dropping the legacy adjacent-duplicate content suppression changes real output
- [ ] Exception / throttle frames: still never observed live, so `:exception-type` handling
      remains unproven against a real payload

> [!WARNING]
> Two regressions in this area were found ONLY by live use, both with green unit tests.
> The `toolUseEvent` routing bug returned `7` and `{}` instead of real arguments, and the
> event-name mapping bug (`messageMetadata*` never existing on the wire) silently dropped
> completion, context and credit events. Frame-shape assumptions must be confirmed against
> captured traffic, not fixtures.

## 4. OpenCode integration

Verified end to end on 2026-08-15 with OpenCode 1.18.18 (native Linux build) against the
gateway: `GET /integrations/opencode.json` produced a config that registered all 19 models,
plain completions worked, and the full agentic tool loop wrote files correctly on both
`claude-sonnet-4.5` and `gpt-5.6-terra`.

- [x] Generated config imports cleanly; `opencode models kiro` lists all 19
- [x] Tool-calling loop works through the gateway
- [ ] **`reasoning: true` does not work.** With `"reasoning": true` plus
      `"interleaved": {"field": "reasoning_content"}`, OpenCode reports `"reasoning": 0` tokens
      and emits no reasoning part, even though the gateway demonstrably sends
      `reasoning_content` (41 SSE chunks in the same request). Harmless but ineffective, which
      is why `?reasoning=` defaults to false. The correct AI SDK config for surfacing reasoning
      through `@ai-sdk/openai-compatible` is still unknown.
- [x] Step 3: the setup / doctor / update wrapper — `opencode/opencode_config.py` (stdlib
      only, dry-run by default, merges `provider.<id>` only; see `opencode/README.md`).
      Original scope: `doctor` should diff a
      user's existing config against the live catalog and report models they list but cannot
      access, models they are missing, and stale limits. It must MERGE, never overwrite, since
      users keep agents, plugins and MCP config in the same file.
- [ ] Credit display (the reason the payload mapping was completed): `meteringEvent` carries
      `{"unit":"credit","unitPlural":"credits","usage":<float>}` and is now retained as
      `parser.last_metering`. Nothing consumes it yet. Exposing per-request credit spend would
      pair well with the `rate_multiplier` already on `/v1/models`.
- [ ] Attachment/vision capability is not expressed in the generated config. We know
      `supported_input_types` per model, but no documented OpenCode field was found for it, so
      nothing is emitted rather than inventing schema.
- [ ] Verify against a free-tier Kiro account that the generated config contains only the
      models that account can reach (the per-account rationale is sound but untested).

> [!NOTE]
> `opencode run` hangs indefinitely when stdin is not a TTY and not redirected. Use
> `opencode run ... < /dev/null` in scripts and CI. This cost real debugging time and looked
> like a gateway hang; it is not.

## 5. Deployment and platform

- [ ] Docker deployment end to end — never run in this session. The credential volume mounts in
      `docker-compose.yml` are commented out by default, so first-run UX is unverified
- [ ] `#209` container user permissions on mounted credentials
- [ ] `#151` Windows path handling (upstream PR 152 exists)
- [ ] `VPN_PROXY_URL` routing, and the `FIX-08` proxy-precedence change, on a real proxy
- [ ] `FIX-08` offline tokenizer cache on a host with no network access to the tiktoken CDN
- [ ] Multi-account pooling and failover — only ever run with a single account, so rotation,
      state persistence across restarts, and error-driven failover are untested

## 6. Feature gaps and accuracy

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

## 7. Known deliberate divergences from real kiro-cli

Not bugs. Recorded so they are not "fixed" by accident.

- `x-amzn-codewhisperer-optout` stays `true`; the real client sends `false`. Keeping `true` opts
  out of content being used upstream. Override with `CODEWHISPERER_OPTOUT` if desired.
- No mid-stream resume by default. The real client simply fails and surfaces the error, so
  matching it is correct rather than lazy.

## 8. Housekeeping

- [ ] The seven translated READMEs under `docs/` still describe the old README structure
- [ ] `README.md` now diverges from upstream, so merges from `jwadow/kiro-gateway` will conflict
- [ ] `tests/conftest.py` contains a JWT-shaped dummy token; harmless, but worth confirming it is
      not mistaken for a real credential by secret scanners
