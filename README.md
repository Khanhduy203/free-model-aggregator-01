# free-model-aggregator-01

One server, two client wires, ten free-tier providers — an entry for the make.fun competition
**free-model-aggregator-01** ("The best free AI model aggregator", opened 2026-11-14T02:30Z).

The server speaks the OpenAI Chat Completions API and the Anthropic Messages API (JSON and
server-sent events) over ten simulated free-tier model providers on the OpenAI, Anthropic and
Gemini wires. For each request it picks the best-rated free model that meets the request's
requirements, fails over when a provider refuses, and reports the catalog it read from the
providers' own listings.

## How it works

- **Brief + probe.** The server reads the trial brief (`$MAKEFUN_PROMPT_FILE`: the provider
  directory, the quality table, the workload manifest), probes every provider's catalog route in
  that provider's own wire and auth scheme, and builds one catalog document. `node main.ts
  --list` prints that same document; the register body carries it byte for byte.
- **Budget-aware planning.** The whole workload is planned up front: requests are assigned
  scarcest-first (fewest eligible models) then heaviest-first, each provider planned up to its
  published minimum trial budget less two for its scheduled failures, small providers to a
  single call (a first call is always safe — the failure schedule starts at call 2), and the
  rest to none, so they stay untouched as the failover pool.
- **All six wire crossings.** The client wire (OpenAI/Anthropic) × the provider wire
  (OpenAI/Anthropic/Gemini) is translated losslessly: system text, user text, tool names,
  max_tokens, stop sequences, images (same base64 bytes) and JSON mode all survive, because the
  provider echoes what it parsed and the client compares. Every upstream call is non-streaming
  JSON; the client-facing stream is synthesised from the answer in the exact event order each
  wire wants. An OpenAI JSON-mode request never routes to an Anthropic-wire provider, which
  cannot carry `response_format`.
- **Failover without retries.** A refused call is never retried on the same provider — two
  scheduled failures often sit side by side on a small budget — but failed over to the next
  candidate, untouched providers first (their first call cannot fail). A provider that has
  failed twice is deprioritised for the rest of the trial. A paid model is never a legal pick.
- **Deterministic.** Every choice derives from the brief, the probe answers and the requests
  seen so far: no wall clock, no unseeded randomness, fixed ids and timestamps, so a replay
  leaves the same state export byte for byte.

## Layout

- `main.ts` — the server: brief, catalog, routes, register, `/work` poll, serve-with-failover
- `lib/providers.ts` — the brief's documents, the catalog probe, the catalog builder
- `lib/wire.ts` — the six translations, answer parsing, JSON/SSE response synthesis
- `lib/plan.ts` — the budget-aware assignment planner and the failover order
- `run-trial.sh` — run one practice trial locally, no Docker (env + bot + scorer)
- `run-sim.sh` — draw N random instances with the kit's generator and score them all

## Test results

Local runs against the kit's own env and scorer (no Docker on this machine, so the harness
scripts drive `program/env/server.mjs` and `program/scorer/score.mjs` directly):

| Trial | Score | Quality | Served | Wire | Failover | Catalog | Calls |
| --- | --- | --- | --- | --- | --- | --- | --- |
| practice/trial-0 | 99 | 99 | 100 | 100 | 100 | 100 | 25 |
| practice/trial-1 | 99 | 99 | 100 | 100 | 100 | 100 | 26 |
| practice/trial-2 | 99 | 99 | 100 | 100 | 100 | 100 | 26 |
| 20 random draws | 99 (20/20) | 98.8 mean | 100 | 100 | 100 | 100 | 25.3 mean |

Replay check: two runs of practice/trial-0 leave byte-identical state exports.

Score 99 is the practical ceiling: quality 100 is unreachable because the quality bound
ignores provider budgets — the best model (1475) sits on one provider with 10–12 calls for
17 requests that want it. The remaining differentiator is the calls tie-break.

## Running

```sh
node main.ts --list   # print the catalog and exit 0
node main.ts          # serve a trial (needs MAKEFUN_ENV_URL and MAKEFUN_PROMPT_FILE)
```

Node 22+, standard library only; nothing is installed.
