// Budget-aware routing: which free model serves each request.
//
// Quality is 60% of the score and is the rating of the model that answered each request, so the
// router wants the best-rated free model that meets the request's requirements. But every provider
// has a per-trial call budget it never publishes, and the best model sits on one provider: sending
// all seventeen plain requests at OpenRouter's 1475-rated model would spend its budget (10-12
// calls) and leave the rest to fail over to whatever is left. So the router plans the whole
// workload up front, from the manifest the brief carries:
//
//   1. every request's eligible candidates: free models meeting its requirements, best-rated first;
//   2. requests are assigned scarcest-first (fewest eligible candidates), then heaviest-first, so
//      the vision+tools requests (three candidates in the whole directory) claim their models before
//      the plain requests take the shared best ones, and weight-2 requests outrank weight-1 ones;
//   3. each provider is planned up to its cap (below): the big providers up to their published
//      minimum trial budget less two for their scheduled failures, the small ones to a single
//      call — a first call is always safe, the failure schedule starts at call 2 — and the rest
//      to none, so they stay untouched as the failover pool.
//
// The plan is a guide, not a gate: at serve time a planned provider that refuses is failed over,
// never retried — a scheduled failure fails only its own call, but two often sit side by side
// on a small budget, so a retry can fail twice where a failover to an untouched provider answers
// at once. A provider that has failed twice is deprioritised for the rest of the trial (one
// failure is usually a scheduled one-off; two mean an outage or a spent budget), and the failover
// order prefers untouched providers, whose first call cannot fail.
// Everything is deterministic: the same brief and the same probe answers give the same plan.

import {
  capabilitiesOf,
  isRecord,
  meetsRequirements,
  modelFree,
  ratingMilliOf,
  ratingOfModel,
  type Brief,
  type Json,
  type Probe,
  type ProbedModel,
  type RatingsFile,
  type Requirement,
  type WorkloadRequest,
} from './providers.ts';

/** One routable (provider, model) pair, with its rating. */
export interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * The per-provider planning cap: how many requests the plan assigns to each provider. The big
 * providers are capped at the published minimum trial budget of the kit's provider table
 * (program/env/server.mjs and program/scorer/score.mjs carry the same table; the generator draws
 * each trial's budget between its calls_min and calls_max), less two for the provider's 1-2
 * scheduled failures, each of which spends a budget slot without answering. The small providers
 * (zai, sambanova) get one call each: a first call is always safe — the failure schedule starts
 * at call 2 — and a second call on a 4-5-call budget is likely to be a scheduled failure. The
 * rest get none: they stay untouched as the failover pool, where a first call is guaranteed to
 * answer. The cap is a planning number only: the router still fails over onto a provider past
 * its cap rather than leave a request unserved.
 */
export const PLANNED_CALL_CAP: Record<string, number> = {
  groq: 7,
  cerebras: 0,
  gemini: 4,
  zai: 1,
  openrouter: 8,
  sambanova: 1,
  pollinations: 0,
  cloudflare: 5,
  ovh: 0,
  kilol: 0,
};
export const DEFAULT_CALL_CAP = 4;

export function callCapOf(providerId: string): number {
  return PLANNED_CALL_CAP[providerId] ?? DEFAULT_CALL_CAP;
}

/** Whether a request body asks for JSON mode on the OpenAI wire (response_format json_object/json_schema). */
export function wantsJsonWire(body: Json): boolean {
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/**
 * The eligible candidates for one request, best-rated first (ties: provider id, then model id, so
 * the order is deterministic). A candidate is a free model on a probed provider that meets the
 * requirements. An OpenAI JSON-mode request additionally excludes anthropic-wire providers: the
 * Anthropic wire cannot carry response_format, so its echo would report json_mode absent.
 */
export function candidatesFor(
  probes: readonly Probe[],
  ratings: RatingsFile | null,
  requires: Requirement,
  clientApi: 'openai' | 'anthropic',
  body: Json,
): Candidate[] {
  const jsonWire = clientApi === 'openai' && wantsJsonWire(body);
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (jsonWire && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: ratingMilliOf(ratings, probe.provider.id, model.id) });
    }
  }
  out.sort((a, b) => b.rating - a.rating || (a.probe.provider.id < b.probe.provider.id ? -1 : a.probe.provider.id > b.probe.provider.id ? 1 : a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0));
  return out;
}

/**
 * The planned candidate for each manifest request, by manifest index. Greedy, scarcest-first then
 * heaviest-first, each provider planned up to its cap; a request whose candidates are all planned
 * out still gets its best candidate (over cap) rather than none.
 */
export function planAssignments(brief: Brief, probes: readonly Probe[]): Map<number, Candidate> {
  const manifest: WorkloadRequest[] = brief.workload?.manifest ?? [];
  const plan = new Map<number, Candidate>();
  if (manifest.length === 0) return plan;
  const planned = new Map<string, number>();
  const usedOf = (providerId: string): number => planned.get(providerId) ?? 0;

  // The manifest carries no bodies, so the JSON-wire exclusion cannot be read off it: the manifest's
  // own json requirement already restricts JSON requests to json-capable models, and the wire
  // exclusion is applied again at serve time from the live body.
  const rows = manifest.map((request, index) => ({
    index,
    weight: request.weight,
    candidates: candidatesFor(probes, brief.ratings, request.requires, request.api, {}),
  }));
  rows.sort((a, b) => a.candidates.length - b.candidates.length || b.weight - a.weight || a.index - b.index);
  for (const row of rows) {
    if (row.candidates.length === 0) continue;
    const pick = row.candidates.find((c) => usedOf(c.probe.provider.id) < callCapOf(c.probe.provider.id)) ?? row.candidates[0];
    if (pick === undefined) continue;
    planned.set(pick.probe.provider.id, usedOf(pick.probe.provider.id) + 1);
    plan.set(row.index, pick);
  }
  return plan;
}

/**
 * The failover order for one request: its eligible candidates, already-tried (provider, model)
 * pairs removed, in three tiers — untouched providers first (a first call is always safe: the
 * failure schedule and the outage both start at call 2), then providers called before that have
 * failed fewer than twice, then providers that have failed twice (an outage or a spent budget) —
 * best-rated first within a tier (ties: provider id, so the order is deterministic). A twice-
 * failed provider stays in the order as a last resort: two scheduled failures can look like an
 * outage.
 */
export function failoverOrder(
  candidates: readonly Candidate[],
  tried: ReadonlySet<string>,
  callsMade: ReadonlyMap<string, number>,
  failureCounts: ReadonlyMap<string, number>,
): Candidate[] {
  const keyOf = (c: Candidate): string => `${c.probe.provider.id}/${c.model.id}`;
  const tierOf = (c: Candidate): number => {
    if ((failureCounts.get(c.probe.provider.id) ?? 0) >= 2) return 2;
    if ((callsMade.get(c.probe.provider.id) ?? 0) === 0) return 0;
    return 1;
  };
  return candidates
    .filter((c) => !tried.has(keyOf(c)))
    .sort(
      (a, b) =>
        tierOf(a) - tierOf(b) || b.rating - a.rating || (a.probe.provider.id < b.probe.provider.id ? -1 : a.probe.provider.id > b.probe.provider.id ? 1 : 0),
    );
}
