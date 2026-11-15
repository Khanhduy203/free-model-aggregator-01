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
 * The per-provider planning cap: how many requests the plan assigns to each provider. Every
 * provider's first call is safe (the failure schedule and the outage both start at call 2), so
 * the plan uses all ten providers' first calls, then buys its extra calls where they are
 * cheapest. A provider's marginal failure risk per extra call follows its budget: the big
 * providers (kilol 12-16, pollinations 10-14, openrouter 10-12, groq 9-12) spread 1-2 failures
 * over many call numbers, so their 2nd-7th calls each cost only ~0.13-0.26 expected failures;
 * the small providers (zai, sambanova 4-5, cerebras, ovh 5-6) concentrate theirs, so a 2nd call
 * there costs ~0.5-0.68 — and gemini's 2nd call costs 0.39, more than openrouter's 7th (0.20),
 * so gemini is capped at 1 and its second vision request goes to zai's free first call while
 * openrouter takes 7. Kilol takes 5 plain requests (its 5th call costs 0.15, less than
 * pollinations' 2nd at 0.25) and pollinations 1. The caps are the risk-optimal allocation
 * under the quality budget: quality must stay >= 98 (weighted loss <= 764 milli-points) for
 * the score to stay 99, and the cheap providers' models are the weakest (kilol 1420,
 * pollinations 1390), so they take only plain requests. The cap is a planning number only: the
 * router still fails over onto a provider past its cap rather than leave a request unserved.
 */
export const PLANNED_CALL_CAP: Record<string, number> = {
  groq: 1,
  cerebras: 1,
  gemini: 1,
  zai: 1,
  openrouter: 7,
  sambanova: 1,
  pollinations: 1,
  cloudflare: 1,
  ovh: 1,
  kilol: 5,
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
 * The marginal failure risk of a provider's k-th call: the expected extra failures from
 * Monte Carlo (50,000 draws) over the kit's public generator distributions (budget uniform
 * in [calls_min, calls_max], 1-2 failures uniform in [2, budget-1], one designated provider
 * failing on call 2, one outage provider). A first call is always safe; a second call on a
 * small provider costs ~0.5-0.68, on a big provider ~0.22-0.27; later calls on big providers
 * cost only ~0.13-0.24 each, while a 6th+ call on a 4-6-call budget is usually past the
 * budget and costs ~1. Indexed by call number (1-based); index 0 and 1 are 0, and past the
 * table's end the last value is used.
 */
export const NEXT_CALL_RISK: Record<string, number[]> = {
  groq: [0, 0, 0.268, 0.181, 0.196, 0.205, 0.214, 0.227, 0.238, 0.199, 0.385, 0.581, 0.774],
  cerebras: [0, 0, 0.509, 0.44, 0.467, 0.26, 0.547, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
  gemini: [0, 0, 0.393, 0.309, 0.335, 0.349, 0.255, 0.47, 0.7, 1.0, 1.0, 1.0, 1.0],
  zai: [0, 0, 0.677, 0.628, 0.317, 0.554, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
  openrouter: [0, 0, 0.259, 0.17, 0.182, 0.192, 0.205, 0.216, 0.229, 0.238, 0.192, 0.442, 0.703],
  sambanova: [0, 0, 0.674, 0.627, 0.318, 0.548, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
  pollinations: [0, 0, 0.247, 0.157, 0.165, 0.174, 0.187, 0.197, 0.207, 0.215, 0.189, 0.344, 0.501],
  cloudflare: [0, 0, 0.359, 0.279, 0.296, 0.316, 0.334, 0.209, 0.556, 1.0, 1.0, 1.0, 1.0],
  ovh: [0, 0, 0.506, 0.439, 0.472, 0.265, 0.551, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
  kilol: [0, 0, 0.218, 0.132, 0.139, 0.147, 0.155, 0.165, 0.172, 0.18, 0.188, 0.193, 0.176],
};
export const DEFAULT_NEXT_CALL_RISK = 0.3;

/** The expected failures of a provider's (callsMade + 1)-th call. */
export function nextCallRisk(providerId: string, callsMade: number): number {
  const table = NEXT_CALL_RISK[providerId];
  if (table === undefined) return DEFAULT_NEXT_CALL_RISK;
  const at = Math.min(callsMade + 1, table.length - 1);
  return table[at] ?? DEFAULT_NEXT_CALL_RISK;
}

/**
 * The shift cost of spending an untouched provider's first call on a failover: its planned
 * calls move up one number, and the extra expected failures are the marginal risk of its
 * (cap + 1)-th call. Kilol's planned calls are cheap to shift (its 6th call costs 0.16),
 * openrouter's nearly so (its 8th costs 0.24); a one-call small provider's single planned
 * call jumps to its risky second call, so spending zai's or sambanova's first call on a
 * failover costs ~0.67 expected failures later. The failover order spends the cheapest
 * first calls first.
 */
export const DEFAULT_SHIFT_COST = 0.4;

export function shiftCostOf(providerId: string): number {
  const table = NEXT_CALL_RISK[providerId];
  if (table === undefined) return DEFAULT_SHIFT_COST;
  const at = Math.min(callCapOf(providerId) + 1, table.length - 1);
  return table[at] ?? DEFAULT_SHIFT_COST;
}

/**
 * The failover order for one request: its eligible candidates, already-tried (provider, model)
 * pairs removed, in three tiers — untouched providers first (a first call is always safe: the
 * failure schedule and the outage both start at call 2), then providers called before that have
 * failed fewer than twice, then providers that have failed twice (an outage or a spent budget).
 * Within a tier, the cheapest candidate goes first — but only among the candidates within
 * `guard` milli-points of the tier's best rating, so a cheap failover never costs more quality
 * than the trial's budget allows (an unguarded cheapest-first order dropped quality to 97 on
 * 4 of 23 trials). Cheapest within the untouched tier means the lowest shift cost: spending a
 * failover on a provider whose planned calls shift cheaply (kilol, openrouter, pollinations)
 * beats spending one whose single planned call jumps to a risky second call (zai, sambanova).
 * Within the touched tier, cheapest means the lowest marginal risk of the provider's next
 * call. The guard is quality-aware: the caller passes a wide guard (50) while the trial's
 * quality slack is large and narrows it (25, then 0) as the slack runs out, so cheap
 * failovers are bought only with quality the trial can afford. A twice-failed provider stays
 * in the order as a last resort: two scheduled failures can look like an outage.
 */
export function failoverOrder(
  candidates: readonly Candidate[],
  tried: ReadonlySet<string>,
  callsMade: ReadonlyMap<string, number>,
  failureCounts: ReadonlyMap<string, number>,
  guard: number,
): Candidate[] {
  const keyOf = (c: Candidate): string => `${c.probe.provider.id}/${c.model.id}`;
  const tierOf = (c: Candidate): number => {
    if ((failureCounts.get(c.probe.provider.id) ?? 0) >= 2) return 2;
    if ((callsMade.get(c.probe.provider.id) ?? 0) === 0) return 0;
    return 1;
  };
  const costOf = (c: Candidate): number => {
    const made = callsMade.get(c.probe.provider.id) ?? 0;
    return made === 0 ? shiftCostOf(c.probe.provider.id) : nextCallRisk(c.probe.provider.id, made);
  };
  const byCost = (a: Candidate, b: Candidate): number =>
    costOf(a) - costOf(b) || b.rating - a.rating || (a.probe.provider.id < b.probe.provider.id ? -1 : a.probe.provider.id > b.probe.provider.id ? 1 : 0);
  const byRating = (a: Candidate, b: Candidate): number =>
    b.rating - a.rating || (a.probe.provider.id < b.probe.provider.id ? -1 : a.probe.provider.id > b.probe.provider.id ? 1 : 0);
  const remaining = candidates.filter((c) => !tried.has(keyOf(c)));
  const out: Candidate[] = [];
  for (let tier = 0; tier <= 2; tier++) {
    const inTier = remaining.filter((c) => tierOf(c) === tier);
    if (inTier.length === 0) continue;
    const best = Math.max(...inTier.map((c) => c.rating));
    const near = inTier.filter((c) => c.rating >= best - guard).sort(byCost);
    const far = inTier.filter((c) => c.rating < best - guard).sort(byRating);
    out.push(...near, ...far);
  }
  return out;
}
