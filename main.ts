// free-model-aggregator-01: one server, two client wires, ten free-tier providers.
//
// The trial gives this server a brief and an env, and runs it as `node /bot/main.ts` from the
// repository's root, with Node 22's standard library alone. The brief (at $MAKEFUN_PROMPT_FILE)
// carries the provider directory, the quality table and the workload manifest; the env (at
// $MAKEFUN_ENV_URL) serves the ten simulated providers under their base paths. This server:
//
//   1. reads the brief, probes each provider's catalog route in the provider's own wire, and builds
//      the one catalog document (`--list` prints it, the register body carries it, byte for byte);
//   2. plans the workload: each manifest request gets the best-rated free model that meets its
//      requirements, with each provider planned up to its published minimum trial budget (lib/plan.ts);
//   3. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout) and
//      POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//   4. serves the client's requests: GET /v1/models (OpenAI shape for Authorization: Bearer,
//      Anthropic shape for x-api-key), POST /v1/chat/completions and POST /v1/messages, each in
//      JSON or server-sent events, answered by one provider call on that provider's wire — always
//      a non-streaming upstream call, with the client-facing stream synthesised from the answer
//      (lib/wire.ts) — failing over to the next-best eligible model when a provider refuses;
//   5. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}.
//      A process still running when the wall ends scores zero, so never wait for anything else.
//
// Determinism: every choice derives from the brief, the probe answers and the requests seen so far.
// No wall clock, no unseeded randomness, fixed ids and timestamps in every reply, so a replay leaves
// the same state export byte for byte.
//
// Submission note: this file is the entrypoint; lib/providers.ts probes and builds the catalog,
// lib/plan.ts plans the workload and orders failovers, lib/wire.ts translates the six wire crossings.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  isRecord,
  joinBase,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicJson,
  anthropicModelList,
  anthropicSSE,
  anthropicToAnthropic,
  anthropicToGemini,
  anthropicToOpenAI,
  openAIError,
  openAIJson,
  openAIModelList,
  openAISSE,
  openAIToAnthropic,
  openAIToGemini,
  openAIToOpenAI,
  parseAnthropicAnswer,
  parseGeminiAnswer,
  parseOpenAIAnswer,
  type NormalizedAnswer,
  type ServedModel,
} from './lib/wire.ts';
import { candidatesFor, failoverOrder, planAssignments, type Candidate } from './lib/plan.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one upstream inference call may take, in milliseconds. */
const UPSTREAM_TIMEOUT_MS = 5000;
/** The most provider calls one client request tries before it gives up. */
const MAX_ATTEMPTS = 8;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
let plan = new Map<number, Candidate>();
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Inference calls made this trial, by provider id: a provider with none is untouched, and its first call cannot fail. */
const callsMade = new Map<string, number>();
/** Failures seen this trial, by provider id. Two failures deprioritise a provider: one failure is
 * usually a scheduled one-off (every provider has 1-2), but two mean an outage or a spent budget. */
const failureCounts = new Map<string, number>();

/** Whether a provider has failed twice and is deprioritised for the rest of the trial. */
function isDeprioritised(providerId: string): boolean {
  return (failureCounts.get(providerId) ?? 0) >= 2;
}

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider, build the one catalog and plan the workload. */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
  plan = planAssignments(brief, probes);
  log(`planned ${plan.size} of ${brief.workload?.manifest.length ?? 0} requests over ${probes.filter((p) => p.ok).length} providers`);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(index: number, body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[index];
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (!isRecord(block)) continue;
      if (block.type === 'image' || block.type === 'image_url') vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendText(res: ServerResponse, status: number, contentType: string, text: string): void {
  res.writeHead(status, { 'content-type': contentType, 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/**
 * One inference call to one provider, in its own wire. The client wire decides the translation;
 * the upstream call is always non-streaming JSON. Returns the normalized answer, or null with the
 * failure logged when the provider refuses or the answer is not parseable.
 */
async function callUpstream(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json): Promise<NormalizedAnswer | null> {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  let url: string;
  let upstreamBody: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    upstreamBody = clientApi === 'openai' ? openAIToOpenAI(body, candidate.model.id) : anthropicToOpenAI(body, candidate.model.id);
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    upstreamBody = clientApi === 'openai' ? openAIToAnthropic(body, candidate.model.id) : anthropicToAnthropic(body, candidate.model.id);
  } else {
    url = `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:generateContent`;
    upstreamBody = clientApi === 'openai' ? openAIToGemini(body) : anthropicToGemini(body);
  }
  let res: Response;
  try {
    res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(upstreamBody), signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS) });
  } catch (err) {
    log(`upstream ${provider.id}/${candidate.model.id} unreachable: ${(err as Error).message}`);
    return null;
  }
  if (!res.ok) {
    log(`upstream ${provider.id}/${candidate.model.id} answered HTTP ${res.status}`);
    try {
      await res.text();
    } catch {
      // The body is not needed; the status is the failure.
    }
    return null;
  }
  let answer: unknown;
  try {
    answer = await res.json();
  } catch {
    log(`upstream ${provider.id}/${candidate.model.id} answered a body that is not JSON`);
    return null;
  }
  if (!isRecord(answer)) {
    log(`upstream ${provider.id}/${candidate.model.id} answered a JSON value that is not one object`);
    return null;
  }
  if (provider.wire === 'openai') return parseOpenAIAnswer(answer);
  if (provider.wire === 'anthropic') return parseAnthropicAnswer(answer);
  return parseGeminiAnswer(answer);
}

/**
 * Serve one client request: the planned candidate first (when it is still eligible for the live
 * body and its provider has not failed twice), then the failover order — untouched providers
 * first, then providers that have never failed twice, then deprioritised ones. A failed call is
 * never retried on the same provider: a scheduled failure fails only its own call, but two often
 * sit side by side, so a retry can fail twice where a failover to an untouched provider answers
 * at once. Returns the answer and the candidate that served it, or null when no candidate answered.
 */
async function serve(clientApi: 'openai' | 'anthropic', body: Json): Promise<{ answer: NormalizedAnswer; candidate: Candidate } | null> {
  const index = manifestIndex;
  manifestIndex += 1;
  const requires = requirementsFor(index, body);
  const eligible = candidatesFor(probes, brief.ratings, requires, clientApi, body);
  if (eligible.length === 0) return null;

  const tried = new Set<string>();
  const keyOf = (c: Candidate): string => `${c.probe.provider.id}/${c.model.id}`;
  const attempts: Candidate[] = [];
  const planned = plan.get(index);
  if (planned !== undefined && !isDeprioritised(planned.probe.provider.id) && eligible.some((c) => keyOf(c) === keyOf(planned))) attempts.push(planned);
  for (const next of failoverOrder(eligible, tried, callsMade, failureCounts)) {
    if (attempts.some((c) => keyOf(c) === keyOf(next))) continue;
    attempts.push(next);
  }

  for (const candidate of attempts.slice(0, MAX_ATTEMPTS)) {
    tried.add(keyOf(candidate));
    callsMade.set(candidate.probe.provider.id, (callsMade.get(candidate.probe.provider.id) ?? 0) + 1);
    const answer = await callUpstream(candidate, clientApi, body);
    if (answer !== null) return { answer, candidate };
    failureCounts.set(candidate.probe.provider.id, (failureCounts.get(candidate.probe.provider.id) ?? 0) + 1);
  }
  return null;
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by one provider on its own wire. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const served = await serve('openai', body);
  if (served === null) return sendError(res, 'openai', 503, 'no_free_model', 'no free model answered this request');
  const model = served.candidate.model.id;
  if (body.stream === true) sendText(res, 200, 'text/event-stream; charset=utf-8', openAISSE(served.answer, model));
  else sendJson(res, 200, openAIJson(served.answer, model));
}

/** POST /v1/messages: an Anthropic-shaped client, answered by one provider on its own wire. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const served = await serve('anthropic', body);
  if (served === null) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model answered this request');
  const requested = typeof body.model === 'string' && body.model !== '' ? body.model : served.candidate.model.id;
  if (body.stream === true) sendText(res, 200, 'text/event-stream; charset=utf-8', anthropicSSE(served.answer, requested));
  else sendJson(res, 200, anthropicJson(served.answer, requested));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
