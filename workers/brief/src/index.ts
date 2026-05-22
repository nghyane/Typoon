/**
 * Brief worker — 2-phase vision pass.
 *
 * Phase 1 (1 call, all storyboard images):
 *   → CHARACTERS, ADDRESS, GLOSSARY, BRIEF  (chapter-level context)
 *
 * Phase 2 (N calls in parallel, 1 per storyboard chunk):
 *   → SPEAKERS per key, NOISE per key
 *   System prompt injects Phase 1 context so model doesn't re-derive it.
 *
 * Phase 2 calls are parallel; Phase 1 must complete first.
 */

import { WorkerEntrypoint } from "cloudflare:workers";

import { callLLMVision, LLMFatalError } from "@typoon/shared";
import type { LLMEnv } from "@typoon/shared";
import { getBytes, putBytes, putJson, keys as r2keys } from "@typoon/shared";
import { decodeMsgpack } from "@typoon/shared";
import { isAutoSkip } from "@typoon/shared";
import type {
  BriefAddressPair, BriefCharacter, BriefIndex,
  WorkContext,
} from "@typoon/shared";
import { EMPTY_WORK_CONTEXT } from "@typoon/shared";

import { buildBriefPrompt, storyboardSystem } from "./prompt";
import { parseReply, EMPTY_RESULT, type ChunkResult } from "./parse";

interface Env extends LLMEnv {
  R2: R2Bucket;
}

interface ScanGroup {
  idx:         number;
  key:         string;
  page_index:  number;
  source_text: string;
  polygon:     [number, number][];
  bbox:        [number, number, number, number];
  shape_kind:  string;
}

interface ScanPage {
  page_index: number;
  groups:     ScanGroup[];
}

export interface BriefJobArgs {
  job_id:          number;
  source_lang:     string;
  target_lang:     string;
  is_color:        boolean;
  strategy:        "one_to_one" | "stitch";
  scan_keys:       string[];
  storyboard_keys: string[];
  /** Optional R2 key of seed WorkContext (gzip+JSON) from client. */
  context_in_key?: string;
}

export interface BriefJobResult {
  index_key:       string;
  /** R2 key of merged WorkContext (gzip+JSON). Always emitted; clients
   *  download this to refresh their local Work.context. */
  context_out_key: string;
  chunk_count:     number;
  has_prose:       boolean;
  has_chars:       boolean;
  has_address:     boolean;
  has_gloss:       boolean;
  has_notes:       boolean;
  has_noise:       boolean;
  noise_count:     number;
  noise_pages:     number[];
  timing_ms:       Record<string, number>;
}

type Bubble = {
  key: string; page_index: number; text: string;
  bbox: [number,number,number,number]; polygon: [number,number][];
};

export class BriefService extends WorkerEntrypoint<Env> {
  async briefJob(args: BriefJobArgs): Promise<BriefJobResult> {
    if (!this.env.PACKY_API_KEY) throw new Error("PACKY_API_KEY not configured");
    const t0 = Date.now();

    // ── Load seed WorkContext if client supplied one ─────────────────────────
    const seed: WorkContext = args.context_in_key
      ? await loadSeedContext(this.env.R2, args.context_in_key,
                              args.source_lang, args.target_lang)
      : EMPTY_WORK_CONTEXT(args.source_lang, args.target_lang);

    // ── Load scan pages ──────────────────────────────────────────────────────
    const scanBytes = await Promise.all(args.scan_keys.map(k => getBytes(this.env.R2, k)));
    const scanPages = scanBytes.map(b => decodeMsgpack<ScanPage>(b));

    const bubbles: Bubble[] = [];
    for (const sp of scanPages) {
      for (const g of sp.groups.slice().sort((a, b) => a.idx - b.idx)) {
        bubbles.push({ key: g.key, page_index: g.page_index,
                       text: g.source_text, bbox: g.bbox, polygon: g.polygon });
      }
    }
    const validKeys = new Set(bubbles.map(b => b.key));

    // Deterministic noise — no vision needed.
    const deterministicNoise = new Set<string>();
    for (const b of bubbles) if (isAutoSkip(b.text)) deterministicNoise.add(b.key);

    if (deterministicNoise.size === bubbles.length) {
      return this.persist(args, seed, {
        ...EMPTY_RESULT, noise: deterministicNoise,
      }, bubbles, 0, t0, 0);
    }

    // ── Load storyboard images ───────────────────────────────────────────────
    const sbBytes = await Promise.all(args.storyboard_keys.map(k => getBytes(this.env.R2, k)));

    // ── Phase 1: full-chapter context (1 call, all images) ──────────────────
    const tVision0 = Date.now();

    const p1System = storyboardSystem({
      sourceLang: args.source_lang,
      targetLang: args.target_lang,
      isColor:    args.is_color,
    });
    const p1User = buildBriefPrompt({
      sourceLang: args.source_lang,
      targetLang: args.target_lang,
      bubbles:    bubbles.map(b => ({ key: b.key, page_index: b.page_index, text: b.text })),
    });
    const p1Parts: import("@typoon/shared").ContentPart[] = [
      { type: "input_text", text: p1User },
      ...sbBytes.map(b => ({
        type: "input_image" as const,
        image_url: "data:image/jpeg;base64," + bytesToBase64(b),
      })),
    ];

    let ctx: ChunkResult;
    try {
      const resp = await callLLMVision(this.env, p1System, p1Parts);
      ctx = parseReply(resp.text, validKeys);
    } catch (e) {
      if (e instanceof LLMFatalError) throw e;
      ctx = { ...EMPTY_RESULT };
    }

    const tVision = Date.now() - tVision0;

    // SPEAKERS + NOISE are handled downstream by the translate agent.
    const merged = mergeAll(deterministicNoise, ctx, [], validKeys);
    return this.persist(args, seed, merged, bubbles, 1, t0, tVision);
  }

  private async persist(
    args:       BriefJobArgs,
    seed:       WorkContext,
    m:          ChunkResult,
    bubbles:    { key: string; page_index: number }[],
    chunkCount: number,
    t0:         number,
    visionMs:   number,
  ): Promise<BriefJobResult> {
    const noisePages = args.strategy === "stitch" ? [] : fullNoisePages(bubbles, m.noise);
    const keyNotes   = renderKeyNotes(m.speakers);

    const writes: Promise<void>[] = [];
    if (m.brief_prose) writes.push(
      putBytes(this.env.R2, r2keys.briefProse(String(args.job_id)),
        new TextEncoder().encode(m.brief_prose), "text/plain; charset=utf-8"));
    if (m.glossary.size > 0) writes.push(
      putJson(this.env.R2, r2keys.briefGloss(String(args.job_id)), Object.fromEntries(m.glossary)));
    if (m.characters.length > 0) writes.push(
      putJson(this.env.R2, r2keys.briefChars(String(args.job_id)), m.characters));
    if (m.address.length > 0) writes.push(
      putJson(this.env.R2, r2keys.briefAddress(String(args.job_id)), m.address));
    if (Object.keys(keyNotes).length > 0) writes.push(
      putJson(this.env.R2, r2keys.briefNotes(String(args.job_id)), keyNotes));
    if (m.noise.size > 0 || noisePages.length > 0) writes.push(
      putJson(this.env.R2, r2keys.briefNoise(String(args.job_id)),
        { noise_keys: Array.from(m.noise).sort(), noise_pages: noisePages }));

    const index: BriefIndex = {
      job_id: args.job_id, version: 1, chunk_count: chunkCount,
      has_prose:   !!m.brief_prose,
      has_chars:   m.characters.length > 0,
      has_address: m.address.length > 0,
      has_gloss:   m.glossary.size > 0,
      has_notes:   Object.keys(keyNotes).length > 0,
      has_noise:   m.noise.size > 0 || noisePages.length > 0,
      noise_count: m.noise.size,
      noise_pages: noisePages,
      timing_ms:   { total_ms: 0, vision_ms: visionMs },
    };
    writes.push(putJson(this.env.R2, r2keys.brief(String(args.job_id)), index));

    // ── Merged WorkContext output (gzip) ────────────────────────────────────
    const mergedContext = mergeWorkContext(seed, m, args.source_lang, args.target_lang);
    const ctxOutKey     = r2keys.ctxOut(String(args.job_id));
    writes.push(putGzipJson(this.env.R2, ctxOutKey, mergedContext));

    await Promise.all(writes);
    index.timing_ms.total_ms = Date.now() - t0;

    return {
      index_key:       r2keys.brief(String(args.job_id)),
      context_out_key: ctxOutKey,
      chunk_count:     chunkCount,
      has_prose:       index.has_prose,
      has_chars:       index.has_chars,
      has_address:     index.has_address,
      has_gloss:       index.has_gloss,
      has_notes:       index.has_notes,
      has_noise:       index.has_noise,
      noise_count:     index.noise_count,
      noise_pages:     index.noise_pages,
      timing_ms:       index.timing_ms,
    };
  }
}

// ── Merge ──────────────────────────────────────────────────────────────────

function mergeAll(
  deterministicNoise: Set<string>,
  ctx:      ChunkResult,
  p2:       PromiseSettledResult<ChunkResult>[],
  validKeys: Set<string>,
): ChunkResult {
  const out: ChunkResult = {
    characters:  ctx.characters,
    address:     ctx.address,
    glossary:    ctx.glossary,
    brief_prose: ctx.brief_prose,
    style:       [],
    speakers:    new Map(),
    noise:       new Set(deterministicNoise),
  };
  for (const s of p2) {
    if (s.status !== "fulfilled") continue;
    for (const [k, v] of s.value.speakers) if (validKeys.has(k)) out.speakers.set(k, v);
    for (const k of s.value.noise)         if (validKeys.has(k)) out.noise.add(k);
  }
  return out;
}

// ── Helpers ────────────────────────────────────────────────────────────────

function renderKeyNotes(speakers: Map<string, string>): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [k, note] of speakers) {
    if (note) out[k] = note;
  }
  return out;
}

function fullNoisePages(
  bubbles: { page_index: number; key: string }[],
  noise:   Set<string>,
): number[] {
  const byPage = new Map<number, { all: number; noise: number }>();
  for (const b of bubbles) {
    const e = byPage.get(b.page_index) ?? { all: 0, noise: 0 };
    e.all++;
    if (noise.has(b.key)) e.noise++;
    byPage.set(b.page_index, e);
  }
  return [...byPage.entries()]
    .filter(([, e]) => e.all > 0 && e.noise === e.all)
    .map(([pi]) => pi).sort((a, b) => a - b);
}

function bytesToBase64(bytes: Uint8Array): string {
  let s = ""; const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK)
    s += String.fromCharCode(...bytes.subarray(i, Math.min(i + CHUNK, bytes.length)));
  return btoa(s);
}

// ── WorkContext I/O ────────────────────────────────────────────────────────

async function loadSeedContext(
  r2:          R2Bucket,
  key:         string,
  source_lang: string,
  target_lang: string,
): Promise<WorkContext> {
  const obj = await r2.get(key);
  if (!obj) return EMPTY_WORK_CONTEXT(source_lang, target_lang);
  try {
    const gz   = await obj.arrayBuffer();
    const json = await gunzipToString(new Uint8Array(gz));
    const parsed = JSON.parse(json) as Partial<WorkContext>;
    return {
      version:     parsed.version     ?? 0,
      source_lang: parsed.source_lang ?? source_lang,
      target_lang: parsed.target_lang ?? target_lang,
      characters:  parsed.characters  ?? [],
      glossary:    parsed.glossary    ?? [],
      address:     parsed.address     ?? [],
      style_notes: parsed.style_notes ?? "",
    };
  } catch (e) {
    console.warn("brief: failed to load seed context", key, e);
    return EMPTY_WORK_CONTEXT(source_lang, target_lang);
  }
}

/** Merge brief output (ChunkResult) into the seed WorkContext.
 *  Strategy: union by canonical key (character.name, glossary.source_term,
 *  address pair). New entries from this chapter are appended; existing
 *  entries are updated in-place (last writer wins on field-by-field). */
function mergeWorkContext(
  seed: WorkContext,
  m:    ChunkResult,
  source_lang: string,
  target_lang: string,
): WorkContext {
  const out: WorkContext = {
    ...seed,
    source_lang,
    target_lang,
    version: seed.version + 1,
  };

  // Characters — keyed by `name`
  const charMap = new Map(out.characters.map(c => [c.name, c]));
  for (const c of m.characters) {
    const prev = charMap.get(c.name);
    charMap.set(c.name, {
      name:        c.name,
      target_name: c.target_name || prev?.target_name || c.name,
      gender:      c.gender ?? prev?.gender,
      role:        c.role ?? prev?.role,
      voice:       c.voice ?? prev?.voice,
      aliases:     prev?.aliases,
    });
  }
  out.characters = [...charMap.values()];

  // Glossary — keyed by `source_term`
  const glossMap = new Map(out.glossary.map(g => [g.source_term, g]));
  for (const [src, tgt] of m.glossary) {
    const prev = glossMap.get(src);
    glossMap.set(src, {
      source_term: src,
      target_term: tgt,
      notes:       prev?.notes,
    });
  }
  out.glossary = [...glossMap.values()];

  // Address pairs — keyed by `${speaker}→${listener}`
  const addrMap = new Map(out.address.map(a => [`${a.speaker}->${a.listener}`, a]));
  for (const a of m.address) {
    addrMap.set(`${a.speaker}->${a.listener}`, {
      speaker:  a.speaker,
      listener: a.listener,
      pair:     a.pair,
    });
  }
  out.address = [...addrMap.values()];

  // style_notes: append brief_prose if present and not already covered
  if (m.brief_prose && !out.style_notes.includes(m.brief_prose.slice(0, 40))) {
    const next = out.style_notes
      ? out.style_notes + "\n" + m.brief_prose
      : m.brief_prose;
    out.style_notes = next.slice(0, 2000);  // sanity cap
  }

  return out;
}

async function putGzipJson(r2: R2Bucket, key: string, value: unknown): Promise<void> {
  const json  = JSON.stringify(value);
  const bytes = new TextEncoder().encode(json);
  const gz    = await gzipBytes(bytes);
  await r2.put(key, gz, {
    httpMetadata: { contentType: "application/json", contentEncoding: "gzip" },
  });
}

async function gzipBytes(bytes: Uint8Array): Promise<Uint8Array> {
  const cs = new CompressionStream("gzip");
  const w  = cs.writable.getWriter();
  void w.write(bytes); void w.close();
  return new Uint8Array(await new Response(cs.readable).arrayBuffer());
}

async function gunzipToString(bytes: Uint8Array): Promise<string> {
  const ds = new DecompressionStream("gzip");
  const w  = ds.writable.getWriter();
  void w.write(bytes); void w.close();
  const buf = await new Response(ds.readable).arrayBuffer();
  return new TextDecoder().decode(buf);
}

export default {
  async fetch(_req: Request, _env: Env): Promise<Response> {
    return Response.json({ ok: true, service: "typoon-brief", rpc: "BriefService.briefJob" });
  },
};
