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

import { callLLMVision, LLMFatalError, type LLMEnv } from "../../shared/src/llm";
import { getBytes, putBytes, putJson, keys as r2keys } from "../../shared/src/r2";
import { decodeMsgpack } from "../../shared/src/codec/msgpack";
import { isAutoSkip } from "../../shared/src/noise";
import type { BriefAddressPair, BriefCharacter, BriefIndex } from "../../shared/src/types";

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

export interface BriefChapterArgs {
  chapter_id:      string;
  source_lang:     string;
  target_lang:     string;
  is_color:        boolean;
  strategy:        "one_to_one" | "stitch";
  scan_keys:       string[];
  storyboard_keys: string[];
}

export interface BriefChapterResult {
  index_key:   string;
  chunk_count: number;
  has_prose:   boolean;
  has_chars:   boolean;
  has_address: boolean;
  has_gloss:   boolean;
  has_notes:   boolean;
  has_noise:   boolean;
  noise_count: number;
  noise_pages: number[];
  timing_ms:   Record<string, number>;
}

type Bubble = {
  key: string; page_index: number; text: string;
  bbox: [number,number,number,number]; polygon: [number,number][];
};

export class BriefService extends WorkerEntrypoint<Env> {
  async briefChapter(args: BriefChapterArgs): Promise<BriefChapterResult> {
    if (!this.env.PACKY_API_KEY) throw new Error("PACKY_API_KEY not configured");
    const t0 = Date.now();

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
      return this.persist(args, {
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
    const p1Parts: import("../../shared/src/llm").ContentPart[] = [
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
    return this.persist(args, merged, bubbles, 1, t0, tVision);
  }

  private async persist(
    args:       BriefChapterArgs,
    m:          ChunkResult,
    bubbles:    { key: string; page_index: number }[],
    chunkCount: number,
    t0:         number,
    visionMs:   number,
  ): Promise<BriefChapterResult> {
    const noisePages = args.strategy === "stitch" ? [] : fullNoisePages(bubbles, m.noise);
    const keyNotes   = renderKeyNotes(m.speakers);

    const writes: Promise<void>[] = [];
    if (m.brief_prose) writes.push(
      putBytes(this.env.R2, r2keys.briefProse(args.chapter_id),
        new TextEncoder().encode(m.brief_prose), "text/plain; charset=utf-8"));
    if (m.glossary.size > 0) writes.push(
      putJson(this.env.R2, r2keys.briefGloss(args.chapter_id), Object.fromEntries(m.glossary)));
    if (m.characters.length > 0) writes.push(
      putJson(this.env.R2, r2keys.briefChars(args.chapter_id), m.characters));
    if (m.address.length > 0) writes.push(
      putJson(this.env.R2, r2keys.briefAddress(args.chapter_id), m.address));
    if (Object.keys(keyNotes).length > 0) writes.push(
      putJson(this.env.R2, r2keys.briefNotes(args.chapter_id), keyNotes));
    if (m.noise.size > 0 || noisePages.length > 0) writes.push(
      putJson(this.env.R2, r2keys.briefNoise(args.chapter_id),
        { noise_keys: Array.from(m.noise).sort(), noise_pages: noisePages }));

    const index: BriefIndex = {
      chapter_id: args.chapter_id, version: 1, chunk_count: chunkCount,
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
    writes.push(putJson(this.env.R2, r2keys.brief(args.chapter_id), index));
    await Promise.all(writes);
    index.timing_ms.total_ms = Date.now() - t0;

    return {
      index_key: r2keys.brief(args.chapter_id), chunk_count: chunkCount,
      has_prose: index.has_prose, has_chars: index.has_chars,
      has_address: index.has_address, has_gloss: index.has_gloss,
      has_notes: index.has_notes, has_noise: index.has_noise,
      noise_count: index.noise_count, noise_pages: index.noise_pages,
      timing_ms: index.timing_ms,
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
  // speakers map now contains KEY_NOTES: key → "free text note"
  // Pass through directly as translator hints.
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

export default {
  async fetch(_req: Request, _env: Env): Promise<Response> {
    return Response.json({ ok: true, service: "typoon-brief", rpc: "BriefService.briefChapter" });
  },
};
