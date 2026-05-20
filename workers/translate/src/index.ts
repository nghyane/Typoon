/**
 * Translate worker — port of typoon/stages/translate.translate_chapter.
 *
 *   POST /translate
 *     body (JSON): {
 *       chapter_id:   number,
 *       source_lang:  string,
 *       target_lang:  string,
 *       brief?:       string,                       // optional context block
 *       pages: [
 *         {
 *           page_index: number,
 *           blocks: [
 *             {
 *               idx:         number,                // reading-order idx (from scan)
 *               bbox:        [x1,y1,x2,y2],
 *               text:        string,                // source bubble text
 *               fit_w?:      number,                // drawable W (px)
 *               fit_h?:      number,                // drawable H (px)
 *               lines_hint?: number,                // estimated lines
 *             }, ...
 *           ]
 *         }
 *       ]
 *     }
 *
 *   Response (JSON): {
 *     translations: [{ key, page_index, block_idx, kind, text }],
 *     windows:      [{ num, keys, char_count, latency_ms, usage }],
 *     missing:      string[],                      // empty if all resolved
 *     timing_ms:    { assign, window_plan, llm_total, retry, normalize, total }
 *   }
 */

import { WorkerEntrypoint } from "cloudflare:workers";

import { assignKey } from "@typoon/shared";
import { isAutoSkip } from "@typoon/shared";
import { normalizeForRender } from "./normalize";
import { pageSystem } from "./prompt";
import {
  makeWindows, partitionWindow, contextWindow, buildWindowPrompt,
  type KeyedBubble,
} from "./window";
import { parseTranslationReply, type ParsedOp } from "./parse";
import { callLLM, LLMFatalError } from "@typoon/shared";
import type { LLMEnv } from "@typoon/shared";
import { loadBrief, briefSlice } from "./brief";

import { getJson, getBytes, putJson, keys as r2keys } from "@typoon/shared";
import { decodeMsgpack } from "@typoon/shared";
import type { ScanPageResult, TranslateResult, TranslationOp } from "@typoon/shared";

interface WindowMeta {
  num:        number;
  keys:       string[];
  char_count: number;
  latency_ms: number;
  usage?:     { input_tokens?: number; output_tokens?: number; total_tokens?: number };
}

interface Env extends LLMEnv {
  R2: R2Bucket;
}

async function translateWindow(args: {
  env:         Env;
  windowNum:   number;
  totalWindows: number;
  briefBlock:  string;          // already rendered from `briefSlice` for these keys
  windowKeys:  string[];
  orderedKeys: KeyedBubble[];
  keyMap:      Map<string, KeyedBubble>;
  sourceLang:  string;
  targetLang:  string;
}): Promise<{ ops: ParsedOp[]; meta: WindowMeta }> {
  const { env, windowNum, totalWindows, briefBlock, windowKeys, orderedKeys, keyMap,
          sourceLang, targetLang } = args;

  const { autoSkipped, active } = partitionWindow(windowKeys, keyMap);
  if (active.size === 0) {
    return {
      ops:  autoSkipped.map(s => ({ ...s, text: "" })),
      meta: { num: windowNum, keys: windowKeys, char_count: 0, latency_ms: 0 },
    };
  }

  const ctxKeys = contextWindow(orderedKeys, active);
  const userPrompt = buildWindowPrompt({
    contextBlock: briefBlock, contextKeys: ctxKeys, active, keyMap,
  });
  const system = pageSystem({ sourceLang, targetLang });

  const result = await callLLM(env, system, userPrompt);
  const ops = parseTranslationReply(result.text, active, keyMap);

  return {
    ops: [...autoSkipped.map(s => ({ ...s, text: "" })), ...ops],
    meta: {
      num:        windowNum,
      keys:       windowKeys,
      char_count: windowKeys.reduce((s, k) => s + (keyMap.get(k)?.source_text.length ?? 0), 0),
      latency_ms: result.latency_ms,
      usage:      result.usage,
    },
  };
}

export class TranslateService extends WorkerEntrypoint<Env> {
  /**
   * Read per-page scan JSONs from R2, run windowed LLM translation,
   * write the per-chapter translation JSON back to R2.
   *
   * The brief is optional — when omitted the prompt has an empty
   * context block and the model relies on bubble proximity only.
   */
  async translateChapter(args: {
    chapter_id:   number;
    scan_keys:    string[];          // one key per page, in page order
    source_lang:  string;
    target_lang:  string;
    /** When false, skip brief loading entirely. Default: true. */
    use_brief?:   boolean;
  }): Promise<{
    output_key:   string;
    translations: number;
    missing:      number;
    errors?:      string[];
  }> {
    if (!this.env.PACKY_API_KEY) throw new Error("PACKY_API_KEY not configured");
    if (!args.scan_keys?.length) throw new Error("scan_keys required");

    const t0 = Date.now();

    // 1. Fetch brief and per-page scans in parallel. The brief loader
    //    returns an empty struct when no brief was produced — the chapter
    //    still translates, just without context hints.
    const useBrief = args.use_brief !== false;
    const [brief, scanResults] = await Promise.all([
      useBrief ? loadBrief(this.env.R2, args.chapter_id) : Promise.resolve(null),
      Promise.all(args.scan_keys.map(k => getBytes(this.env.R2, k).then(b => decodeMsgpack<ScanPageResult>(b)))),
    ]);

    // 2. Assign opaque keys per (chapter, page, group_idx). Reading order
    //    is the grouper's `idx`; we trust it.
    const usedKeys = new Set<string>();
    const orderedKeys: KeyedBubble[] = [];
    const blockLookup = new Map<string, { page_index: number; block_idx: number }>();
    for (const sp of scanResults) {
      // Whole-page noise (e.g. credits page) → translate emits skip ops
      // for every bubble on it without an LLM round trip.
      const isNoisePage = brief?.noise_pages.has(sp.page_index) ?? false;
      const groups = sp.groups.slice().sort((a, b) => a.idx - b.idx);
      for (const g of groups) {
        const key = assignKey(
          { chapter_id: args.chapter_id, page_index: sp.page_index, idx: g.idx },
          usedKeys,
        );
        const w = g.bbox[2] - g.bbox[0];
        const h = g.bbox[3] - g.bbox[1];
        orderedKeys.push({
          key,
          page_index:  sp.page_index,
          idx:         g.idx,
          source_text: isNoisePage ? "" : g.source_text,
          class:       g.class,
          fit_w:       w,
          fit_h:       h,
          lines_hint:  g.typesetting?.line_count ?? 1,
        });
        blockLookup.set(key, { page_index: sp.page_index, block_idx: g.idx });
      }
    }
    const keyMap = new Map(orderedKeys.map(bk => [bk.key, bk]));

    if (orderedKeys.length === 0) {
      const empty: TranslateResult = {
        chapter_id: args.chapter_id, translations: [], missing: [], windows: [],
      };
      const ok = r2keys.translate(String(args.chapter_id));
      await putJson(this.env.R2, ok, empty);
      return { output_key: ok, translations: 0, missing: 0 };
    }

    // 3. Pre-window noise filter + SFX split.
    const briefNoise = brief?.noise_keys ?? new Set<string>();
    const noiseOps = new Map<string, ParsedOp>();
    const sfxBubbles: KeyedBubble[] = [];
    const translatable: KeyedBubble[] = [];
    for (const bk of orderedKeys) {
      if (briefNoise.has(bk.key)) {
        noiseOps.set(bk.key, { key: bk.key, kind: "skip", text: "" });
      } else if (bk.class === "sfx") {
        sfxBubbles.push(bk);
      } else {
        translatable.push(bk);
      }
    }

    // 4. SFX translate — each SFX is independent, no window context needed.
    //    Batch into one window per 20 SFX to amortise LLM round trips.
    const SFX_BATCH = 20;
    const sfxWindows: string[][] = [];
    for (let i = 0; i < sfxBubbles.length; i += SFX_BATCH)
      sfxWindows.push(sfxBubbles.slice(i, i + SFX_BATCH).map(b => b.key));

    // 5. Window pack + fan-out (dialogue + narration only).
    const windows = makeWindows(translatable);
    const settled = await Promise.allSettled([
      // SFX windows — no context block, no neighbours.
      ...sfxWindows.map((wk, i) => translateWindow({
        env: this.env, windowNum: i, totalWindows: sfxWindows.length,
        briefBlock: "(none)", windowKeys: wk, orderedKeys: sfxBubbles, keyMap,
        sourceLang: args.source_lang, targetLang: args.target_lang,
      })),
      // Dialogue/narration windows.
      ...windows.map((wk, i) => {
        const briefBlock = brief ? briefSlice(brief, wk) : "(none)";
        return translateWindow({
          env: this.env, windowNum: i, totalWindows: windows.length,
          briefBlock, windowKeys: wk, orderedKeys: translatable, keyMap,
          sourceLang: args.source_lang, targetLang: args.target_lang,
        });
      }),
    ]);

    // Auth / quota errors on ANY window are fatal. The translate stage
    // re-raises them so the workflow step retries (or surfaces the
    // failure to the operator) instead of writing a half-translated
    // chapter where some windows silently dropped.
    for (const s of settled) {
      if (s.status === "rejected" && s.reason instanceof LLMFatalError) {
        throw s.reason;
      }
    }

    // Fold pre-window noise skip ops into allOps before any retry logic.
    const allOps = new Map<string, ParsedOp>(noiseOps);
    const windowMetas: TranslateResult["windows"] = [];
    const windowErrors: string[] = [];
    const allWindows = [...sfxWindows, ...windows];
    for (let i = 0; i < settled.length; i++) {
      const s = settled[i];
      if (s.status === "fulfilled") {
        for (const op of s.value.ops) allOps.set(op.key, op);
        windowMetas.push(s.value.meta);
      } else {
        windowErrors.push(`window ${i}: ${(s.reason as Error).message}`);
        windowMetas.push({ num: i, keys: allWindows[i], char_count: 0, latency_ms: 0 });
      }
    }

    // 5. Retry pass for missing keys (only if no window-level errors).
    const missing: string[] = [];
    for (const bk of translatable) {
      if (allOps.has(bk.key)) continue;
      if (isAutoSkip(bk.source_text)) continue;
      missing.push(bk.key);
    }
    if (missing.length > 0 && windowErrors.length === 0) {
      try {
        const briefBlock = brief ? briefSlice(brief, missing) : "(none)";
        const r = await translateWindow({
          env: this.env, windowNum: windows.length, totalWindows: windows.length + 1,
          briefBlock, windowKeys: missing, orderedKeys: translatable, keyMap,
          sourceLang: args.source_lang, targetLang: args.target_lang,
        });
        for (const op of r.ops) allOps.set(op.key, op);
        windowMetas.push({ ...r.meta, num: windows.length });
      } catch (e) {
        windowErrors.push(`retry: ${(e as Error).message}`);
      }
    }
    const stillMissing = missing.filter(k => !allOps.has(k));

    // 6. Normalize + materialize. We iterate `orderedKeys` (not just
    //    translatable) so the output JSON carries skip ops for every
    //    noise bubble — renderer needs them to drop the page-counter
    //    bubbles cleanly.
    const translations = [];
    for (const bk of orderedKeys) {
      const loc = blockLookup.get(bk.key);
      if (!loc) continue;
      const op = allOps.get(bk.key);
      if (!op) continue;
      translations.push({
        key:        bk.key,
        page_index: loc.page_index,
        block_idx:  loc.block_idx,
        kind:       op.kind,
        class:      bk.class,
        text:       op.kind === "skip" ? "" : normalizeForRender(op.text),
      });
    }

    const result: TranslateResult = {
      chapter_id:   args.chapter_id,
      translations,
      missing:      stillMissing,
      windows:      windowMetas,
      errors:       windowErrors.length ? windowErrors : undefined,
    };
    const ok = r2keys.translate(String(args.chapter_id));
    await putJson(this.env.R2, ok, result);

    return {
      output_key:   ok,
      translations: translations.length,
      missing:      stillMissing.length,
      errors:       windowErrors.length ? windowErrors : undefined,
    };
  }
}

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    return Response.json({
      ok: true,
      service: "translate",
      rpc: "translateChapter",
      model: env.PACKY_MODEL,
    });
  },
};
