/**
 * Typeset Worker — follows CF docs WASM pattern:
 *   https://developers.cloudflare.com/workers/wrangler/bundling/
 *
 * Wrangler resolves `.wasm` imports to WebAssembly.Module at runtime.
 * No custom build script — wrangler deploy handles everything.
 */

import { DurableObject, WorkerEntrypoint } from "cloudflare:workers";

// ── WASM imports — Wrangler resolves these to WebAssembly.Module ──────────
// Render core (wasm-bindgen --target web)
import renderWasm from "./typoon_render_bg.wasm";
import { initSync, render_page, version } from "./typoon_render.js";

// PNG codec WASM (squoosh OxiPNG)
import pngWasm from "../../node_modules/@jsquash/png/codec/pkg/squoosh_png_bg.wasm";

// ── Top-level init — runs once per isolate at module evaluation ───────────
// Per CF docs: instantiate at top level, not inside fetch handler.
// initSync calls new WebAssembly.Instance(module, imports) synchronously.
initSync({ module: renderWasm });

// Set PNG WASM global that shared codec uses via requireWasm("__PNG_WASM__")
(globalThis as any).__PNG_WASM__ = pngWasm;

// Now safe to import codecs — their top-level init() will find __PNG_WASM__
import { decodePng } from "../../shared/src/codec/png-dec";
import { encodePng } from "../../shared/src/codec/png-enc";
import { getBytes, getJson, putBytes, keys as K } from "../../shared/src/r2";
import { decodeMsgpack } from "../../shared/src/codec/msgpack";
import type { ScanPageResult, TranslateResult, TypesettingHint } from "../../shared/src/types";

// ── Types ─────────────────────────────────────────────────────────────────

interface Env {
  R2:         R2Bucket;
  FONT_KEY:   string;
  TYPESET_DO: DurableObjectNamespace<TypesetRender>;
}

export interface TypesetPageArgs {
  chapter_id:    string;
  page_index:    number;
  inpaint_key:   string;
  scan_key:      string;
  translate_key: string;
  page_width:    number;
  output_key?:   string;
}

export interface TypesetPageResult {
  output_key:  string;
  size_bytes:  number;
  width:       number;
  height:      number;
  timings_ms:  Record<string, number>;
  bubbles:     { font_size_px: number; line_height: number; overflow: boolean; rect: number[] }[];
}

// ── Durable Object ────────────────────────────────────────────────────────

export class TypesetRender extends DurableObject<Env> {
  private warmTimings: Record<string, number> = {};
  private initialized = false;

  private async ensureReady(): Promise<void> {
    if (this.initialized) return;
    const t0 = Date.now();
    try {
      const obj = await this.env.R2.head(this.env.FONT_KEY);
      this.warmTimings.font_present = obj ? 1 : 0;
    } catch { /* font embedded in WASM, non-fatal */ }
    this.warmTimings = {
      ...this.warmTimings,
      wasm_init_ms:  0,
      font_head_ms:  Date.now() - t0,
      total_init_ms: Date.now() - t0,
      version_sig:   hashStr(version()),
    };
    this.initialized = true;
  }

  async typesetPage(args: TypesetPageArgs): Promise<TypesetPageResult> {
    const t0 = Date.now();
    await this.ensureReady();
    const tWarm = Date.now() - t0;

    const tFetch0 = Date.now();
    const [cleanBytes, scanRaw, translateJson] = await Promise.all([
      getBytes(this.env.R2, args.inpaint_key),
      getBytes(this.env.R2, args.scan_key),
      getJson<TranslateResult>(this.env.R2, args.translate_key),
    ]);
    const scanJson = decodeMsgpack<ScanPageResult>(scanRaw);
    const tFetch = Date.now() - tFetch0;

    const tDec0 = Date.now();
    const cleanImg = await decodePng(cleanBytes);
    const tDec = Date.now() - tDec0;

    const translationByIdx = new Map<number, { text: string; kind: string }>();
    for (const op of translateJson.translations) {
      if (op.page_index !== args.page_index) continue;
      translationByIdx.set(op.block_idx, { text: op.text, kind: op.kind });
    }

    const polygons: [number, number][][] = [];
    const texts:    string[]             = [];
    const hints:    (TypesettingHint | null)[] = [];

    for (const g of scanJson.groups) {
      const op = translationByIdx.get(g.idx);
      if (!op || op.kind === "skip" || !op.text.trim()) continue;
      polygons.push(g.polygon);
      texts.push(op.text);
      hints.push(g.typesetting);
    }

    const cleanRgba = new Uint8Array(cleanImg.data.buffer, cleanImg.data.byteOffset, cleanImg.data.byteLength);
    const tRender0 = Date.now();
    const out = render_page(
      { width: cleanImg.width, height: cleanImg.height, polygons, texts, page_width: args.page_width, hints },
      cleanRgba,
    );
    const tRender = Date.now() - tRender0;

    const tEnc0 = Date.now();
    const rgba  = new Uint8ClampedArray(out.rgba.buffer, out.rgba.byteOffset, out.rgba.byteLength);
    const png   = await encodePng(rgba, out.width, out.height);
    const tEnc  = Date.now() - tEnc0;

    const outKey = args.output_key ?? K.typeset(args.chapter_id, args.page_index);
    await putBytes(this.env.R2, outKey, png, "image/png");

    return {
      output_key: outKey,
      size_bytes: png.byteLength,
      width:      out.width,
      height:     out.height,
      bubbles:    out.bubbles,
      timings_ms: {
        warm_ms:   tWarm,
        fetch_ms:  tFetch,
        decode_ms: tDec,
        render_ms: tRender,
        encode_ms: tEnc,
        total_ms:  Date.now() - t0,
        ...this.warmTimings,
      },
    };
  }

  async warm(): Promise<{ ok: true; cached: boolean; timings: Record<string, number> }> {
    const cached = this.initialized;
    await this.ensureReady();
    return { ok: true, cached, timings: this.warmTimings };
  }
}

// ── Service entrypoint ────────────────────────────────────────────────────

export class TypesetService extends WorkerEntrypoint<Env> {
  async typesetPage(args: TypesetPageArgs): Promise<TypesetPageResult> {
    const id = this.env.TYPESET_DO.idFromName(args.chapter_id);
    return this.env.TYPESET_DO.get(id).typesetPage(args);
  }
  async warm(chapter_id: string) {
    const id = this.env.TYPESET_DO.idFromName(chapter_id);
    return this.env.TYPESET_DO.get(id).warm();
  }
}

function hashStr(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0;
  return h >>> 0;
}

export default {
  async fetch(): Promise<Response> {
    return Response.json({ ok: true, service: "typeset", rpc: "typesetPage|warm" });
  },
};
