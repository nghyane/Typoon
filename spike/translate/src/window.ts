/**
 * Window assembly — port of stages/translate._make_windows + stages/page.
 *
 * One "window" = the contiguous slice of bubbles we ship to the LLM in
 * a single call. Greedy-packed by source-char budget (3000) so the
 * model output (≈1.5× expansion) stays well under typical max_tokens.
 */

import { isAutoSkip } from "../../shared/src/noise";

import type { BlockClass } from "../../shared/src/types";

export const WINDOW_CHAR_BUDGET = 3000;
export const CONTEXT_SIZE = 20;

export interface KeyedBubble {
  key:         string;
  page_index:  number;
  idx:         number;
  source_text: string;
  class:       BlockClass;
  fit_w?:      number;
  fit_h?:      number;
  lines_hint?: number;
}

/** Greedy pack ordered keys into windows whose total source chars ≤ budget.
 *  SFX are excluded — they bypass the window and translate independently. */
export function makeWindows(ordered: KeyedBubble[]): string[][] {
  const out: string[][] = [];
  let cur: string[] = [];
  let chars = 0;
  for (const bk of ordered) {
    if (bk.class === "sfx") continue;   // SFX bypass
    const n = bk.source_text.length;
    if (cur.length && chars + n > WINDOW_CHAR_BUDGET) {
      out.push(cur);
      cur = [];
      chars = 0;
    }
    cur.push(bk.key);
    chars += n;
  }
  if (cur.length) out.push(cur);
  return out;
}

/** Partition window keys into auto-skip ops and the remaining active set. */
export function partitionWindow(
  windowKeys: string[],
  keyMap: Map<string, KeyedBubble>,
): { autoSkipped: { key: string; kind: "skip" }[]; active: Set<string> } {
  const autoSkipped: { key: string; kind: "skip" }[] = [];
  const active = new Set<string>();
  for (const k of windowKeys) {
    const bk = keyMap.get(k);
    if (!bk) continue;
    if (isAutoSkip(bk.source_text)) autoSkipped.push({ key: k, kind: "skip" });
    else                            active.add(k);
  }
  return { autoSkipped, active };
}

/** ±CONTEXT_SIZE neighbours around the active set, in reading order. */
export function contextWindow(
  ordered: KeyedBubble[], active: Set<string>,
): string[] {
  const allKeys = ordered.map(bk => bk.key);
  const positions: number[] = [];
  for (let i = 0; i < allKeys.length; i++) if (active.has(allKeys[i])) positions.push(i);
  if (positions.length === 0) return [];
  const lo = Math.max(0, positions[0] - CONTEXT_SIZE);
  const hi = Math.min(allKeys.length, positions[positions.length - 1] + CONTEXT_SIZE + 1);
  return allKeys.slice(lo, hi);
}

/**
 * Render the user message for one window. Format mirrors the Python
 * `_build_window_prompt`: `>>> KEY page=N active w=W h=H lines=N\n<text>`.
 *
 * Caller supplies an optional `contextBlock` (brief slice). When the brief
 * worker is wired up, pass its output; the spike accepts plain strings.
 */
export function buildWindowPrompt(args: {
  contextBlock: string;
  contextKeys:  string[];
  active:       Set<string>;
  keyMap:       Map<string, KeyedBubble>;
}): string {
  const { contextBlock, contextKeys, active, keyMap } = args;
  const blocks: string[] = [];
  for (const k of contextKeys) {
    const bk = keyMap.get(k);
    if (!bk) continue;
    const flag = active.has(k) ? " active" : "";
    const dims = bubbleDims(bk);
    blocks.push(`>>> ${k} page=${bk.page_index}${flag}${dims}\n${bk.source_text}`);
  }
  return `${contextBlock}\n\n${blocks.join("\n")}`;
}

function bubbleDims(bk: KeyedBubble): string {
  const w = bk.fit_w ?? 0, h = bk.fit_h ?? 0;
  if (w <= 0 || h <= 0) return "";
  const lines = bk.lines_hint && bk.lines_hint > 0
    ? bk.lines_hint
    : Math.max(1, Math.round(h / 28));   // 28px fallback, matches Python
  return ` w=${w} h=${h} lines=${lines}`;
}
