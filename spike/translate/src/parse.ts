/**
 * Reply parsing — port of typoon/stages/page._parse_translation_reply.
 *
 * Output format from the model:
 *
 *   @@ KEY dialogue
 *   translated line 1
 *   translated line 2
 *   @@ KEY2 sfx
 *   RẦM
 *   @@ KEY3 skip
 *
 * Tolerant of preamble/postamble (Codex-like reasoning models sometimes
 * emit `<think>...</think>` or wrap in ```fences```). Drops unknown
 * keys, inactive keys, duplicates, and empty bodies. Auto-skip bubbles
 * are forced to kind="skip" regardless of model emission.
 */

import { isAutoSkip } from "../../shared/src/noise";
import type { KeyedBubble } from "./window";

const HEADER_RE = /^@@ ([A-Z0-9]{7}) (dialogue|sfx|skip)\s*$/i;

export type TranslationKind = "dialogue" | "sfx" | "skip";

export interface TranslationOp {
  key:  string;
  kind: TranslationKind;
  text: string;
}

interface RawBlock { key: string; kind: TranslationKind; text: string; }

function stripEnvelope(text: string): string {
  let s = text;
  const thinkEnd = s.lastIndexOf("</think>");
  if (thinkEnd !== -1) s = s.slice(thinkEnd + "</think>".length);
  s = s.trim();
  if (s.startsWith("```")) {
    const nl = s.indexOf("\n");
    if (nl !== -1) s = s.slice(nl + 1);
    if (s.endsWith("```")) s = s.slice(0, -3);
  }
  return s;
}

function* iterRawBlocks(text: string): Generator<RawBlock> {
  const stripped = stripEnvelope(text);
  let key:  string | null = null;
  let kind: TranslationKind = "dialogue";
  let body: string[] = [];
  for (const line of stripped.split(/\r?\n/)) {
    const m = HEADER_RE.exec(line);
    if (m) {
      if (key !== null) yield { key, kind, text: body.join("\n").trim() };
      key  = m[1];
      kind = m[2].toLowerCase() as TranslationKind;
      body = [];
    } else if (key !== null) {
      body.push(line);
    }
  }
  if (key !== null) yield { key, kind, text: body.join("\n").trim() };
}

export function parseTranslationReply(
  text: string,
  active: Set<string>,
  keyMap: Map<string, KeyedBubble>,
): TranslationOp[] {
  const ops: TranslationOp[] = [];
  const seen = new Set<string>();
  for (const blk of iterRawBlocks(text)) {
    if (!keyMap.has(blk.key) || !active.has(blk.key) || seen.has(blk.key)) continue;
    seen.add(blk.key);
    const bk = keyMap.get(blk.key)!;
    if (isAutoSkip(bk.source_text)) {
      ops.push({ key: blk.key, kind: "skip", text: "" });
      continue;
    }
    if (blk.kind === "skip") {
      ops.push({ key: blk.key, kind: "skip", text: "" });
      continue;
    }
    if (!blk.text) {
      // Empty body for a real bubble — treat as missing so caller retries.
      seen.delete(blk.key);
      continue;
    }
    ops.push({ key: blk.key, kind: blk.kind, text: blk.text });
  }
  return ops;
}
