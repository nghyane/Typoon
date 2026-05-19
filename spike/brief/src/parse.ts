/**
 * Reply parsing for brief vision agent (two phases).
 *
 * Phase 1 (full-chapter context call) emits:
 *   @@@ CHARACTERS, @@@ ADDRESS, @@@ GLOSSARY, @@@ BRIEF
 *
 * Phase 2 (per-chunk speaker call) emits:
 *   @@@ SPEAKERS, @@@ NOISE
 *
 * Both use the same section/line format. Unknown sections are ignored.
 * Keys not in `validKeys` are dropped to prevent model hallucinations.
 */

import type { BriefCharacter, BriefAddressPair } from "../../shared/src/types";

const SECTION_RE    = /^@@@\s+(\w+)\s*$/;
const LINE_RE       = /^@@\s+(.*)$/;
const PROSE_SECTIONS = new Set(["BRIEF"]);

export interface ChunkResult {
  characters:  BriefCharacter[];
  speakers:    Map<string, string>;   // key → speaker note (from KEY_NOTES)
  noise:       Set<string>;
  style:       string[];
  glossary:    Map<string, string>;
  address:     BriefAddressPair[];
  brief_prose: string;
}

export const EMPTY_RESULT: ChunkResult = {
  characters: [], speakers: new Map(), noise: new Set(),
  style: [], glossary: new Map(), address: [], brief_prose: "",
};

export function parseReply(text: string, validKeys: Set<string>): ChunkResult {
  const sections = splitSections(stripThink(text));
  return {
    characters:  parseCharacters(sections.get("CHARACTERS")  ?? []),
    speakers:    parseKeyNotes(sections.get("KEY_NOTES")     ?? [], validKeys),
    noise:       new Set(),           // brief no longer emits NOISE — handled by isAutoSkip() + translate inline
    style:       [],
    glossary:    parseGlossary(sections.get("GLOSSARY")      ?? []),
    address:     parseAddress(sections.get("ADDRESS")        ?? []),
    brief_prose: parseBrief(sections.get("BRIEF")            ?? []),
  };
}

function stripThink(text: string): string {
  const idx = text.lastIndexOf("</think>");
  return idx === -1 ? text : text.slice(idx + "</think>".length);
}

function splitSections(text: string): Map<string, string[]> {
  const out = new Map<string, string[]>();
  let cur: string | null = null;
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.replace(/\s+$/, "");
    const sm = SECTION_RE.exec(line);
    if (sm) {
      cur = sm[1].toUpperCase();
      if (!out.has(cur)) out.set(cur, []);
      continue;
    }
    if (cur === null) continue;
    if (PROSE_SECTIONS.has(cur)) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      const lm = LINE_RE.exec(line);
      out.get(cur)!.push(lm ? lm[1].trim() : trimmed);
    } else {
      const lm = LINE_RE.exec(line);
      if (lm) out.get(cur)!.push(lm[1].trim());
    }
  }
  return out;
}

function parseKv(line: string): Record<string, string> {
  const out: Record<string, string> = {};
  const re = /(\w+)=("([^"\\]*(?:\\.[^"\\]*)*)"|(\S+))/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(line)) !== null) {
    const k = m[1];
    const v = m[3] !== undefined ? m[3].replace(/\\"/g, '"') : (m[4] ?? "");
    out[k] = v;
  }
  return out;
}

function parseCharacters(bodies: string[]): BriefCharacter[] {
  const out: BriefCharacter[] = [];
  for (const body of bodies) {
    const kv = parseKv(body);
    const name = (kv.name ?? "").trim();
    if (!name) continue;
    const gender = (kv.gender ?? "").trim().toLowerCase();
    out.push({
      name,
      target_name: (kv.target ?? "").trim(),
      gender: (gender === "male" || gender === "female") ? gender : "unknown",
      role:  (kv.role  ?? "").trim(),
      voice: (kv.voice ?? "").trim(),
    });
  }
  return out;
}

function parseKeyNotes(bodies: string[], validKeys: Set<string>): Map<string, string> {
  // @@ KEY note="free text"
  const out = new Map<string, string>();
  for (const body of bodies) {
    const keyMatch = /^(\S+)/.exec(body);
    if (!keyMatch) continue;
    const key = keyMatch[1];
    if (!validKeys.has(key)) continue;
    const noteMatch = /note="([^"]*)"/.exec(body) ?? /note=(\S+)/.exec(body);
    const note = noteMatch ? noteMatch[1].trim() : body.slice(key.length).trim();
    if (note) out.set(key, note);
  }
  return out;
}

function parseNoise(bodies: string[], validKeys: Set<string>): Set<string> {
  const out = new Set<string>();
  for (const body of bodies) {
    const key = body.split(/\s+/, 1)[0];
    if (key && validKeys.has(key)) out.add(key);
  }
  return out;
}

function parseGlossary(bodies: string[]): Map<string, string> {
  const out = new Map<string, string>();
  for (const body of bodies) {
    const idx = body.indexOf("=");
    if (idx <= 0) continue;
    const src = body.slice(0, idx).trim();
    const tgt = body.slice(idx + 1).trim();
    if (!src || !tgt || src === tgt) continue;
    out.set(src, tgt);
  }
  return out;
}

function parseAddress(bodies: string[]): BriefAddressPair[] {
  const out: BriefAddressPair[] = [];
  for (const body of bodies) {
    const arrow = body.includes("→") ? "→" : "->";
    const colon = body.lastIndexOf(":");
    if (colon === -1) continue;
    const lhs  = body.slice(0, colon);
    const pair = body.slice(colon + 1).trim();
    const ai   = lhs.indexOf(arrow);
    if (ai === -1) continue;
    const speaker  = lhs.slice(0, ai).trim();
    const listener = lhs.slice(ai + arrow.length).trim();
    if (!speaker || !listener || !pair) continue;
    if (speaker.toLowerCase() === "unknown" || listener.toLowerCase() === "unknown") continue;
    out.push({ speaker, listener, pair });
  }
  return out;
}

function parseBrief(bodies: string[]): string {
  return bodies.join("\n").trim();
}
