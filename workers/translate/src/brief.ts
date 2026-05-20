/**
 * Brief context loader + slicer.
 *
 * Reads the brief artefacts produced by brief-worker. The translate
 * stage uses two facets of the brief:
 *
 *   - `noise_keys` / `noise_pages` to skip bubbles BEFORE windowing
 *     (saves both LLM input and output tokens).
 *   - everything else → free-form prose injected into the user prompt,
 *     mirroring `typoon/stages/brief.brief_slice`.
 *
 * `BriefIndex` is the only fixed key the caller needs; everything else
 * is loaded on demand based on which `has_*` flags are set. Skipping
 * absent files avoids paying R2 GET ops for empty objects.
 */

import { getBytes, getJson, keys as r2keys } from "@typoon/shared";
import type {
  BriefIndex, BriefCharacter, BriefAddressPair,
} from "@typoon/shared";

export interface LoadedBrief {
  index:        BriefIndex;
  prose:        string;
  glossary:     Map<string, string>;
  characters:   BriefCharacter[];
  address:      BriefAddressPair[];
  key_notes:    Record<string, string>;
  noise_keys:   Set<string>;
  noise_pages:  Set<number>;
}

const EMPTY_BRIEF: LoadedBrief = {
  index: {
    chapter_id: 0, version: 1, chunk_count: 0,
    has_prose: false, has_chars: false, has_address: false,
    has_gloss: false, has_notes: false, has_noise: false,
    noise_count: 0, noise_pages: [], timing_ms: {},
  },
  prose: "",
  glossary:    new Map(),
  characters:  [],
  address:     [],
  key_notes:   {},
  noise_keys:  new Set(),
  noise_pages: new Set(),
};

/**
 * Load the brief for a chapter from R2. Returns `EMPTY_BRIEF` when no
 * brief was produced (brief stage was skipped or failed silently).
 *
 * Errors on the index read propagate — that means brief-worker
 * promised an output, the workflow assumed it exists, and a real
 * problem prevented it (storage outage, eventual-consistency miss).
 * Per-facet read errors degrade silently: better translation with
 * partial context than no translation at all.
 */
export async function loadBrief(
  r2: R2Bucket, chapter_id: number,
): Promise<LoadedBrief> {
  let index: BriefIndex;
  try {
    index = await getJson<BriefIndex>(r2, r2keys.brief(String(chapter_id)));
  } catch {
    return EMPTY_BRIEF;     // no brief written → ChapterBrief() equivalent
  }

  const tasks: Promise<unknown>[] = [];
  let prose = "";
  let glossary    = new Map<string, string>();
  let characters: BriefCharacter[]    = [];
  let address:    BriefAddressPair[]  = [];
  let key_notes:  Record<string, string> = {};
  const noise_keys  = new Set<string>();
  const noise_pages = new Set(index.noise_pages);

  if (index.has_prose) {
    tasks.push(getBytes(r2, r2keys.briefProse(String(chapter_id)))
      .then(b => { prose = new TextDecoder().decode(b); })
      .catch(() => { /* leave empty */ }));
  }
  if (index.has_gloss) {
    tasks.push(getJson<Record<string, string>>(r2, r2keys.briefGloss(String(chapter_id)))
      .then(o => { glossary = new Map(Object.entries(o)); })
      .catch(() => {}));
  }
  if (index.has_chars) {
    tasks.push(getJson<BriefCharacter[]>(r2, r2keys.briefChars(String(chapter_id)))
      .then(o => { characters = o; })
      .catch(() => {}));
  }
  if (index.has_address) {
    tasks.push(getJson<BriefAddressPair[]>(r2, r2keys.briefAddress(String(chapter_id)))
      .then(o => { address = o; })
      .catch(() => {}));
  }
  if (index.has_notes) {
    tasks.push(getJson<Record<string, string>>(r2, r2keys.briefNotes(String(chapter_id)))
      .then(o => { key_notes = o; })
      .catch(() => {}));
  }
  if (index.has_noise) {
    tasks.push(getJson<{ noise_keys: string[]; noise_pages: number[] }>(
      r2, r2keys.briefNoise(String(chapter_id)),
    )
      .then(o => {
        for (const k of o.noise_keys) noise_keys.add(k);
        for (const p of o.noise_pages) noise_pages.add(p);
      })
      .catch(() => {}));
  }
  await Promise.all(tasks);

  return { index, prose, glossary, characters, address, key_notes,
           noise_keys, noise_pages };
}

/**
 * Render the brief subset relevant to one translation window as a
 * single text block. Order matches `typoon/stages/brief.brief_slice`:
 * prose first, glossary, address, characters, then per-bubble notes.
 *
 * `windowKeys` filters key_notes to the bubbles actually visible to
 * the LLM in this call — every other speaker hint is irrelevant noise.
 */
export function briefSlice(
  brief: LoadedBrief,
  windowKeys: string[],
): string {
  const parts: string[] = [];

  if (brief.prose) {
    parts.push(`## Chapter brief\n${brief.prose}`);
  }

  if (brief.characters.length > 0) {
    const charLines = brief.characters.map(c => {
      const display = c.target_name || c.name;
      const gender  = c.gender && c.gender !== "unknown" ? c.gender : "?";
      let line = `- ${c.name} → ${display} (${gender})`;
      if (c.role)  line += `: ${c.role}`;
      if (c.voice) line += ` | voice: ${c.voice}`;
      return line;
    });
    parts.push("## Characters\n" + charLines.join("\n"));
  }

  if (brief.address.length > 0) {
    const addrLines = brief.address.map(
      a => `- ${a.speaker} → ${a.listener}: ${a.pair}  [BINDING]`
    );
    parts.push("## Address (xưng hô — BINDING for these pairs)\n" + addrLines.join("\n"));
  }

  if (brief.glossary.size > 0) {
    parts.push("## Glossary\n" + Array.from(brief.glossary, ([s, t]) => `- ${s} → ${t}`).join("\n"));
  }

  // KEY_NOTES: only the subset relevant to this window
  const relevant: string[] = [];
  for (const k of windowKeys) {
    const note = brief.key_notes[k];
    if (note) relevant.push(`- ${k}: ${note}`);
  }
  if (relevant.length > 0) {
    parts.push("## Bubble notes (vision-confirmed context)\n" + relevant.join("\n"));
  }

  return parts.length > 0 ? parts.join("\n\n") : "(none)";
}
