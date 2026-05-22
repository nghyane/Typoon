/**
 * Work-level translation context — transport DTO between client and pipeline.
 *
 * Authoritative copy lives in client IndexedDB (per-user). The server
 * only ever sees ephemeral R2 snapshots:
 *
 *   ctx/{job_id}/input.json.gz   ← client uploads at job create time
 *   ctx/{job_id}/output.json.gz  ← brief stage writes merged result
 *
 * Both expire with the rest of the job (7d R2 lifecycle).
 *
 * `version` is a client-bumped monotonic counter for local merge logic
 * — server treats it as opaque metadata.
 */

export interface ContextCharacter {
  /** Canonical source-language name. */
  name:         string;
  /** Localized name in target language. */
  target_name:  string;
  /** Other source-lang spellings seen across chapters. */
  aliases?:     string[];
  gender?:      "male" | "female" | "unknown";
  /** Free-form role, e.g. "protagonist", "antagonist", "side". */
  role?:        string;
  /** One-line voice/style hint for the translator agent. */
  voice?:       string;
}

export interface ContextGlossaryEntry {
  source_term: string;
  target_term: string;
  notes?:      string;
}

export interface ContextAddressPair {
  speaker:  string;
  listener: string;
  /** e.g. "anh/em", "tớ/cậu", "I/you". */
  pair:     string;
}

export interface WorkContext {
  /** Client monotonic counter, bumped on every merge. */
  version:     number;
  source_lang: string;
  target_lang: string;
  characters:  ContextCharacter[];
  glossary:    ContextGlossaryEntry[];
  address:     ContextAddressPair[];
  /** Free-form notes appended to translator system prompt. ≤ 500 chars. */
  style_notes: string;
}

export const EMPTY_WORK_CONTEXT = (
  source_lang: string,
  target_lang: string,
): WorkContext => ({
  version: 0,
  source_lang,
  target_lang,
  characters: [],
  glossary:   [],
  address:    [],
  style_notes: "",
});
