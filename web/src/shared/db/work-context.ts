// Client-side translation context — mirror of the server WorkContext DTO.
//
// Wire format on the network is gzipped JSON; this is the in-memory shape
// after gunzip + JSON.parse. The authoritative copy lives in Cloudflare KV
// (`ctx:{user_id}:{work_id}`); the client refetches on demand and caches
// only briefly via TanStack Query (never persisted to IndexedDB).

export interface ContextCharacter {
  /** Canonical source-language name. */
  name:         string
  /** Localized name in target language. */
  target_name:  string
  /** Other source-lang spellings seen across chapters. */
  aliases?:     string[]
  gender?:      'male' | 'female' | 'unknown'
  /** Free-form role hint, e.g. "protagonist". */
  role?:        string
  /** One-line voice/style hint for the translator agent. */
  voice?:       string
}

export interface ContextGlossaryEntry {
  source_term: string
  target_term: string
  notes?:      string
}

export interface ContextAddressPair {
  speaker:  string
  listener: string
  pair:     string
}

export interface WorkContext {
  version:     number
  source_lang: string
  target_lang: string
  characters:  ContextCharacter[]
  glossary:    ContextGlossaryEntry[]
  address:     ContextAddressPair[]
  /** Free-form notes appended to translator system prompt. ≤ 2000 chars. */
  style_notes: string
}

export function emptyWorkContext(
  source_lang: string,
  target_lang: string,
): WorkContext {
  return {
    version:     0,
    source_lang,
    target_lang,
    characters:  [],
    glossary:    [],
    address:     [],
    style_notes: '',
  }
}
