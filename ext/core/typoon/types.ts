// Engine API types. Copied verbatim from web/src/shared/api/api.ts so the
// extension uses the exact same shapes the SPA does. Drift is policed by
// hand at PR time — when the engine adds a field, sync both copies.
//
// Only the subset the extension actually consumes is mirrored: projects,
// chapters, tokens, quota. Bubbles/glossary/search live in the web app.

export type ChapterState = 'idle' | 'pending' | 'running' | 'error' | 'done'

export interface ApiProject {
  project_id:   number
  slug:         string
  title:        string
  description:  string | null
  cover_url:    string | null
  source_lang:  string
  target_lang:  string
  source_url:   string | null
  owner_id:     number | null
  shared:       boolean
  is_owner:     boolean
  is_pinned:    boolean
  created_at:   string | null
  updated_at:   string | null
}

/** /api/me/projects — slim shape for tools (no owner flags, no timestamps). */
export interface ApiMeProject {
  project_id:  number
  slug:        string
  title:       string
  cover_url:   string | null
  source_lang: string
  target_lang: string
  shared:      boolean
}

export interface ApiChapter {
  chapter_id: number
  project_id: number
  /** Display chapter number, free-form: "4", "4.5", "Extra". */
  number:     string
  position:   number
  title:      string | null
  state:      ChapterState
  stage:      string
  page_count: number
  error:      string
  updated_at: string | null
  progress: {
    stage:      string
    page_index: number
    page_total: number
  } | null
  archive_url: string | null
}

export interface ApiTokenInfo {
  id:         number
  name:       string
  prefix:     string
  last_used:  string | null
  created_at: string | null
}

export interface ApiMe {
  user_id:      number
  display_name: string
  avatar_url:   string | null
  is_admin:     boolean
}

/** Inputs for creating a project from the extension's "+ create" flow. */
export interface CreateProjectOpts {
  title:       string
  source_lang: string
  target_lang: string
  description?: string
}

// Multipart inbox shapes are imported from `@typoon/upload-sdk`; nothing
// engine-specific lives here anymore.
