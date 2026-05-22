// Typoon v3 API client.
//
// Three concerns:
//
//   1. Auth: Bearer JWT from localStorage. 401 → dispatch `typoon:unauthorized`.
//   2. Jobs: POST/GET/DELETE jobs, presigned upload to R2, WebSocket progress.
//   3. Works: KV-backed WorkContext CRUD.
//
// Everything else lives in IndexedDB (library/history/settings) — see
// `@shared/db`. Source adapters (mangadex/otruyen/...) are unchanged
// (`@features/browse/`) and route through `/cdn/c/<host><path>` for CORS.

import type { WorkContext } from '@shared/db/work-context'

// ── Base URL + auth ─────────────────────────────────────────────────

const API_BASE = window.location.hostname.endsWith('.discordsays.com')
  ? ''
  : (import.meta.env.VITE_PUBLIC_BASE_URL ?? '')

const TOKEN_KEY = 'typoon_token'

export const getToken   = (): string | null => localStorage.getItem(TOKEN_KEY)
export const setToken   = (t: string): void => localStorage.setItem(TOKEN_KEY, t)
export const clearToken = (): void          => localStorage.removeItem(TOKEN_KEY)

function authHeaders(): Record<string, string> {
  const t = getToken()
  return t ? { Authorization: `Bearer ${t}` } : {}
}

export class BackendUnavailableError extends Error {
  constructor() {
    super('Máy chủ tạm thời không phản hồi. Thử lại trong giây lát.')
  }
}

export class ApiError extends Error {
  readonly status: number
  readonly detail?: unknown
  constructor(status: number, message: string, detail?: unknown) {
    super(message)
    this.status = status
    this.detail = detail
    this.name = 'ApiError'
  }
}

async function request<T = unknown>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  let res: Response
  try {
    res = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: {
        ...(init.headers ?? {}),
        ...authHeaders(),
      },
    })
  } catch {
    throw new BackendUnavailableError()
  }

  if (res.status === 401) {
    clearToken()
    window.dispatchEvent(new CustomEvent('typoon:unauthorized'))
    throw new ApiError(401, 'Unauthorized')
  }

  if (!res.ok) {
    let detail: unknown
    let message = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      detail = body
      if (typeof body?.error === 'string') message = body.error
      else if (typeof body?.error?.message === 'string') message = body.error.message
    } catch { /* not JSON */ }
    throw new ApiError(res.status, message, detail)
  }

  if (res.status === 204) return undefined as T
  const ct = res.headers.get('content-type') ?? ''
  if (ct.includes('application/json')) return res.json() as Promise<T>
  return undefined as T
}


// ── Wire types ──────────────────────────────────────────────────────

export interface ApiTierInfo {
  id:                    string
  name:                  string
  monthly_chapters:      number
  max_pages_per_chapter: number
  concurrent_jobs:       number
  sync_quota_bytes:      number
  can_use_api_tokens:    boolean
}

export interface ApiSessionUser {
  id:                    number
  display_name:          string
  avatar_url:            string | null
  email:                 string | null
  is_admin:              boolean
  preferred_target_lang: string | null
  tier:                  ApiTierInfo
}

export type JobState =
  | 'init' | 'uploading' | 'pending' | 'running'
  | 'done' | 'error' | 'expired'

export type JobKind = 'translate' | 'analyze'

export interface ApiJob {
  id:              number
  state:           JobState
  kind:            JobKind
  work_id:         string | null
  source_lang:     string
  target_lang:     string
  progress_stage:  string | null
  progress_index:  number | null
  progress_total:  number | null
  page_count:      number | null
  estimated_pages: number | null
  archive_url:     string | null
  context_out_url: string | null
  context_version: number | null
  error_message:   string | null
  created_at:      string
  started_at:      string | null
  finished_at:     string | null
  expires_at:      string
}

export interface ApiJobInit {
  job_id:           number
  kind:             JobKind
  parts:            { number: number; url: string }[]
  part_size:        number
  expires_in:       number
  context_hydrated: boolean
}

export interface ApiQuota {
  tier:           ApiTierInfo
  used_chapters:  number
  active_jobs:    number
  reset_at:       string
}


// ── Endpoints ───────────────────────────────────────────────────────

export const api = {
  // ── Auth ────────────────────────────────────────────────────────
  authConfig: (): Promise<{ discord_client_id: string }> =>
    request('/api/auth/config'),

  discordExchange: (code: string, redirect_uri: string): Promise<{
    token: string
    user:  ApiSessionUser
  }> =>
    request('/api/auth/discord/exchange', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ code, redirect_uri }),
    }),

  authMe: (): Promise<ApiSessionUser> =>
    request('/api/auth/me'),

  updatePreferences: (body: {
    preferred_target_lang?: string | null
  }): Promise<ApiSessionUser> =>
    request('/api/auth/me/preferences', {
      method:  'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(body),
    }),

  // ── Jobs ────────────────────────────────────────────────────────
  jobsCreate: (body: {
    byte_size:    number
    source_lang:  string
    target_lang?: string
    work_id?:     string
    kind?:        JobKind
  }): Promise<ApiJobInit> =>
    request('/api/jobs', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(body),
    }),

  jobsStart: (id: number, parts: { number: number; etag: string }[]):
    Promise<ApiJob> =>
    request(`/api/jobs/${id}/start`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ parts }),
    }),

  jobsGet: (id: number): Promise<ApiJob> =>
    request(`/api/jobs/${id}`),

  jobsDelete: (id: number): Promise<void> =>
    request(`/api/jobs/${id}`, { method: 'DELETE' }),

  jobsList: (limit = 50): Promise<ApiJob[]> =>
    request(`/api/me/jobs?limit=${limit}`),

  quota: (): Promise<ApiQuota> =>
    request('/api/me/quota'),

  /** Per-user WebSocket URL: one connection per device multiplexes all jobs.
   *  Browsers can't set Authorization on the upgrade, so the JWT goes as a
   *  ?token= query param (server middleware accepts it as fallback). */
  meEventsWsUrl: (): string => {
    const t = getToken()
    const base = API_BASE || window.location.origin
    return `${base.replace(/^http/, 'ws')}/api/me/events?token=${t ?? ''}`
  },

  // ── Works (context KV) ──────────────────────────────────────────

  /** GET /works/:id/context — returns null on 404. Returns parsed JSON
   *  alongside the version header so the caller can pass If-Match-Version
   *  on update. */
  contextGet: async (work_id: string): Promise<{
    context: WorkContext
    version: number
  } | null> => {
    let res: Response
    try {
      res = await fetch(`${API_BASE}/api/works/${encodeURIComponent(work_id)}/context`, {
        headers: authHeaders(),
      })
    } catch {
      throw new BackendUnavailableError()
    }
    if (res.status === 404) return null
    if (res.status === 401) {
      clearToken()
      window.dispatchEvent(new CustomEvent('typoon:unauthorized'))
      throw new ApiError(401, 'Unauthorized')
    }
    if (!res.ok) throw new ApiError(res.status, `${res.status} ${res.statusText}`)
    const buf = await res.arrayBuffer()
    const json = await gunzipToString(new Uint8Array(buf))
    const context = JSON.parse(json) as WorkContext
    const version = Number(res.headers.get('X-Context-Version') ?? '0')
    return { context, version }
  },

  /** PUT /works/:id/context — body is gzipped JSON; If-Match-Version
   *  enforces optimistic concurrency. Pass `base_version` from the last
   *  GET, or `null` to skip the check (last-writer-wins). */
  contextPut: async (
    work_id:      string,
    context:      WorkContext,
    base_version: number | null,
  ): Promise<{ version: number }> => {
    const json  = JSON.stringify(context)
    const gz    = await gzipBytes(new TextEncoder().encode(json))
    const headers: Record<string, string> = {
      ...authHeaders(),
      'Content-Type':     'application/json',
      'Content-Encoding': 'gzip',
    }
    if (base_version !== null) {
      headers['If-Match-Version'] = String(base_version)
    }
    const res = await fetch(`${API_BASE}/api/works/${encodeURIComponent(work_id)}/context`, {
      method:  'PUT',
      headers,
      body:    gz,
    })
    if (res.status === 409) {
      const body = await res.json().catch(() => ({}))
      throw new ApiError(409, 'stale context version', body)
    }
    if (!res.ok) throw new ApiError(res.status, `${res.status} ${res.statusText}`)
    return res.json() as Promise<{ version: number }>
  },

  contextDelete: (work_id: string): Promise<void> =>
    request(`/api/works/${encodeURIComponent(work_id)}/context`, { method: 'DELETE' }),
}


// ── gzip helpers (used by contextGet/Put) ───────────────────────────

async function gzipBytes(bytes: Uint8Array): Promise<Uint8Array> {
  const cs = new CompressionStream('gzip')
  const w  = cs.writable.getWriter()
  void w.write(bytes); void w.close()
  return new Uint8Array(await new Response(cs.readable).arrayBuffer())
}

async function gunzipToString(bytes: Uint8Array): Promise<string> {
  const ds = new DecompressionStream('gzip')
  const w  = ds.writable.getWriter()
  void w.write(bytes); void w.close()
  const buf = await new Response(ds.readable).arrayBuffer()
  return new TextDecoder().decode(buf)
}
