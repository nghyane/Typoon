// Typoon engine API client.
//
// Pure web-API surface (`fetch`, `Blob`) so the same client runs in the
// popup, the service worker, the offscreen document, and content scripts
// unchanged. No `chrome.*`, no React, no DOM.
//
// Auth is `Authorization: Bearer <token>` against an API token issued
// from /api/me/tokens (RFC-008). Cookies are not used — the extension
// origin (`chrome-extension://<id>`) cannot share cookies with the
// engine domain anyway.

import {
  BackendUnavailableError,
  QuotaExceededError,
  UnauthorizedError,
} from './errors'
import type {
  ApiChapter, ApiMe, ApiMeProject, ApiProject,
  ApiTokenInfo, CreateProjectOpts,
} from './types'
import type {
  UploadInitBody, UploadInitOut,
  UploadFinalizeBody, UploadAbortBody,
  ApiChapterLike, UploadHttpClient,
} from '@typoon/upload-sdk'

export interface TypoonClientOpts {
  /** Engine origin, e.g. `https://typoon.example.com` (no trailing slash). */
  apiUrl: string
  /** API token (`typ_…`) issued by the engine. */
  token:  string
  /** Inject a fetch implementation — useful in tests / non-browser hosts. */
  fetch?: typeof fetch
}

export class TypoonClient implements UploadHttpClient {
  private readonly base:  string
  private readonly token: string
  private readonly _fetch: typeof fetch

  constructor(opts: TypoonClientOpts) {
    // Tolerate trailing slash and `/api` suffix in user/build-time URLs.
    // The transport always appends `/api${path}` itself.
    this.base   = opts.apiUrl.replace(/\/+$/, '').replace(/\/api$/, '')
    this.token  = opts.token
    this._fetch = opts.fetch ?? globalThis.fetch.bind(globalThis)
  }

  // ── Auth verify ────────────────────────────────────────────────────

  /** Lightweight ping used by the Setup view to validate URL + token in
   *  one round-trip. Returns `{ ok: true, profile }` only when the
   *  engine is reachable AND the token is accepted. */
  async verify(): Promise<
    | { ok: true; profile: ApiMe }
    | { ok: false; reason: 'unauthorized' | 'unreachable'; detail?: string }
  > {
    try {
      const profile = await this.get<ApiMe>('/auth/me')
      return { ok: true, profile }
    } catch (e) {
      if (e instanceof UnauthorizedError) return { ok: false, reason: 'unauthorized' }
      const detail = (e as Error)?.message
      console.error('[typoon] verify failed:', detail)
      return { ok: false, reason: 'unreachable', detail }
    }
  }

  /** GET /api/auth/me — user profile. */
  me(): Promise<ApiMe> {
    return this.get<ApiMe>('/auth/me')
  }

  // ── Multipart chapter upload (shared with web SPA) ─────────────────
  //
  // Implements `UploadHttpClient` from `@typoon/upload-sdk` so the SDK's
  // `uploadChapterZip` driver works against this client unchanged. The
  // SDK owns the zip pack + part PUT loop; this class only handles the
  // three engine round-trips.

  uploadInit(projectId: number, body: UploadInitBody): Promise<UploadInitOut> {
    return this.post<UploadInitOut>(
      `/projects/${projectId}/chapters/upload-init`,
      body,
    )
  }

  uploadFinalize(projectId: number, body: UploadFinalizeBody): Promise<ApiChapterLike> {
    return this.post<ApiChapter>(
      `/projects/${projectId}/chapters/upload-finalize`,
      body,
    )
  }

  uploadAbort(projectId: number, body: UploadAbortBody): Promise<void> {
    return this.post<void>(
      `/projects/${projectId}/chapters/upload-abort`,
      body,
    )
  }

  // ── Projects ───────────────────────────────────────────────────────

  /** User's own + pinned projects. Slim shape (no flags/timestamps). */
  myProjects(): Promise<ApiMeProject[]> {
    return this.get<ApiMeProject[]>('/me/projects')
  }

  createProject(opts: CreateProjectOpts): Promise<ApiProject> {
    return this.post<ApiProject>('/projects', opts)
  }

  // ── Chapters ───────────────────────────────────────────────────────

  /** Existing chapters in a project — used for client-side dedupe before
   *  upload (warn the user when `number` is already present). */
  listChapters(projectId: number): Promise<ApiChapter[]> {
    return this.get<ApiChapter[]>(`/projects/${projectId}/chapters`)
  }

  // ── Tokens (read-only — token creation lives in the SPA) ──────────

  listTokens(): Promise<ApiTokenInfo[]> {
    return this.get<ApiTokenInfo[]>('/me/tokens')
  }

  // ── Transport ──────────────────────────────────────────────────────

  private get<T>(path: string): Promise<T> {
    return this.request<T>(path, { method: 'GET' })
  }

  private post<T>(path: string, body: unknown): Promise<T> {
    return this.request<T>(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
  }

  private async request<T>(path: string, init: RequestInit): Promise<T> {
    let res: Response
    try {
      res = await this._fetch(`${this.base}/api${path}`, {
        ...init,
        headers: {
          Authorization: `Bearer ${this.token}`,
          ...(init.headers ?? {}),
        },
      })
    } catch {
      throw new BackendUnavailableError()
    }

    if (res.status === 401)                      throw new UnauthorizedError()
    if (res.status === 429)                      throw new QuotaExceededError(await readErrorText(res))
    if (res.status === 502 || res.status === 503 || res.status === 504) {
      throw new BackendUnavailableError()
    }
    if (!res.ok) {
      const text = await readErrorText(res)
      throw new Error(`${res.status} ${res.statusText}${text ? ` — ${text}` : ''}`)
    }
    return res.status === 204 ? (undefined as T) : (res.json() as Promise<T>)
  }
}

async function readErrorText(res: Response): Promise<string> {
  try {
    const text = await res.text()
    return text.slice(0, 200)
  } catch {
    return ''
  }
}
