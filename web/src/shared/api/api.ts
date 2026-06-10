// Typoon API client — session via httpOnly cookie (Go backend).
//
// No localStorage token. Cookies are sent automatically by the browser.
// 401 → dispatch `typoon:unauthorized` for the UI to redirect.

// ── Base URL ────────────────────────────────────────────────────────

const API_BASE = window.location.hostname.endsWith('.discordsays.com')
  ? ''
  : (import.meta.env.VITE_PUBLIC_BASE_URL ?? '')

// ── Helpers ─────────────────────────────────────────────────────────

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
      credentials: 'include',
      headers: {
        ...(init.headers ?? {}),
      },
    })
  } catch {
    throw new BackendUnavailableError()
  }

  if (res.status === 401) {
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

// ── Types ───────────────────────────────────────────────────────────

export interface SessionUser {
  id:           string
  display_name: string
  avatar_url:   string | null
}

// ── Endpoints ───────────────────────────────────────────────────────

export const api = {
  // Auth
  getSession: (): Promise<SessionUser> =>
    request('/api/auth/session'),

  logout: (): Promise<{ ok: boolean }> =>
    request('/api/auth/logout', { method: 'POST' }),

  // Coin / wallet
  getCoinPackages: (): Promise<{ packages: Array<{
    id:    string
    name:  string
    xu:    number
    bonus: number
    price: number
  }> }> =>
    request('/api/coin-packages'),

  getWallet: (): Promise<{ available: number; held: number }> =>
    request('/api/wallet'),

  // Translation
  createTranslationSession: (input: {
    contentKey: string
    targetLang: string
    mode:       string
    pageCount?: number
    unitCount?: number
  }): Promise<{ id: string; state: string; xuState: string; priceXu: number }> =>
    request('/api/translation-sessions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(input),
    }),

  refineWindow: (sessionId: string, input: {
    sessionId:     string
    sourceLang:    string | null
    targetLang:    string
    units:         unknown[]
    activeUnitIds: string[]
    contextBlock?: string
  }) =>
    request(`/api/translation-sessions/${sessionId}/refine-windows`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(input),
    }),
}
