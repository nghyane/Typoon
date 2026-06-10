// Session — cookie-based auth for PWA (Go backend).
//
// No localStorage token. Cookies are httpOnly, sent automatically.
// `GET /api/auth/session` returns user info or 401.

import { useCallback } from 'react'
import {
  useMutation, useQuery, useQueryClient, type QueryClient,
} from '@tanstack/react-query'
import { qk } from '@shared/api/keys'
import { api, type SessionUser } from '@shared/api/api'

export type { SessionUser }

// ── Storage flags ──────────────────────────────────────────────────

const ERROR_KEY       = 'typoon_login_error'
const INVALIDATED_KEY = 'typoon_session_invalidated'

export function takeLoginError(): string | null {
  const e = sessionStorage.getItem(ERROR_KEY)
  if (e) sessionStorage.removeItem(ERROR_KEY)
  return e
}

export function setLoginError(msg: string) {
  sessionStorage.setItem(ERROR_KEY, msg)
}

function markSessionInvalidated() {
  sessionStorage.setItem(INVALIDATED_KEY, '1')
}

export function takeSessionInvalidated(): boolean {
  const hit = sessionStorage.getItem(INVALIDATED_KEY) === '1'
  if (hit) sessionStorage.removeItem(INVALIDATED_KEY)
  return hit
}

// ── Session query ──────────────────────────────────────────────────

class SessionRejectedError extends Error {
  constructor() { super('session rejected') }
}

async function fetchSession(): Promise<SessionUser | null> {
  try {
    return await api.getSession()
  } catch (err) {
    if ((err as { status?: number })?.status === 401) {
      markSessionInvalidated()
      throw new SessionRejectedError()
    }
    throw err
  }
}

export type SessionState =
  | { status: 'loading';         user: null }
  | { status: 'unauthenticated'; user: null }
  | { status: 'authenticated';   user: SessionUser }
  | { status: 'error';           user: null; error: Error }

export function useSession(): SessionState {
  const q = useQuery<SessionUser | null, Error>({
    queryKey:             qk.session.self(),
    queryFn:              fetchSession,
    staleTime:            5 * 60_000,
    gcTime:               10 * 60_000,
    retry:                false,
    refetchOnWindowFocus: false,
  })
  if (q.isPending) return { status: 'loading', user: null }
  if (q.error) {
    if (q.error instanceof SessionRejectedError) {
      return { status: 'unauthenticated', user: null }
    }
    return { status: 'error', user: null, error: q.error }
  }
  if (!q.data) return { status: 'unauthenticated', user: null }
  return { status: 'authenticated', user: q.data }
}

export function useSessionUser(): SessionUser | null {
  return useSession().user
}

// ── Auth URLs ──────────────────────────────────────────────────────

export function loginUrl() {
  return '/api/auth/discord/start?returnTo=' + encodeURIComponent(window.location.origin + '/')
}

// ── Sign-in / sign-out ─────────────────────────────────────────────

export function useSignIn() {
  const qc = useQueryClient()
  return useCallback(async () => {
    sessionStorage.removeItem(INVALIDATED_KEY)
    qc.clear()
    await qc.refetchQueries({ queryKey: qk.session.self() })
  }, [qc])
}

export function useSignOut() {
  const qc = useQueryClient()
  return useCallback(async () => {
    try { await api.logout() } catch { /* ignore */ }
    qc.clear()
  }, [qc])
}

// ── Global 401 hook ────────────────────────────────────────────────

export function handleUnauthorized(qc: QueryClient): void {
  markSessionInvalidated()
  qc.clear()
}

// ── Preferences (placeholder — Go endpoint TBD) ──────────────────

export function useUpdatePreferredLang() {
  return useCallback((_lang: string | null) => {
    // TODO: POST /api/auth/me/preferences when Go backend supports it
  }, [])
}
