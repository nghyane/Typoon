// Session — cookie-based auth for PWA (Go backend).
//
// No localStorage token. Cookies are httpOnly, sent automatically.
// `GET /api/auth/session` returns user info or 401.

import { useCallback } from 'react'
import {
  useQuery, useQueryClient, type QueryClient,
} from '@tanstack/react-query'
import { discordSdk } from '@shared/discord/sdk'
import { qk } from '@shared/api/keys'
import { api, setDaToken, type SessionUser } from '@shared/api/api'

export type { SessionUser }

// ── Session query ──────────────────────────────────────────────────

class SessionRejectedError extends Error {
  constructor() { super('session rejected') }
}

async function fetchSession(): Promise<SessionUser | null> {
  try {
    return await api.getSession()
  } catch (err) {
    if ((err as { status?: number })?.status === 401) {
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

export { isDiscordActivity } from '@shared/discord/sdk'

// ── Auth URLs ──────────────────────────────────────────────────────

export function loginUrl(returnTo = '/') {
  return '/api/auth/discord/start?returnTo=' + encodeURIComponent(safeReturnTo(returnTo))
}

export function safeReturnTo(value: unknown): string {
  if (typeof value !== 'string') return '/'
  if (!value.startsWith('/') || value.startsWith('//')) return '/'
  if (value.startsWith('/login') || value.startsWith('/auth/callback')) return '/'
  return value
}

// ── DA (Discord Activity) silent login ─────────────────────────────

export async function discordActivityLogin(): Promise<void> {
  await discordSdk.ready()
  const { code } = await discordSdk.commands.authorize({
    client_id:     import.meta.env.VITE_DISCORD_CLIENT_ID as string,
    response_type: 'code',
    state:         '',
    prompt:        'none',
    scope:         ['identify', 'guilds', 'guilds.members.read'],
  })

  const result = await api.daExchange(code)
  setDaToken(result.token)
}

// ── Sign-in / sign-out ─────────────────────────────────────────────

export function useSignIn() {
  const qc = useQueryClient()
  return useCallback(async () => {
    qc.clear()
    const user = await api.getSession()
    qc.setQueryData(qk.session.self(), user)
    return user
  }, [qc])
}

export function useRefreshSession() {
  const qc = useQueryClient()
  return useCallback(() => (
    qc.invalidateQueries({ queryKey: qk.session.self() })
  ), [qc])
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
  qc.clear()
}

// ── Preferences (placeholder — Go endpoint TBD) ──────────────────

type UpdatePreferredLang = (lang: string | null) => void

export function useUpdatePreferredLang(): UpdatePreferredLang {
  return useCallback(() => {
    // TODO: POST /api/auth/me/preferences when Go backend supports it
  }, [])
}
