// Session — single source of truth for "who is the current viewer".
//
// One endpoint (`GET /api/auth/me`), one cache key (`qk.session.self`),
// one hook (`useSession`), one mutation surface (`useSignIn`/`useSignOut`).
//
// Token storage is `localStorage` (DA iframes block 3rd-party cookies).
// Cache layer is React Query; routes read `useSession()` and AppLayout's
// auth guard branches on `status` directly — no useEffect ping-pong.

import { useCallback } from 'react'
import {
  useMutation, useQuery, useQueryClient, type QueryClient,
} from '@tanstack/react-query'

import { discordSdk } from '@shared/discord/sdk'
import { qk } from '@shared/api/keys'
import { api, type ApiSessionUser, clearToken, setToken } from '@shared/api/api'


// Public auth config the /login page needs before kicking off OAuth.
export interface AuthConfig {
  discord_client_id: string
}

export type SessionUser = ApiSessionUser


// ── Storage flags ────────────────────────────────────────────────────

const ERROR_KEY       = 'typoon_login_error'
const STATE_KEY       = 'typoon_oauth_state'
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


// ── Session query ────────────────────────────────────────────────────

class SessionRejectedError extends Error {
  constructor() { super('session rejected') }
}

async function fetchSession(): Promise<SessionUser | null> {
  // No token → unauthenticated without round-trip
  if (!localStorage.getItem('typoon_token')) return null
  try {
    return await api.authMe()
  } catch (err) {
    // ApiError 401 already cleared token via the global handler
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


// ── Auth config ──────────────────────────────────────────────────────

export function useAuthConfig() {
  return useQuery<AuthConfig, Error>({
    queryKey:  qk.session.config(),
    queryFn:   () => api.authConfig(),
    staleTime: Infinity,
    gcTime:    Infinity,
    retry:     1,
  })
}


// ── Preferences mutation ─────────────────────────────────────────────

export function useUpdatePreferredLang() {
  const qc = useQueryClient()
  const mut = useMutation({
    mutationFn: (lang: string | null) =>
      api.updatePreferences({ preferred_target_lang: lang }),
    onMutate: async (lang) => {
      await qc.cancelQueries({ queryKey: qk.session.self() })
      const prev = qc.getQueryData<SessionUser>(qk.session.self())
      if (prev) {
        qc.setQueryData<SessionUser>(qk.session.self(), {
          ...prev, preferred_target_lang: lang,
        })
      }
      return { prev }
    },
    onError: (_e: Error, _lang, ctx) => {
      if (ctx?.prev) qc.setQueryData(qk.session.self(), ctx.prev)
    },
    onSuccess: (fresh) => {
      qc.setQueryData(qk.session.self(), fresh)
    },
  })
  return useCallback((lang: string | null) => mut.mutate(lang), [mut])
}


// ── OAuth flow ───────────────────────────────────────────────────────

const REDIRECT_PATH = '/auth/callback'

function redirectUri(): string {
  return `${window.location.origin}${REDIRECT_PATH}`
}

export function buildAuthorizeUrl(clientId: string): string {
  const state = crypto.randomUUID()
  sessionStorage.setItem(STATE_KEY, state)
  const params = new URLSearchParams({
    client_id:     clientId,
    redirect_uri:  redirectUri(),
    response_type: 'code',
    scope:         'identify email guilds guilds.members.read',
    state,
    prompt:        'consent',
  })
  return `https://discord.com/api/oauth2/authorize?${params}`
}

export function verifyState(received: string | null): boolean {
  if (!received) return false
  const expected = sessionStorage.getItem(STATE_KEY)
  sessionStorage.removeItem(STATE_KEY)
  return !!expected && expected === received
}

export async function exchangeCode(
  code: string, overrideRedirectUri?: string,
): Promise<string> {
  const result = await api.discordExchange(code, overrideRedirectUri ?? redirectUri())
  return result.token
}

/** Discord Activity login — SDK authorize → exchange code → JWT. */
export async function discordActivityLogin(clientId: string): Promise<string> {
  await discordSdk.ready()
  const { code } = await discordSdk.commands.authorize({
    client_id:     clientId,
    response_type: 'code',
    state:         '',
    prompt:        'none',
    scope:         ['identify', 'email', 'guilds', 'guilds.members.read', 'rpc.activities.write'],
  })
  return exchangeCode(code, 'https://127.0.0.1')
}


// ── Sign-in / sign-out hooks ─────────────────────────────────────────

export function useSignIn() {
  const qc = useQueryClient()
  return useCallback(async (token: string) => {
    setToken(token)
    sessionStorage.removeItem(INVALIDATED_KEY)
    qc.clear()
    await qc.refetchQueries({ queryKey: qk.session.self() })
  }, [qc])
}

export function useSignOut() {
  const qc = useQueryClient()
  return useCallback(async () => {
    clearToken()
    qc.clear()
  }, [qc])
}


// ── Global 401 hook ──────────────────────────────────────────────────

export function handleUnauthorized(qc: QueryClient): void {
  clearToken()
  markSessionInvalidated()
  qc.clear()
}
