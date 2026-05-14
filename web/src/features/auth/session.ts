// Session — single source of truth for "who is the current viewer".
//
// Before this module the SPA had three parallel auth surfaces:
//
//   • `useCurrentUser` — raw fetch+useState, no cache, 3 callsites
//     meant 3 independent `/api/auth/me` round-trips on every mount.
//   • `useMe`           — React Query against `/api/me`, a slim
//     subset of the same payload.
//   • `fetchAuthConfig` — raw fetch on every LoginPage mount.
//
// Two endpoints / two shapes / two caches for the same concept led
// to a real bug: after a 401 mid-session, LoginPage's auto-DA-login
// effect could race the (still-pending) `useCurrentUser` fetch and
// silently re-authorize before `markSessionInvalidated()` had a
// chance to run — looping the user through Discord.
//
// The new design:
//
//   • One endpoint  — `GET /api/auth/me` (backend `SessionUser`),
//     covering identity + admin flag + reading-lang preference.
//   • One cache key — `qk.session.self()` via React Query.
//   • One hook      — `useSession()` returning a discriminated state
//     (`'unauthenticated' | 'loading' | 'authenticated' | 'error'`).
//   • One mutation API — `signIn`/`signOut` plus the OAuth helpers,
//     all of which update the cache cooperatively instead of forcing
//     a hard `window.location.replace`.
//
// Token storage is still `localStorage` (no service worker, no
// httpOnly cookies — DA iframes block 3rd-party cookies anyway).
// The cache layer is React Query; routes read `useSession()` and
// the auth guard in AppLayout uses `status` directly, no useEffect
// redirect ping-pong.

import { useCallback } from 'react'
import {
  useMutation, useQuery, useQueryClient,
  type QueryClient,
} from '@tanstack/react-query'

import { discordSdk } from '@shared/discord/sdk'
import { qk } from '@shared/api/keys'


// ── Wire types ───────────────────────────────────────────────────────────────

/** Canonical session payload. Wire-mirror of backend `SessionUser`. */
export interface SessionUser {
  id:                    number
  display_name:          string
  avatar_url:            string | null
  is_admin:              boolean
  preferred_target_lang: string | null
}


/** Public auth config the /login page needs before kicking off OAuth.
 *  Backend currently only exposes the client id — keep this surface
 *  honest so callers don't render fields that always come back null. */
export interface AuthConfig {
  discord_client_id: string
}


// ── Token storage ────────────────────────────────────────────────────────────

const TOKEN_KEY        = 'typoon_token'
const ERROR_KEY        = 'typoon_login_error'
const STATE_KEY        = 'typoon_oauth_state'
/** Marks "had a token, server rejected it". sessionStorage scope:
 *  survives the same-tab navigate to /login but resets on new tabs
 *  / app launches where silent auto-login is the right default. */
const INVALIDATED_KEY  = 'typoon_session_invalidated'

const getToken   = (): string | null => localStorage.getItem(TOKEN_KEY)
const setToken   = (token: string)   => localStorage.setItem(TOKEN_KEY, token)
const clearToken = ()                => localStorage.removeItem(TOKEN_KEY)

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

/** Read + clear the "session was invalidated" flag. The /login page
 *  uses this to suppress silent DA re-authorize (which would log the
 *  user back in under whichever Discord account is currently active
 *  in the client — surprising when they were expecting their old
 *  session back). */
export function takeSessionInvalidated(): boolean {
  const hit = sessionStorage.getItem(INVALIDATED_KEY) === '1'
  if (hit) sessionStorage.removeItem(INVALIDATED_KEY)
  return hit
}


// ── API base + low-level fetch ───────────────────────────────────────────────

// Production hosts the SPA + API behind one origin (the DA host
// fronts /api). Inside the DA iframe we use same-origin paths;
// outside (plain browser) we cross-origin to the same host via the
// build-time `VITE_PUBLIC_BASE_URL`.
const isDA     = window.location.hostname.endsWith('.discordsays.com')
const API_BASE = isDA ? '' : (import.meta.env.VITE_PUBLIC_BASE_URL ?? '')

async function authFetch(path: string, init?: RequestInit): Promise<Response> {
  const token   = getToken()
  const headers = new Headers(init?.headers)
  if (token) headers.set('Authorization', `Bearer ${token}`)
  return fetch(`${API_BASE}${path}`, { ...init, headers })
}


// ── Session query ────────────────────────────────────────────────────────────

/** Thrown by the session query when the stored token was rejected
 *  by the server. The hook turns this into `status: 'unauthenticated'`
 *  rather than `'error'` so the guard can route to /login without
 *  surfacing the technical detail. */
class SessionRejectedError extends Error {
  constructor() { super('session rejected') }
}


/** Fetch the canonical session payload. Throws `SessionRejectedError`
 *  on 401 (token present but server-side row gone or expired) so the
 *  caller can distinguish "logged out" from "network error". */
async function fetchSession(): Promise<SessionUser | null> {
  if (!getToken()) return null
  const r = await authFetch('/api/auth/me')
  if (r.status === 401) {
    clearToken()
    markSessionInvalidated()
    throw new SessionRejectedError()
  }
  if (!r.ok) throw new Error(`session: ${r.status} ${r.statusText}`)
  return (await r.json()) as SessionUser
}


/** Discriminated session state. Components branch on `status` and
 *  TypeScript narrows the payload — no more `if (loading) ... else
 *  if (user) ...` shapes that miss the error case. */
export type SessionState =
  | { status: 'loading';         user: null }
  | { status: 'unauthenticated'; user: null }
  | { status: 'authenticated';   user: SessionUser }
  | { status: 'error';           user: null; error: Error }


/** Subscribe to the session cache. Every consumer in the app shares
 *  one `qk.session.self()` query — three callers = one fetch. */
export function useSession(): SessionState {
  const q = useQuery<SessionUser | null, Error>({
    queryKey:             qk.session.self(),
    queryFn:              fetchSession,
    staleTime:            5 * 60_000,
    gcTime:               10 * 60_000,
    retry:                false,    // 401 must NOT retry
    refetchOnWindowFocus: false,
  })

  if (q.isPending) return { status: 'loading',         user: null }
  if (q.error) {
    if (q.error instanceof SessionRejectedError) {
      return { status: 'unauthenticated', user: null }
    }
    return { status: 'error', user: null, error: q.error }
  }
  if (!q.data) return { status: 'unauthenticated', user: null }
  return { status: 'authenticated', user: q.data }
}


/** Convenience: `useSession().user` when the caller already knows
 *  it's inside an auth-required route. Returns `null` until the
 *  guard in AppLayout has resolved. */
export function useSessionUser(): SessionUser | null {
  return useSession().user
}


// ── Auth config query ───────────────────────────────────────────────────────

async function fetchAuthConfig(): Promise<AuthConfig> {
  const r = await fetch(`${API_BASE}/api/auth/config`)
  if (!r.ok) {
    let msg = `auth config: ${r.status}`
    try { const b = await r.json(); if (b?.detail) msg = b.detail } catch { /* */ }
    throw new Error(msg)
  }
  return r.json()
}

/** Cached `/api/auth/config`. Static at runtime (it carries the
 *  Discord client id baked into the deployment) so we treat it as
 *  effectively immutable per session. */
export function useAuthConfig() {
  return useQuery<AuthConfig, Error>({
    queryKey:  qk.session.config(),
    queryFn:   fetchAuthConfig,
    staleTime: Infinity,
    gcTime:    Infinity,
    retry:     1,
  })
}


// ── Preferences mutation ────────────────────────────────────────────────────

/** `PATCH /api/me/preferences`. Returns the full `SessionUser`
 *  payload, which we splice straight into the session cache so every
 *  consumer (header avatar, settings, hero) reflects the change
 *  without a round-trip. */
async function patchPreferences(body: {
  preferred_target_lang?: string | null
}): Promise<SessionUser> {
  const r = await authFetch('/api/me/preferences', {
    method:  'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify(body),
  })
  if (!r.ok) {
    let detail = `${r.status} ${r.statusText}`
    try {
      const b = await r.json()
      if (b?.detail) detail = String(b.detail)
    } catch { /* */ }
    throw new Error(detail)
  }
  return (await r.json()) as SessionUser
}


/** Mutation: update viewer's default reading language. Optimistic +
 *  rollback-on-error; success path overwrites the cache with the
 *  fresh payload from the server. */
export function useUpdatePreferredLang() {
  const qc = useQueryClient()

  const mut = useMutation({
    mutationFn: (lang: string | null) =>
      patchPreferences({ preferred_target_lang: lang }),

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

  return useCallback(
    (lang: string | null) => mut.mutate(lang),
    [mut],
  )
}


// ── OAuth flow ──────────────────────────────────────────────────────────────

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
  const r = await fetch(`${API_BASE}/api/auth/discord/exchange`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({
      code, redirect_uri: overrideRedirectUri ?? redirectUri(),
    }),
  })
  if (!r.ok) {
    let detail = `${r.status} ${r.statusText}`
    try {
      const body = await r.json()
      if (body?.detail) detail = String(body.detail)
    } catch { /* not JSON */ }
    throw new Error(detail)
  }
  const { token } = await r.json()
  return token as string
}

/** Discord Activity login — SDK authorize → exchange code → JWT.
 *  The DA `redirect_uri` is a registered placeholder (must match
 *  Discord Developer Portal → OAuth2 → Redirects). */
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


// ── Sign-in / sign-out hooks ────────────────────────────────────────────────

/** Persist a freshly-issued JWT and prime the session cache.
 *  Returns once `qk.session.self()` has been refetched, so callers
 *  that immediately navigate to a guarded route won't race the
 *  guard's first read. */
export function useSignIn() {
  const qc = useQueryClient()
  return useCallback(async (token: string) => {
    setToken(token)
    sessionStorage.removeItem(INVALIDATED_KEY)
    await qc.invalidateQueries({ queryKey: qk.session.self() })
    // Block on the first refetch so the post-login navigate sees
    // `status: 'authenticated'` instead of `'loading'`.
    await qc.refetchQueries({ queryKey: qk.session.self() })
  }, [qc])
}

/** Wipe the token + session cache. Fires the logout endpoint
 *  best-effort (JWT is stateless server-side, so the network call is
 *  cosmetic — but if we ever add a revocation list, this is where it
 *  hooks in). */
export function useSignOut() {
  const qc = useQueryClient()
  return useCallback(async () => {
    clearToken()
    qc.setQueryData(qk.session.self(), null)
    void authFetch('/api/auth/logout', { method: 'POST' }).catch(() => {})
  }, [qc])
}


// ── Global 401 hook ─────────────────────────────────────────────────────────

/** Called by the boot wiring in `main.tsx` when `@shared/api/api`
 *  dispatches `typoon:unauthorized` (a 401 from any request). Clears
 *  the token, marks the session as invalidated (so /login can skip
 *  silent DA re-authorize), and wipes the React Query cache entry
 *  so a stale `['session']` snapshot doesn't keep rendering a
 *  logged-in shell while AppLayout navigates away.
 *
 *  Does NOT re-dispatch the unauthorized event — that would loop
 *  the listener forever. Routing is AppLayout's job. */
export function handleUnauthorized(qc: QueryClient): void {
  clearToken()
  markSessionInvalidated()
  qc.setQueryData(qk.session.self(), null)
}
