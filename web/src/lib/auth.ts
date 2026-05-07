import { useEffect, useState } from 'react'
import { useNavigate } from '@tanstack/react-router'

const TOKEN_KEY      = 'typoon_token'
const ERROR_KEY      = 'typoon_login_error'
const STATE_KEY      = 'typoon_oauth_state'

const API_BASE = import.meta.env.VITE_API_URL ?? ''

export interface AuthUser {
  id:           number
  display_name: string
  avatar_url:   string | null
  email:        string | null
  tier:         'member' | 'admin'
  created_at:   string | null
  last_login_at: string | null
}

interface AuthConfig {
  discord_client_id: string
  guild_gated:       boolean
}

// ── Token storage ────────────────────────────────────────────────────────────

export const getToken = (): string | null => localStorage.getItem(TOKEN_KEY)
export const setToken = (token: string)   => localStorage.setItem(TOKEN_KEY, token)
export const clearToken = ()              => localStorage.removeItem(TOKEN_KEY)

export function takeLoginError(): string | null {
  const e = sessionStorage.getItem(ERROR_KEY)
  if (e) sessionStorage.removeItem(ERROR_KEY)
  return e
}

export function setLoginError(msg: string) {
  sessionStorage.setItem(ERROR_KEY, msg)
}

// ── Current user ─────────────────────────────────────────────────────────────

interface AuthState {
  user:    AuthUser | null
  loading: boolean
  error:   string | null
}

export function useCurrentUser(): AuthState {
  const [state, setState] = useState<AuthState>({ user: null, loading: true, error: null })

  useEffect(() => {
    const token = getToken()
    if (!token) {
      setState({ user: null, loading: false, error: null })
      return
    }
    fetch(`${API_BASE}/api/auth/me`, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then(async (r) => {
        if (r.status === 401) {
          clearToken()
          setState({ user: null, loading: false, error: null })
          return
        }
        if (!r.ok) throw new Error(`${r.status} ${r.statusText}`)
        const user = (await r.json()) as AuthUser
        setState({ user, loading: false, error: null })
      })
      .catch((e: Error) => setState({ user: null, loading: false, error: e.message }))
  }, [])

  return state
}

export function useLogout() {
  const nav = useNavigate()
  return () => {
    clearToken()
    fetch(`${API_BASE}/api/auth/logout`, { method: 'POST' }).catch(() => {})
    nav({ to: '/login' })
  }
}

// ── OAuth flow ───────────────────────────────────────────────────────────────

const REDIRECT_PATH = '/auth/callback'

/** Where Discord redirects after consent. The SPA owns this URL — must
 *  exactly match what the backend uses when exchanging the code. */
export function redirectUri(): string {
  return `${window.location.origin}${REDIRECT_PATH}`
}

/** Fetch public auth config (client_id, guild_gated hint). */
export async function fetchAuthConfig(): Promise<AuthConfig> {
  const r = await fetch(`${API_BASE}/api/auth/config`)
  if (!r.ok) throw new Error(`auth config: ${r.status}`)
  return r.json()
}

/** Build the Discord authorize URL. CSRF state is generated here and
 *  stashed in sessionStorage so the callback page can verify it. */
export function buildAuthorizeUrl(clientId: string): string {
  const state = crypto.randomUUID()
  sessionStorage.setItem(STATE_KEY, state)

  const params = new URLSearchParams({
    client_id:     clientId,
    redirect_uri:  redirectUri(),
    response_type: 'code',
    scope:         'identify email guilds',
    state,
    prompt:        'consent',
  })
  return `https://discord.com/api/oauth2/authorize?${params}`
}

/** Verify the state parameter matches what we stored at /login.
 *  Returns true on match, false on mismatch (or no stored state). */
export function verifyState(received: string | null): boolean {
  if (!received) return false
  const expected = sessionStorage.getItem(STATE_KEY)
  sessionStorage.removeItem(STATE_KEY)
  return !!expected && expected === received
}

/** POST the code to the engine, get JWT back. */
export async function exchangeCode(code: string): Promise<string> {
  const r = await fetch(`${API_BASE}/api/auth/discord/exchange`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ code, redirect_uri: redirectUri() }),
  })
  if (!r.ok) {
    const text = await r.text().catch(() => '')
    throw new Error(`exchange: ${r.status} ${text.slice(0, 200)}`)
  }
  const { token } = await r.json()
  return token as string
}
