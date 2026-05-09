import { useEffect, useState } from 'react'
import { useNavigate } from '@tanstack/react-router'
import { discordSdk } from '@shared/discord/sdk'

const TOKEN_KEY  = 'typoon_token'
const ERROR_KEY  = 'typoon_login_error'
const STATE_KEY  = 'typoon_oauth_state'

// In DA, Discord proxy handles relative /api/* paths via URL Mappings on the portal.
// Outside DA, VITE_API_URL is set for cross-origin (CF Pages → API).
const isDA = window.location.hostname.endsWith('.discordsays.com')
const API_BASE = isDA ? '' : (import.meta.env.VITE_API_URL ?? '')

export interface AuthUser {
  id:            number
  display_name:  string
  avatar_url:    string | null
  email:         string | null
  is_admin:      boolean
  created_at:    string | null
  last_login_at: string | null
  guild_name:    string | null
  guild_icon_url: string | null
}

export interface AuthConfig {
  discord_client_id:  string
  guild_gated:        boolean
  guild_name:         string | null
  guild_icon_url:     string | null
  discord_invite_url: string | null
}

// ── Token storage ────────────────────────────────────────────────────────────

export const getToken   = (): string | null => localStorage.getItem(TOKEN_KEY)
export const setToken   = (token: string)   => localStorage.setItem(TOKEN_KEY, token)
export const clearToken = ()                => localStorage.removeItem(TOKEN_KEY)

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
  const [state, setState] = useState<AuthState>(() => ({
    user:    null,
    loading: !!getToken(),
    error:   null,
  }))

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

// ── OAuth flow ────────────────────────────────────────────────────────────────

const REDIRECT_PATH = '/auth/callback'

export function redirectUri(): string {
  return `${window.location.origin}${REDIRECT_PATH}`
}

export async function fetchAuthConfig(): Promise<AuthConfig> {
  const r = await fetch(`${API_BASE}/api/auth/config`)
  if (!r.ok) {
    let msg = `auth config: ${r.status}`
    try { const b = await r.json(); if (b?.detail) msg = b.detail } catch { /* */ }
    throw new Error(msg)
  }
  return r.json()
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

/** Discord Activity login — SDK authorize → exchange code → Typoon JWT. */
export async function discordActivityLogin(clientId: string): Promise<string> {
  await discordSdk.ready()
  const { code } = await discordSdk.commands.authorize({
    client_id:     clientId,
    response_type: 'code',
    state:         '',
    prompt:        'none',
    scope:         ['identify', 'email', 'guilds', 'guilds.members.read'],
  })
  // DA redirect_uri placeholder — must be registered in Discord Developer Portal → OAuth2 → Redirects
  return exchangeCode(code, 'https://127.0.0.1')
}

export function verifyState(received: string | null): boolean {
  if (!received) return false
  const expected = sessionStorage.getItem(STATE_KEY)
  sessionStorage.removeItem(STATE_KEY)
  return !!expected && expected === received
}

export async function exchangeCode(code: string, overrideRedirectUri?: string): Promise<string> {
  const r = await fetch(`${API_BASE}/api/auth/discord/exchange`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ code, redirect_uri: overrideRedirectUri ?? redirectUri() }),
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
