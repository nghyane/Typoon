import { useEffect, useState } from 'react'
import { useNavigate } from '@tanstack/react-router'

const TOKEN_KEY = 'typoon_token'
const ERROR_KEY = 'typoon_login_error'

export interface AuthUser {
  id:           number
  display_name: string
  avatar_url:   string | null
  email:        string | null
  tier:         'member' | 'admin'
  created_at:   string | null
  last_login_at: string | null
}

// ── Token storage ────────────────────────────────────────────────────────────

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY)
}

export function setToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token)
}

export function clearToken(): void {
  localStorage.removeItem(TOKEN_KEY)
}

// One-shot login error from the OAuth bootstrap page (see
// typoon/api/routes/auth.py — _BOOTSTRAP_HTML).
export function takeLoginError(): string | null {
  const e = sessionStorage.getItem(ERROR_KEY)
  if (e) sessionStorage.removeItem(ERROR_KEY)
  return e
}

// ── Auth state hook ──────────────────────────────────────────────────────────

interface AuthState {
  user:    AuthUser | null
  loading: boolean
  error:   string | null
}

const API_BASE = import.meta.env.VITE_API_URL ?? ''

/** Fetches /api/auth/me. On 401, clears token and returns null. */
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

// ── Logout ───────────────────────────────────────────────────────────────────

export function useLogout() {
  const nav = useNavigate()
  return () => {
    clearToken()
    // Hit the logout endpoint best-effort (it's a no-op server-side for
    // JWT, but keeps room for future revocation).
    fetch(`${API_BASE}/api/auth/logout`, { method: 'POST' }).catch(() => {})
    nav({ to: '/login' })
  }
}

// ── OAuth entry URL ──────────────────────────────────────────────────────────

export function loginUrl(): string {
  // Hit the engine's OAuth start endpoint; it sets a state cookie and
  // 302s to Discord. Going through the engine (instead of building the
  // Discord URL on the client) keeps state generation server-side.
  return `${API_BASE}/api/auth/discord/login`
}
