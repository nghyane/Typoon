// Config schema persisted to chrome.storage.local. Single source of
// truth for the user's session.
//
// API URL is NOT a config field — it's baked at build time via
// `__API_URL__` (see wxt.config.ts) so users only paste a token. The
// engine origin is granted via host_permissions at install time, so no
// runtime permission prompt is needed either.

import type { StorageAdapter } from '@core/adapters/storage'

/** Build-time engine URL (e.g. `https://927251094806098001.discordsays.com/api`).
 *  Source: VITE_API_URL → `__API_URL__` define in wxt.config.ts. */
export const API_URL: string = __API_URL__

export interface Config {
  token:         string
  lastProjectId: number | null
}

export const CONFIG_KEY = 'typoon.config'

export const EMPTY_CONFIG: Config = {
  token:         '',
  lastProjectId: null,
}

export async function loadConfig(storage: StorageAdapter): Promise<Config> {
  const cur = await storage.get<Partial<Config>>(CONFIG_KEY)
  return { ...EMPTY_CONFIG, ...(cur ?? {}) }
}

export async function saveConfig(
  storage: StorageAdapter, patch: Partial<Config>,
): Promise<Config> {
  const cur  = await loadConfig(storage)
  const next = { ...cur, ...patch }
  await storage.set(CONFIG_KEY, next)
  return next
}

export function hasAuth(c: Config): boolean {
  return Boolean(c.token)
}

/** User profile (name + avatar) cached after token verify. The
 *  picker overlay reads it from chrome.storage to show "logged in
 *  as X" without a network round-trip. */
export interface UserProfile {
  display_name: string
  avatar_url:   string | null
}

export const PROFILE_KEY = 'typoon.profile'
