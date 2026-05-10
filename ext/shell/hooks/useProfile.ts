// Cached user profile loaded from chrome.storage.local. Populated on
// first verify (SetupView) and refreshed lazily — the popup header
// and picker overlay use it to show "logged in as X".
//
// Self-heal: if storage is empty but a token exists (user upgraded
// from a build that didn't cache profile, or storage was wiped),
// fetch /auth/me once and cache the result. Otherwise the header
// hangs on "Đang tải…" forever because nothing else writes the key.

import { useEffect, useState } from 'react'
import { API_URL, PROFILE_KEY, type UserProfile } from '@core/config'
import { chromeStorage } from '@shell/adapters/chrome-storage'
import { TypoonClient } from '@core/typoon'
import { useConfig } from './useConfig'

export function useProfile(): UserProfile | null {
  const { config } = useConfig()
  const token = config.token
  const [profile, setProfile] = useState<UserProfile | null>(null)

  useEffect(() => {
    let alive = true

    void (async () => {
      const cached = await chromeStorage.get<UserProfile>(PROFILE_KEY)
      if (!alive) return
      if (cached) { setProfile(cached); return }
      if (!token) return

      // Storage miss + we have a token → backfill once. Errors are
      // swallowed: the header just shows the avatar placeholder
      // until the next successful round-trip elsewhere caches it.
      try {
        const me = await new TypoonClient({ apiUrl: API_URL, token }).me()
        if (!alive) return
        const next: UserProfile = {
          display_name: me.display_name,
          avatar_url:   me.avatar_url,
        }
        setProfile(next)
        await chromeStorage.set(PROFILE_KEY, next)
      } catch {
        /* offline / 401 — leave profile null, picker still works */
      }
    })()

    const onChange = (
      changes: Record<string, Browser.storage.StorageChange>,
      area:    Browser.storage.AreaName,
    ) => {
      if (area !== 'local' || !(PROFILE_KEY in changes)) return
      setProfile((changes[PROFILE_KEY]?.newValue as UserProfile | undefined) ?? null)
    }
    browser.storage.onChanged.addListener(onChange)
    return () => {
      alive = false
      browser.storage.onChanged.removeListener(onChange)
    }
  }, [token])

  return profile
}
