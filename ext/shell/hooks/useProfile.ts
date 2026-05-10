// Cached user profile loaded from chrome.storage.local. Populated on
// first verify (SetupView) and refreshed lazily — the popup header
// and picker overlay use it to show "logged in as X". No re-fetch
// on mount: stale name is fine, the source of truth is the engine
// and we already authed the token.

import { useEffect, useState } from 'react'
import { PROFILE_KEY, type UserProfile } from '@core/config'
import { chromeStorage } from '@shell/adapters/chrome-storage'

export function useProfile(): UserProfile | null {
  const [profile, setProfile] = useState<UserProfile | null>(null)

  useEffect(() => {
    let alive = true
    void chromeStorage.get<UserProfile>(PROFILE_KEY).then(p => {
      if (alive && p) setProfile(p)
    })

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
  }, [])

  return profile
}
