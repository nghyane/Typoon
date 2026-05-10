// Subscribe to chrome.storage.local-backed config from any React surface
// (popup, future side panel). Returns [config, save, ready].
//
// Subscription: chrome.storage.onChanged broadcasts to every context, so
// when the popup writes config, the SW + content scripts see the same
// update without polling.

import { useCallback, useEffect, useState } from 'react'
import {
  CONFIG_KEY, EMPTY_CONFIG, hasAuth, loadConfig, saveConfig,
  type Config,
} from '@core/config'
import { chromeStorage } from '@shell/adapters/chrome-storage'

export interface UseConfig {
  config: Config
  ready:  boolean
  /** Returns the merged config so callers can chain on the new state. */
  save:   (patch: Partial<Config>) => Promise<Config>
  authed: boolean
}

export function useConfig(): UseConfig {
  const [config, setConfig] = useState<Config>(EMPTY_CONFIG)
  const [ready,  setReady]  = useState(false)

  useEffect(() => {
    let alive = true
    loadConfig(chromeStorage).then(c => {
      if (!alive) return
      setConfig(c)
      setReady(true)
    })

    // Cross-context updates: another surface (or the SW) edits config.
    const onChange = (
      changes: Record<string, Browser.storage.StorageChange>,
      area:    Browser.storage.AreaName,
    ) => {
      if (area !== 'local' || !(CONFIG_KEY in changes)) return
      const next = (changes[CONFIG_KEY]?.newValue as Partial<Config> | undefined) ?? {}
      setConfig({ ...EMPTY_CONFIG, ...next })
    }
    browser.storage.onChanged.addListener(onChange)
    return () => {
      alive = false
      browser.storage.onChanged.removeListener(onChange)
    }
  }, [])

  const save = useCallback(async (patch: Partial<Config>) => {
    const next = await saveConfig(chromeStorage, patch)
    setConfig(next)
    return next
  }, [])

  return { config, ready, save, authed: hasAuth(config) }
}
