// chrome.storage.local-backed StorageAdapter. Used by every surface
// (popup, SW, offscreen, content) — chrome.storage is shared across
// contexts inside a single extension, which is exactly the property the
// queue and config rely on.

import type { StorageAdapter } from '@core/adapters/storage'

export const chromeStorage: StorageAdapter = {
  async get<T>(key: string): Promise<T | undefined> {
    const out = await browser.storage.local.get(key)
    return out[key] as T | undefined
  },
  async set<T>(key: string, value: T): Promise<void> {
    await browser.storage.local.set({ [key]: value })
  },
  async del(key: string): Promise<void> {
    await browser.storage.local.remove(key)
  },
}
