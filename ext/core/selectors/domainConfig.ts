// Per-origin selector the user trains during the picker wizard.
// Phase 1 stores ONE selector per domain — the image container. Title
// + chapter number do not survive long enough on dynamic sites to be
// worth caching; the user pastes them (optional) at upload time.

import type { StorageAdapter } from '@core/adapters/storage'

export interface DomainConfig {
  /** Origin host, e.g. `mangadex.org`. */
  domain:           string
  /** CSS selector that matched the image container at training time. */
  imagesSelector:   string
  /** Last-known image count under the selector. UI uses this to warn
   *  ("trained against 24 images, only finds 3 now"). */
  expectedCount:    number
  createdAt:        number
}

export const DOMAINS_KEY = 'typoon.domains'

type DomainMap = Record<string, DomainConfig>

export async function loadDomains(storage: StorageAdapter): Promise<DomainMap> {
  return (await storage.get<DomainMap>(DOMAINS_KEY)) ?? {}
}

export async function getDomainConfig(
  storage: StorageAdapter, domain: string,
): Promise<DomainConfig | null> {
  const all = await loadDomains(storage)
  return all[domain] ?? null
}

export async function saveDomainConfig(
  storage: StorageAdapter, cfg: DomainConfig,
): Promise<void> {
  const all = await loadDomains(storage)
  all[cfg.domain] = cfg
  await storage.set(DOMAINS_KEY, all)
}

export async function clearDomainConfig(
  storage: StorageAdapter, domain: string,
): Promise<void> {
  const all = await loadDomains(storage)
  delete all[domain]
  await storage.set(DOMAINS_KEY, all)
}

/** Normalise a tab URL to the storage key. Strips port, query, hash. */
export function originKey(url: string): string {
  try {
    return new URL(url).host
  } catch {
    return ''
  }
}
