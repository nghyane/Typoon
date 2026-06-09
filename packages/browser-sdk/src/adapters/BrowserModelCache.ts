import type { ModelAssetCache } from './modelTypes'

export class BrowserModelCache implements ModelAssetCache {
  constructor(private readonly cacheName: string) {}

  async match(key: string): Promise<ArrayBuffer | null> {
    if (!('caches' in globalThis)) return null
    const cache = await caches.open(this.cacheName)
    const cached = await cache.match(new Request(key, { mode: 'cors' }))
    return cached ? cached.arrayBuffer() : null
  }

  async put(key: string, bytes: ArrayBuffer): Promise<void> {
    if (!('caches' in globalThis)) return
    const cache = await caches.open(this.cacheName)
    await cache.put(new Request(key, { mode: 'cors' }), new Response(bytes))
  }
}
