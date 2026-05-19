/**
 * Cache helpers for idempotent stage RPC.
 *
 * Workers' default Cache API (`caches.default`) is a key/value store
 * scoped to the colo. A cache hit is ~5ms; a miss costs whatever the
 * inner function costs plus a `waitUntil` write. We use it to make
 * `prepare` and `scan` reruns free during development and replay.
 *
 * Cache scheme:
 *   - Key  = synthetic `https://cache.internal/<stage>/<digest>` URL.
 *   - Body = JSON-serialised return value.
 *   - TTL  = 24 h by default.
 *
 * Stages that mutate R2 (which is the case here) need a final write
 * outside the cache, otherwise the second caller of the same key
 * would see the JSON return value but find no R2 objects. Two
 * options:
 *
 *   a) Cache only when the R2 outputs already exist.
 *   b) Cache the return-value JSON which contains R2 keys, and
 *      re-publish R2 outputs on miss; on hit, trust R2 retention.
 *
 * We use (b) — R2 work/ has a 24h lifecycle anyway, so cache TTL
 * matches retention. Hit path = pure JSON return, no R2 writes.
 */

export interface CacheCtx { waitUntil(p: Promise<unknown>): void; }

export async function digestHex(parts: (string | number | boolean | null | undefined)[]): Promise<string> {
  const payload = JSON.stringify(parts);
  const hash = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(payload));
  return [...new Uint8Array(hash)].map(b => b.toString(16).padStart(2, "0")).join("").slice(0, 32);
}

export async function cached<T>(args: {
  stage:    string;
  digest:   string;
  ctx:      CacheCtx;
  ttlSec?:  number;
  compute:  () => Promise<T>;
}): Promise<{ result: T; hit: boolean }> {
  const ttl = args.ttlSec ?? 86400;
  const cache = (globalThis as any).caches?.default;
  // In rare local-dev contexts caches may not be present.
  if (!cache) return { result: await args.compute(), hit: false };

  const key = new Request(`https://cache.internal/${args.stage}/${args.digest}`);
  const hit = await cache.match(key);
  if (hit) {
    const cached = await hit.json<T>();
    return { result: cached, hit: true };
  }

  const result = await args.compute();
  const resp = new Response(JSON.stringify(result), {
    headers: {
      "Content-Type":  "application/json",
      "Cache-Control": `public, max-age=${ttl}`,
    },
  });
  args.ctx.waitUntil(cache.put(key, resp));
  return { result, hit: false };
}
