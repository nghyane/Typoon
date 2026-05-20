/** Container fan-out: shard tiles across container instances + cache + retry.
 *
 * The inpaint container's `getRandom(ns, N)` distributes calls across N warm
 * instances. We further bundle our tiles into "shards" so:
 *   - Tiles inside one shard run serially against the SAME container instance
 *     → amortise the cold start across multiple tiles
 *   - Shards themselves fan out in parallel via Promise.all
 *
 * Without sharding, each tile call would prefer a fresh container instance
 * and pay the ~5s cold start individually. With shards, only the first tile
 * in a shard pays it; siblings ride the warm session.
 *
 * R2 cache: tile inference is deterministic over the EXACT body bytes
 * (RGB padded + binary mask). Same chapter re-run, or two chapters that
 * crop identical sub-tiles, get a free hit. */

import { TILES_PER_SHARD, MAX_SHARDS_PER_PAGE } from "./constants";
import { composeTile, type Tile } from "./tile";

/** Inpaint container RPC stub. */
export interface InpaintStub {
  inpaintTile(body: Uint8Array, W: number, H: number): Promise<Uint8Array>;
}

export interface InpaintEnv {
  R2:      R2Bucket;
  INPAINT: InpaintStub;
}

async function tileHashHex(body: Uint8Array): Promise<string> {
  const h = await crypto.subtle.digest("SHA-256", body);
  return [...new Uint8Array(h)].map(b => b.toString(16).padStart(2, "0")).join("").slice(0, 32);
}

/** Call container for one tile with R2 cache + retry. */
async function callTileWithRetry(
  env: InpaintEnv, ctx: ExecutionContext, tile: Tile, attempts = 3,
): Promise<Uint8Array> {
  const hash     = await tileHashHex(tile.body);
  const cacheKey = `inpaint-tiles/${tile.W}x${tile.H}/${hash}.bin`;
  const hit      = await env.R2.get(cacheKey);
  if (hit) return new Uint8Array(await hit.arrayBuffer());

  let lastErr: unknown;
  for (let i = 0; i < attempts; i++) {
    try {
      const out = await env.INPAINT.inpaintTile(tile.body, tile.W, tile.H);
      // Write-through cache; doesn't block the response.
      ctx.waitUntil(env.R2.put(cacheKey, out, {
        httpMetadata: { contentType: "application/octet-stream" },
      }));
      return out;
    } catch (e) {
      lastErr = e;
      const msg = (e instanceof Error ? e.message : String(e)).toLowerCase();
      const retriable = msg.includes("container") || msg.includes("network")
        || msg.includes("reset") || msg.includes("502") || msg.includes("503");
      if (!retriable) throw e;
      if (i < attempts - 1) await new Promise(r => setTimeout(r, 500 * (i + 1)));
    }
  }
  throw lastErr;
}

/** Run one shard's tiles serially, composing each result into `composite`
 *  immediately so only one tile output buffer lives at a time. */
async function runShardInto(
  env: InpaintEnv, ctx: ExecutionContext,
  tiles: Tile[],
  composite: Uint8Array, srcW: number, origMask: Uint8Array,
): Promise<void> {
  for (const tile of tiles) {
    const tileRgb = await callTileWithRetry(env, ctx, tile);
    composeTile(composite, srcW, origMask, tile, tileRgb);
  }
}

/** Shard tiles across container instances and compose results into `composite`.
 *  Mutates `composite` in place. */
export async function shardAndCompose(
  env: InpaintEnv, ctx: ExecutionContext,
  tiles: Tile[],
  composite: Uint8Array, srcW: number, origMask: Uint8Array,
): Promise<void> {
  if (tiles.length === 0) return;

  const shardCount = Math.min(
    MAX_SHARDS_PER_PAGE,
    Math.max(1, Math.ceil(tiles.length / TILES_PER_SHARD)),
  );

  // Round-robin distribute tiles → even bucket sizes per shard
  const shards: Tile[][] = Array.from({ length: shardCount }, () => []);
  for (let i = 0; i < tiles.length; i++) shards[i % shardCount].push(tiles[i]);

  await Promise.all(shards.map(shard =>
    runShardInto(env, ctx, shard, composite, srcW, origMask),
  ));
}
