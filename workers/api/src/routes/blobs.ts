/**
 * Blobs route — R2 blob proxy for pipeline worker-to-worker calls.
 *
 * These endpoints are worker-facing, not user-facing.
 * Auth: requires "worker" scope in JWT (issued to pipeline service account).
 *
 *   PUT    /api/blobs/:key  → stream body to R2
 *   GET    /api/blobs/:key  → download from R2
 *   HEAD   /api/blobs/:key  → presence check
 *   DELETE /api/blobs/:key  → delete from R2
 *
 * Key safety: reject absolute paths and path traversal.
 */

import { Hono }        from "hono";
import type { Env, ContextVars } from "../types";

const router = new Hono<{ Bindings: Env; Variables: ContextVars }>();

// Worker scope guard — auth is already applied globally; this just checks the scope.
router.use("/*", async (ctx, next) => {
  if (!(ctx.get("jwtRoles") ?? []).includes("worker")) {
    return ctx.json({ error: "Forbidden: worker scope required" }, 403);
  }
  return next();
});

// Extract and validate the R2 key from the Hono wildcard param.
// Hono mounts this router at /api/blobs, so param("*") is everything after that.
function safeKey(ctx: { req: { param: (k: string) => string } }): string | null {
  const key = ctx.req.param("*");
  if (!key || key.startsWith("/") || key.includes("..")) return null;
  return key;
}

// ── PUT /api/blobs/* ──────────────────────────────────────────────────

router.put("/*", async (ctx) => {
  const key = safeKey(ctx);
  if (!key) return ctx.json({ error: "Invalid key" }, 400);

  const body = ctx.req.raw.body;
  if (!body) return ctx.json({ error: "Empty body" }, 400);

  await ctx.env.R2.put(key, body);
  return ctx.body(null, 204);
});

// ── GET /api/blobs/* ──────────────────────────────────────────────────

router.get("/*", async (ctx) => {
  const key = safeKey(ctx);
  if (!key) return ctx.json({ error: "Invalid key" }, 400);

  const object = await ctx.env.R2.get(key);
  if (!object) return ctx.json({ error: "Not found" }, 404);

  return ctx.body(object.body, 200, {
    "Content-Type":  "application/octet-stream",
    "Cache-Control": "no-store",
  });
});

// ── HEAD /api/blobs/* ─────────────────────────────────────────────────

router.on("HEAD", "/*", async (ctx) => {
  const key = safeKey(ctx);
  if (!key) return ctx.body(null, 400);

  const object = await ctx.env.R2.head(key);
  return ctx.body(null, object ? 200 : 404);
});

// ── DELETE /api/blobs/* ───────────────────────────────────────────────

router.delete("/*", async (ctx) => {
  const key = safeKey(ctx);
  if (!key) return ctx.json({ error: "Invalid key" }, 400);

  await ctx.env.R2.delete(key);
  return ctx.body(null, 204);
});

export default router;
