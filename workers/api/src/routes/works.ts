/**
 * Work-context endpoints — direct KV access for the client.
 *
 *   GET    /works/:work_id/context   gzip body + X-Context-Version header
 *   PUT    /works/:work_id/context   gzip body, If-Match-Version optional
 *   DELETE /works/:work_id/context   wipe
 *
 * The pipeline writes context automatically on finalize; these endpoints
 * are for the client when the user manually edits context in Settings.
 */

import { Hono } from "hono";
import type { Env, ContextVars } from "../types";
import { getContextWithMeta, putContext, deleteContext } from "../store/work-context";

type AppEnv = { Bindings: Env; Variables: ContextVars };
const router = new Hono<AppEnv>();

// Cap on uploaded context body (matches docs/rfc/v3-architecture.md).
const MAX_CONTEXT_BYTES = 256 * 1024;

router.get("/:work_id/context", async (ctx) => {
  const userId = ctx.get("userId");
  const workId = ctx.req.param("work_id");
  const { stream, metadata } = await getContextWithMeta(ctx.env, userId, workId);
  if (!stream) return ctx.json({ error: "Not found" }, 404);

  return new Response(stream, {
    status: 200,
    headers: {
      "Content-Type":      "application/json",
      "Content-Encoding":  "gzip",
      "X-Context-Version": String(metadata?.version ?? 0),
      "X-Updated-At":      metadata?.updated_at ?? "",
      "Cache-Control":     "private, max-age=0, must-revalidate",
    },
  });
});

router.put("/:work_id/context", async (ctx) => {
  const userId = ctx.get("userId");
  const workId = ctx.req.param("work_id");

  const body = await ctx.req.arrayBuffer();
  if (body.byteLength > MAX_CONTEXT_BYTES) {
    return ctx.json({
      error: `Context too large: ${body.byteLength} bytes (max ${MAX_CONTEXT_BYTES})`,
    }, 413);
  }
  if (body.byteLength === 0) {
    return ctx.json({ error: "Empty body" }, 400);
  }

  const ifMatch = ctx.req.header("If-Match-Version");
  const base    = ifMatch !== undefined ? Number(ifMatch) : null;

  const result = await putContext(ctx.env, userId, workId, body, base);
  if ("conflict" in result) {
    return ctx.json({
      error:          "stale base version",
      server_version: result.conflict,
    }, 409);
  }
  return ctx.json({ version: result.version });
});

router.delete("/:work_id/context", async (ctx) => {
  await deleteContext(ctx.env, ctx.get("userId"), ctx.req.param("work_id"));
  return ctx.body(null, 204);
});

export default router;
