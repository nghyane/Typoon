/**
 * Reports — user-submitted abuse reports for moderation queue.
 *
 *   POST /reports        submit (any authenticated user)
 *   GET  /reports        admin: list open reports
 *   PATCH /reports/:id   admin: resolve/dismiss
 */

import { Hono } from "hono";
import type { Env, ContextVars, ReportStatus } from "../types";
import { requireUser } from "../middleware/auth";

type AppEnv = { Bindings: Env; Variables: ContextVars };
const router = new Hono<AppEnv>();

function isAdmin(ctx: { get: (k: string) => unknown }, adminRoleId?: string): boolean {
  const roles = (ctx.get("jwtRoles") as string[]) ?? [];
  return !!adminRoleId && roles.includes(adminRoleId);
}

// ── POST /reports ───────────────────────────────────────────────────

router.post("/", requireUser(), async (ctx) => {
  const userId = ctx.get("userId");
  const body   = await ctx.req.json<{
    job_id?: number;
    reason:  string;
  }>();
  if (!body.reason?.trim()) {
    return ctx.json({ error: "reason required" }, 400);
  }
  await ctx.env.DB
    .prepare(
      `INSERT INTO reports (reporter_id, job_id, reason, status)
       VALUES (?, ?, ?, 'open')`,
    )
    .bind(userId, body.job_id ?? null, body.reason.trim())
    .run();
  return ctx.body(null, 201);
});

// ── GET /reports (admin) ────────────────────────────────────────────

router.get("/", requireUser(), async (ctx) => {
  if (!isAdmin(ctx, ctx.env.ADMIN_ROLE_ID)) {
    return ctx.json({ error: "Forbidden" }, 403);
  }
  const result = await ctx.env.DB
    .prepare(
      `SELECT * FROM reports
       WHERE status = 'open'
       ORDER BY created_at DESC
       LIMIT 100`,
    )
    .all();
  return ctx.json(result.results ?? []);
});

// ── PATCH /reports/:id (admin) ──────────────────────────────────────

router.patch("/:id", requireUser(), async (ctx) => {
  if (!isAdmin(ctx, ctx.env.ADMIN_ROLE_ID)) {
    return ctx.json({ error: "Forbidden" }, 403);
  }
  const id   = Number(ctx.req.param("id"));
  const body = await ctx.req.json<{ status: ReportStatus; resolution?: string }>();
  if (!body.status) return ctx.json({ error: "status required" }, 400);

  await ctx.env.DB
    .prepare(
      `UPDATE reports
       SET status = ?, resolution = ?, resolver_id = ?, resolved_at = datetime('now')
       WHERE id = ?`,
    )
    .bind(body.status, body.resolution ?? null, ctx.get("userId"), id)
    .run();
  return ctx.body(null, 204);
});

export default router;
