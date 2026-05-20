/**
 * Admin routes — stages, tasks, draft restart.
 * Reports router is exported separately (accessible to any authenticated user).
 *
 * Admin (requireAdmin):
 *   GET    /api/admin/stages              → list paused stages
 *   POST   /api/admin/stages/:name/pause  → pause a stage
 *   POST   /api/admin/stages/:name/resume → resume a stage
 *   GET    /api/admin/tasks               → list pipeline tasks
 *   POST   /api/admin/tasks/:id/retry     → retry a task
 *   DELETE /api/admin/tasks/:id           → delete a task
 *   POST   /api/admin/drafts/:id/restart  → restart a draft
 *
 * Reports (any authenticated user):
 *   POST   /api/reports                   → submit a report
 */

import { Hono } from "hono";
import type { Env, ContextVars } from "../types";
import { requireAdmin } from "../middleware/auth";
import {
  listPausedStages,
  pauseStage,
  resumeStage,
  listTasks,
  retryTask,
  deleteTask,
  restartDraft,
} from "../store/admin";

type AppEnv = { Bindings: Env; Variables: ContextVars };

// ── Reports (any authenticated user) ────────────────────────────────

export const reportsRouter = new Hono<AppEnv>();

reportsRouter.post("", async (ctx) => {
  const userId = ctx.get("userId");

  const user = await ctx.env.DB
    .prepare("SELECT display_name FROM users WHERE id = ?")
    .bind(userId)
    .first<{ display_name: string }>();

  const displayName = user?.display_name ?? `user:${userId}`;

  const body = await ctx.req.json<{
    target_kind: "material" | "chapter" | "draft" | "translation";
    target_id:   number;
    kind?:       "dmca" | "abuse" | "quality" | "other";
    reason:      string;
  }>();

  if (!body.target_kind || !body.target_id || !body.reason) {
    return ctx.json(
      { error: { code: "bad_request", message: "target_kind, target_id and reason required" } },
      400,
    );
  }

  const kind = body.kind ?? "dmca";

  const row = await ctx.env.DB
    .prepare(
      `INSERT INTO reports (reporter_id, reporter_label, target_kind, target_id, kind, reason)
       VALUES (?, ?, ?, ?, ?, ?)
       RETURNING id`,
    )
    .bind(userId, displayName, body.target_kind, body.target_id, kind, body.reason)
    .first<{ id: number }>();

  if (!row) {
    return ctx.json(
      { error: { code: "internal_error", message: "Failed to submit report" } },
      500,
    );
  }

  return ctx.json({ report_id: row.id }, 202);
});

// ── Admin ops (require admin role) ──────────────────────────────────

export const adminRouter = new Hono<AppEnv>();
adminRouter.use("/*", requireAdmin());

// GET /stages — list paused stages
adminRouter.get("/stages", async (ctx) => {
  const stages = await listPausedStages(ctx.env.DB);
  return ctx.json(stages);
});

// POST /stages/:name/pause
adminRouter.post("/stages/:name/pause", async (ctx) => {
  const name = ctx.req.param("name");
  const body = await ctx.req.json<{ reason: string }>();

  if (!body.reason) {
    return ctx.json(
      { error: { code: "bad_request", message: "reason required" } },
      400,
    );
  }

  const actorName = String(ctx.get("userId"));
  await pauseStage(ctx.env.DB, name, body.reason, actorName);
  return ctx.body(null, 204);
});

// POST /stages/:name/resume
adminRouter.post("/stages/:name/resume", async (ctx) => {
  const name = ctx.req.param("name");
  await resumeStage(ctx.env.DB, name);
  return ctx.body(null, 204);
});

// GET /tasks — list pipeline tasks
adminRouter.get("/tasks", async (ctx) => {
  const stage = ctx.req.query("stage") ?? undefined;
  const state = ctx.req.query("state") ?? undefined;

  const tasks = await listTasks(ctx.env.DB, { stage, state });
  return ctx.json(tasks);
});

// POST /tasks/:id/retry
adminRouter.post("/tasks/:id/retry", async (ctx) => {
  const id = Number(ctx.req.param("id"));
  if (!id || isNaN(id)) {
    return ctx.json(
      { error: { code: "bad_request", message: "Invalid task ID" } },
      400,
    );
  }

  await retryTask(ctx.env.DB, id);
  return ctx.body(null, 204);
});

// DELETE /tasks/:id
adminRouter.delete("/tasks/:id", async (ctx) => {
  const id = Number(ctx.req.param("id"));
  if (!id || isNaN(id)) {
    return ctx.json(
      { error: { code: "bad_request", message: "Invalid task ID" } },
      400,
    );
  }

  await deleteTask(ctx.env.DB, id);
  return ctx.body(null, 204);
});

// POST /drafts/:id/restart
adminRouter.post("/drafts/:id/restart", async (ctx) => {
  const id = Number(ctx.req.param("id"));
  if (!id || isNaN(id)) {
    return ctx.json(
      { error: { code: "bad_request", message: "Invalid draft ID" } },
      400,
    );
  }

  await restartDraft(ctx.env.DB, id);
  return ctx.body(null, 204);
});
