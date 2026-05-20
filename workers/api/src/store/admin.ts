/**
 * Admin ops store — D1 queries for pipeline task management.
 *
 * Minimal surface: list tasks, retry, delete.
 */

import type { D1Database } from "@cloudflare/workers-types";
import type { PipelineStage } from "../types";
import { NotFoundError } from "./db";

// ── Stage pause ─────────────────────────────────────────────────────

export interface PausedStage {
  stage:     PipelineStage;
  reason:    string;
  paused_at: string;
  paused_by: string | null;
}

export async function listPausedStages(db: D1Database): Promise<PausedStage[]> {
  const { results } = await db
    .prepare("SELECT stage, reason, paused_at, paused_by FROM stage_pause ORDER BY stage ASC")
    .all<PausedStage>();
  return results;
}

export async function pauseStage(
  db: D1Database,
  stage: string,
  reason: string,
  actorName: string | null,
): Promise<void> {
  await db
    .prepare(
      `INSERT INTO stage_pause (stage, reason, paused_at, paused_by)
       VALUES (?, ?, datetime('now'), ?)
       ON CONFLICT (stage) DO UPDATE
         SET reason = excluded.reason, paused_at = datetime('now'), paused_by = excluded.paused_by`,
    )
    .bind(stage, reason, actorName)
    .run();
}

export async function resumeStage(db: D1Database, stage: string): Promise<void> {
  await db.prepare("DELETE FROM stage_pause WHERE stage = ?").bind(stage).run();
}

// ── Tasks ───────────────────────────────────────────────────────────

export type TaskState = "pending" | "running" | "stale" | "blocked" | "failed";
export type TaskTargetKind = "chapter" | "draft" | "translation";

export interface TaskQueryRow {
  id:             number;
  stage:          string;
  target_kind:    string;
  target_id:      number;
  state:          string;
  attempts:       number;
  claimed_by:     string | null;
  claimed_at:     string | null;
  last_error:     string | null;
  work_id:        number | null;
  chapter_label:  string | null;
  source_lang:    string | null;
  target_lang:    string | null;
  owner_id:       number | null;
}

export interface TaskRow {
  id:             number;
  stage:          PipelineStage;
  target_kind:    TaskTargetKind;
  target_id:      number;
  state:          TaskState;
  attempts:       number;
  claimed_by:     string | null;
  claimed_at:     string | null;
  last_error:     string | null;
  work_id:        number | null;
  chapter_label:  string | null;
  source_lang:    string | null;
  target_lang:    string | null;
  owner_id:       number | null;
}

export async function listTasks(
  db: D1Database,
  filters: {
    stage?: string;
    state?: string;
    limit?: number;
  } = {},
): Promise<TaskRow[]> {
  const where: string[] = [];
  const params: (string | number)[] = [];

  if (filters.stage) { where.push("t.stage = ?"); params.push(filters.stage); }
  if (filters.state) { where.push("t.state = ?");  params.push(filters.state); }

  const whereSql = where.length ? "WHERE " + where.join(" AND ") : "";
  const limit = Math.min(filters.limit ?? 100, 500);

  const { results } = await db
    .prepare(
      `SELECT
         t.id, t.stage, t.target_kind, t.target_id, t.state,
         t.attempts, t.claimed_by, t.claimed_at, t.last_error,
         ctx.work_id, ctx.chapter_label, ctx.source_lang,
         ctx.target_lang, ctx.owner_id
       FROM pipeline_tasks t
       LEFT JOIN task_context ctx ON ctx.stage = t.stage AND ctx.target_id = t.target_id
       ${whereSql}
       ORDER BY t.state ASC, t.attempts DESC, t.claimed_at ASC
       LIMIT ?`,
    )
    .bind(...params, limit)
    .all<TaskQueryRow>();

  return (results || []).map(r => ({
    ...r,
    stage:       r.stage as PipelineStage,
    target_kind: r.target_kind as TaskTargetKind,
    state:       r.state as TaskState,
  }));
}

/** Retry a task: reset state to pending, clear error, keep attempts. */
export async function retryTask(
  db: D1Database,
  taskId: number,
): Promise<void> {
  const result = await db
    .prepare(
      `UPDATE pipeline_tasks
       SET state = 'pending', claimed_by = NULL, claimed_at = NULL, last_error = NULL
       WHERE id = ?`,
    )
    .bind(taskId)
    .run();

  if ((result.meta?.changes ?? 0) === 0) {
    throw new NotFoundError("Task not found");
  }
}

/** Delete a task (typically failed/stale). */
export async function deleteTask(
  db: D1Database,
  taskId: number,
): Promise<void> {
  const result = await db
    .prepare("DELETE FROM pipeline_tasks WHERE id = ?")
    .bind(taskId)
    .run();

  if ((result.meta?.changes ?? 0) === 0) {
    throw new NotFoundError("Task not found");
  }
}

/** Restart a draft regardless of state — admin-only. */
export async function restartDraft(
  db: D1Database,
  draftId: number,
): Promise<void> {
  const draft = await db
    .prepare("SELECT id FROM translation_drafts WHERE id = ?")
    .bind(draftId)
    .first();

  if (!draft) throw new NotFoundError("Draft not found");

  await db
    .prepare(
      `UPDATE translation_drafts
       SET state = 'pending', error_message = NULL,
           progress_stage = NULL, progress_index = NULL, progress_total = NULL,
           updated_at = datetime('now')
       WHERE id = ?`,
    )
    .bind(draftId)
    .run();
}
