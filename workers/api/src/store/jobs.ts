/**
 * Jobs store — D1 queries for the `jobs` table.
 *
 * One job = one chapter translation = one quota unit.
 * Quota math lives here so route handlers stay shape-only.
 */

import type { D1Database } from "@cloudflare/workers-types";
import type { JobState }    from "../types";

export interface JobRow {
  id:               number;
  user_id:          number;
  source_lang:      string;
  target_lang:      string;
  work_id:          string | null;
  kind:             "translate" | "analyze";
  state:            JobState;
  progress_stage:   string | null;
  progress_index:   number | null;
  progress_total:   number | null;
  zip_key:          string | null;
  archive_key:      string | null;
  context_in_key:   string | null;
  context_out_key:  string | null;
  estimated_pages:  number | null;
  page_count:       number | null;
  error_message:    string | null;
  workflow_id:      string | null;
  created_at:       string;
  started_at:       string | null;
  finished_at:      string | null;
  expires_at:       string;
}

const TTL_DAYS = 7;

export async function createJob(
  db: D1Database,
  args: {
    user_id:         number;
    source_lang:     string;
    target_lang:     string;
    estimated_pages: number;
    zip_key:         string;
    work_id?:        string | null;
    kind?:           "translate" | "analyze";
  },
): Promise<JobRow> {
  const row = await db
    .prepare(
      `INSERT INTO jobs (
         user_id, source_lang, target_lang,
         estimated_pages, zip_key, state,
         work_id, kind,
         expires_at
       ) VALUES (?, ?, ?, ?, ?, 'init', ?, ?, datetime('now', '+${TTL_DAYS} days'))
       RETURNING *`,
    )
    .bind(args.user_id, args.source_lang, args.target_lang,
          args.estimated_pages, args.zip_key,
          args.work_id ?? null, args.kind ?? "translate")
    .first<JobRow>();

  if (!row) throw new Error("Job insert returned no row");
  return row;
}

export async function getJob(db: D1Database, id: number): Promise<JobRow | null> {
  return db.prepare("SELECT * FROM jobs WHERE id = ?").bind(id).first<JobRow>();
}

export async function getJobForUser(
  db: D1Database, id: number, user_id: number,
): Promise<JobRow | null> {
  return db
    .prepare("SELECT * FROM jobs WHERE id = ? AND user_id = ?")
    .bind(id, user_id)
    .first<JobRow>();
}

export async function listJobsForUser(
  db: D1Database,
  user_id: number,
  limit = 50,
): Promise<JobRow[]> {
  const result = await db
    .prepare(
      `SELECT * FROM jobs
       WHERE user_id = ?
       ORDER BY created_at DESC
       LIMIT ?`,
    )
    .bind(user_id, limit)
    .all<JobRow>();
  return result.results ?? [];
}

export async function setJobState(
  db: D1Database,
  id: number,
  state: JobState,
  extra: Partial<{
    workflow_id:     string;
    page_count:      number;
    archive_key:     string;
    context_in_key:  string;
    context_out_key: string;
    error_message:   string;
    progress_stage:  string;
    progress_index:  number;
    progress_total:  number;
    started_at:      boolean;   // set to now()
    finished_at:     boolean;   // set to now()
  }> = {},
): Promise<void> {
  const sets: string[] = ["state = ?"];
  const binds: unknown[] = [state];

  if (extra.workflow_id     !== undefined) { sets.push("workflow_id = ?");     binds.push(extra.workflow_id); }
  if (extra.page_count      !== undefined) { sets.push("page_count = ?");      binds.push(extra.page_count); }
  if (extra.archive_key     !== undefined) { sets.push("archive_key = ?");     binds.push(extra.archive_key); }
  if (extra.context_in_key  !== undefined) { sets.push("context_in_key = ?");  binds.push(extra.context_in_key); }
  if (extra.context_out_key !== undefined) { sets.push("context_out_key = ?"); binds.push(extra.context_out_key); }
  if (extra.error_message   !== undefined) { sets.push("error_message = ?");   binds.push(extra.error_message); }
  if (extra.progress_stage  !== undefined) { sets.push("progress_stage = ?");  binds.push(extra.progress_stage); }
  if (extra.progress_index  !== undefined) { sets.push("progress_index = ?");  binds.push(extra.progress_index); }
  if (extra.progress_total  !== undefined) { sets.push("progress_total = ?");  binds.push(extra.progress_total); }
  if (extra.started_at)  sets.push("started_at = datetime('now')");
  if (extra.finished_at) sets.push("finished_at = datetime('now')");

  binds.push(id);
  await db.prepare(`UPDATE jobs SET ${sets.join(", ")} WHERE id = ?`).bind(...binds).run();
}

export async function deleteJob(db: D1Database, id: number, user_id: number): Promise<boolean> {
  const result = await db
    .prepare("DELETE FROM jobs WHERE id = ? AND user_id = ?")
    .bind(id, user_id)
    .run();
  return (result.meta.changes ?? 0) > 0;
}

// ── Quota counters ───────────────────────────────────────────────────

/** Active jobs for concurrency cap.
 *
 *  Stale guard:
 *   - 'init'/'uploading' older than 1h are abandoned (presigned URL TTL
 *     is 1h; client can't resume after that). Don't count them, else
 *     a single failed upload locks the user out until manual delete.
 *   - 'pending'/'running' older than 6h are treated as stuck workflows
 *     (no chapter should take that long). Same reasoning. */
export async function countActiveJobs(
  db: D1Database, user_id: number,
): Promise<number> {
  const row = await db
    .prepare(
      `SELECT COUNT(*) AS n FROM jobs
       WHERE user_id = ?
         AND (
           (state IN ('init','uploading')
              AND created_at >= datetime('now', '-1 hour'))
        OR (state IN ('pending','running')
              AND COALESCE(started_at, created_at) >= datetime('now', '-6 hours'))
         )`,
    )
    .bind(user_id)
    .first<{ n: number }>();
  return row?.n ?? 0;
}

/** Chapters consumed in the current calendar month (UTC). */
export async function countChaptersThisMonth(
  db: D1Database, user_id: number,
): Promise<number> {
  const row = await db
    .prepare(
      `SELECT COUNT(*) AS n FROM chapter_consumes
       WHERE user_id = ?
         AND counted = 1
         AND created_at >= date('now', 'start of month')`,
    )
    .bind(user_id)
    .first<{ n: number }>();
  return row?.n ?? 0;
}

export async function recordConsume(
  db: D1Database,
  args: { user_id: number; job_id: number; page_count: number; counted?: number },
): Promise<void> {
  await db
    .prepare(
      `INSERT INTO chapter_consumes (user_id, job_id, page_count, counted)
       VALUES (?, ?, ?, ?)`,
    )
    .bind(args.user_id, args.job_id, args.page_count, args.counted ?? 1)
    .run();
}
