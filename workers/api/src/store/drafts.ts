/**
 * Translation drafts store — D1 queries.
 *
 * Implements the cache-pool logic:
 *   findOrCreateDraft — cache hit (existing draft) vs miss (new spawn)
 *   updateDraftState  — pipeline progress updates
 */

import type { D1Database } from "@cloudflare/workers-types";
import type { DraftState } from "../types";

export interface DraftRow {
  id:              number;
  chapter_id:      number;
  source_lang:     string;
  target_lang:     string;
  glossary_fp:     string;
  llm_model:       string;
  created_by:      number | null;
  state:           DraftState;
  error_message:   string | null;
  progress_stage:  string | null;
  progress_index:  number | null;
  progress_total:  number | null;
  archive_key:     string | null;
  rendered_at:     string | null;
  takedown_at:     string | null;
  created_at:      string;
  updated_at:      string;
}

export interface FindOrCreateResult {
  draft:     DraftRow;
  cache_hit: boolean;
}

// ── Cache-pool lookup ─────────────────────────────────────────────────

export async function findOrCreateDraft(
  db: D1Database,
  args: {
    chapter_id:  number;
    source_lang: string;
    target_lang: string;
    glossary_fp: string;
    llm_model:   string;
    created_by:  number;
  },
): Promise<FindOrCreateResult> {
  const existing = await db
    .prepare(
      `SELECT * FROM translation_drafts
       WHERE chapter_id = ? AND source_lang = ? AND target_lang = ?
         AND glossary_fp = ? AND takedown_at IS NULL`,
    )
    .bind(args.chapter_id, args.source_lang, args.target_lang, args.glossary_fp)
    .first<DraftRow>();

  if (existing) return { draft: existing, cache_hit: true };

  const draft = await db
    .prepare(
      `INSERT INTO translation_drafts
         (chapter_id, source_lang, target_lang, glossary_fp, llm_model, created_by, state)
       VALUES (?, ?, ?, ?, ?, ?, 'pending')
       RETURNING *`,
    )
    .bind(
      args.chapter_id, args.source_lang, args.target_lang,
      args.glossary_fp, args.llm_model, args.created_by,
    )
    .first<DraftRow>();

  if (!draft) throw new Error("Draft insert returned no row");
  return { draft, cache_hit: false };
}

// ── Read ──────────────────────────────────────────────────────────────

export async function getDraft(
  db: D1Database,
  draft_id: number,
): Promise<DraftRow | null> {
  return db
    .prepare("SELECT * FROM translation_drafts WHERE id = ?")
    .bind(draft_id)
    .first<DraftRow>();
}

// ── Write ─────────────────────────────────────────────────────────────

export async function markDraftRunning(
  db: D1Database,
  draft_id: number,
): Promise<void> {
  await db
    .prepare(
      `UPDATE translation_drafts
       SET state = 'running', updated_at = datetime('now') WHERE id = ?`,
    )
    .bind(draft_id)
    .run();
}

export async function updateDraftProgress(
  db: D1Database,
  draft_id: number,
  stage: string,
  index?: number,
  total?: number,
): Promise<void> {
  await db
    .prepare(
      `UPDATE translation_drafts
       SET state = 'running', progress_stage = ?,
           progress_index = ?, progress_total = ?, updated_at = datetime('now')
       WHERE id = ?`,
    )
    .bind(stage, index ?? null, total ?? null, draft_id)
    .run();
}

export async function markDraftDone(
  db: D1Database,
  draft_id: number,
  archive_key: string,
): Promise<void> {
  await db
    .prepare(
      `UPDATE translation_drafts
       SET state = 'done', archive_key = ?, rendered_at = datetime('now'),
           updated_at = datetime('now')
       WHERE id = ?`,
    )
    .bind(archive_key, draft_id)
    .run();
}

export async function markDraftError(
  db: D1Database,
  draft_id: number,
  message: string,
): Promise<void> {
  await db
    .prepare(
      `UPDATE translation_drafts
       SET state = 'error', error_message = ?, updated_at = datetime('now')
       WHERE id = ?`,
    )
    .bind(message, draft_id)
    .run();
}

// ── Glossary fingerprint ───────────────────────────────────────────────
// 16-char hex prefix of SHA-256 over sorted "source:target" pairs.
// Used as cache-pool key — same glossary → same draft.
// Returns "0000000000000000" when the user has no glossary terms.

export async function computeGlossaryFp(
  db: D1Database,
  userId: number,
  source_lang: string,
  target_lang: string,
): Promise<string> {
  const { results } = await db
    .prepare(
      `SELECT source_term, target_term FROM user_glossary
       WHERE owner_id = ? AND source_lang = ? AND target_lang = ?
       ORDER BY source_term ASC`,
    )
    .bind(userId, source_lang, target_lang)
    .all<{ source_term: string; target_term: string }>();

  if (results.length === 0) return "0".repeat(16);

  const payload = results.map(r => `${r.source_term}:${r.target_term}`).join("|");
  const buf     = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(payload));
  const hex     = Array.from(new Uint8Array(buf))
    .map(b => b.toString(16).padStart(2, "0")).join("");
  return hex.slice(0, 16);
}
