/**
 * Translations store — D1 queries for translations and translation_edits.
 */

import type { D1Database } from "@cloudflare/workers-types";
import type { DraftState } from "../types";

export interface TranslationRow {
  id:              number;
  work_chapter_id: number;
  owner_id:        number;
  target_lang:     string;
  draft_id:        number;
  shared:          number; // 0=private 1=shared
  archive_key:     string | null;
  rendered_at:     string | null;
  takedown_at:     string | null;
  takedown_reason: string | null;
  created_at:      string;
  updated_at:      string;
}

export interface TranslationEditRow {
  translation_id: number;
  page_index:     number;
  bubble_idx:     number;
  edited_text:    string;
  edited_at:      string;
}

export async function getTranslation(
  db: D1Database,
  translationId: number,
): Promise<TranslationRow | null> {
  return db
    .prepare("SELECT * FROM translations WHERE id = ?")
    .bind(translationId)
    .first<TranslationRow>();
}

export async function getOrCreateTranslation(
  db: D1Database,
  args: {
    work_chapter_id: number;
    owner_id:        number;
    target_lang:     string;
    draft_id:        number;
    shared?:         boolean;
  },
): Promise<TranslationRow> {
  const sharedVal = args.shared !== false ? 1 : 0;
  const row = await db
    .prepare(
      `INSERT INTO translations (work_chapter_id, owner_id, target_lang, draft_id, shared)
       VALUES (?, ?, ?, ?, ?)
       ON CONFLICT (work_chapter_id, owner_id, draft_id) DO UPDATE
         SET shared = translations.shared
       RETURNING *`,
    )
    .bind(args.work_chapter_id, args.owner_id, args.target_lang, args.draft_id, sharedVal)
    .first<TranslationRow>();

  if (!row) throw new Error("Translation insert failed");
  return row;
}

export async function listTranslationsForChapters(
  db: D1Database,
  chapterIds: number[],
): Promise<Record<number, any[]>> {
  const out: Record<number, any[]> = {};
  for (const cid of chapterIds) {
    out[cid] = [];
  }

  if (chapterIds.length === 0) return out;

  // D1 prepare binding arrays uses (?, ?, ?) format.
  const placeholders = chapterIds.map(() => "?").join(",");
  const { results } = await db
    .prepare(
      `SELECT
          c.id AS chapter_id,
          t.id, t.work_chapter_id, t.owner_id, t.target_lang,
          t.draft_id, t.shared,
          CASE
              WHEN d.state IS NULL THEN 'done'
              WHEN d.state = 'done' AND (t.archive_key IS NOT NULL OR d.archive_key IS NOT NULL) THEN 'done'
              WHEN d.state = 'done' THEN 'running'
              ELSE d.state
          END AS state,
          u.display_name AS creator_name,
          (t.archive_key IS NULL AND t.draft_id IS NOT NULL) AS uses_default_render
       FROM chapters c
       JOIN translations t ON t.work_chapter_id = c.work_chapter_id
       LEFT JOIN translation_drafts d ON d.id = t.draft_id
       LEFT JOIN users u ON u.id = t.owner_id
       WHERE c.id IN (${placeholders})
         AND t.takedown_at IS NULL
       ORDER BY c.id, t.created_at DESC`,
    )
    .bind(...chapterIds)
    .all<any>();

  for (const r of results) {
    const cid = Number(r.chapter_id);
    delete r.chapter_id;
    r.shared = r.shared === 1;
    r.uses_default_render = r.uses_default_render === 1;
    const list = out[cid];
    if (list) list.push(r);
  }

  return out;
}

export async function listAllTranslationsForChapter(
  db: D1Database,
  chapterId: number,
): Promise<{ id: number; owner_id: number; archive_key: string | null }[]> {
  const { results } = await db
    .prepare(
      `SELECT t.id, t.owner_id, t.archive_key
       FROM translations t
       JOIN chapters c ON c.work_chapter_id = t.work_chapter_id
       WHERE c.id = ?`,
    )
    .bind(chapterId)
    .all<any>();
  return results;
}

export async function listDraftsForChapter(
  db: D1Database,
  chapterId: number,
): Promise<{ id: number; archive_key: string | null }[]> {
  const { results } = await db
    .prepare("SELECT id, archive_key FROM translation_drafts WHERE chapter_id = ?")
    .bind(chapterId)
    .all<any>();
  return results;
}

export interface MyTranslationRow {
  translation_id:       number;
  target_lang:          string;
  updated_at:           string | null;
  state:                string;
  has_archive:          number;
  chapter_id:           number;
  chapter_number:       string;
  chapter_label:        string | null;
  chapter_position:     number;
  chapter_upstream_url: string | null;
  material_id:          number;
  material_title:       string;
  material_cover:       string | null;
  material_source:      string | null;
  material_upstream_ref: string | null;
}

export async function listMyTranslations(
  db: D1Database,
  userId: number,
): Promise<MyTranslationRow[]> {
  const { results } = await db
    .prepare(
      `SELECT
          t.id              AS translation_id,
          t.target_lang,
          t.updated_at      AS updated_at,
          CASE
              WHEN d.state IS NULL THEN 'done'
              WHEN d.state = 'done' AND (t.archive_key IS NOT NULL OR d.archive_key IS NOT NULL) THEN 'done'
              WHEN d.state = 'done' THEN 'running'
              ELSE d.state
          END AS state,
          (t.archive_key IS NOT NULL OR d.archive_key IS NOT NULL) AS has_archive,
          c.id              AS chapter_id,
          wc.number_norm    AS chapter_number,
          c.label           AS chapter_label,
          c.position        AS chapter_position,
          c.upstream_url    AS chapter_upstream_url,
          m.id              AS material_id,
          m.title           AS material_title,
          m.cover_url       AS material_cover,
          m.source          AS material_source,
          m.upstream_ref    AS material_upstream_ref
       FROM translations t
       JOIN translation_drafts d ON d.id = t.draft_id
       JOIN chapters c ON c.id = d.chapter_id
       JOIN work_chapters wc ON wc.id = c.work_chapter_id
       JOIN materials m ON m.id = c.material_id
       WHERE t.owner_id = ?
         AND t.takedown_at IS NULL
       ORDER BY t.updated_at DESC`,
    )
    .bind(userId)
    .all<MyTranslationRow>();

  return results.map(r => ({
    ...r,
    has_archive: r.has_archive === 1 ? 1 : 0,
  }));
}

export async function updateTranslationArchive(
  db: D1Database,
  translationId: number,
  archiveKey: string,
): Promise<void> {
  await db
    .prepare(
      `UPDATE translations
       SET archive_key = ?, rendered_at = datetime('now'), updated_at = datetime('now')
       WHERE id = ?`,
    )
    .bind(archiveKey, translationId)
    .run();
}

export async function takedownTranslation(
  db: D1Database,
  translationId: number,
  reason: string,
): Promise<void> {
  await db
    .prepare(
      `UPDATE translations
       SET takedown_at = datetime('now'), takedown_reason = ?, updated_at = datetime('now')
       WHERE id = ?`,
    )
    .bind(reason, translationId)
    .run();
}

export async function deleteTranslation(
  db: D1Database,
  translationId: number,
): Promise<boolean> {
  const result = await db
    .prepare("DELETE FROM translations WHERE id = ?")
    .bind(translationId)
    .run();
  return (result.meta.changes ?? 0) > 0;
}

// ── Sparse Edits (Layer 4) ─────────────────────────────────────────────

export async function upsertTranslationEdit(
  db: D1Database,
  args: {
    translation_id: number;
    page_index:     number;
    bubble_idx:     number;
    edited_text:    string;
  },
): Promise<void> {
  await db
    .prepare(
      `INSERT INTO translation_edits (translation_id, page_index, bubble_idx, edited_text)
       VALUES (?, ?, ?, ?)
       ON CONFLICT (translation_id, page_index, bubble_idx) DO UPDATE
         SET edited_text = excluded.edited_text, edited_at = datetime('now')`,
    )
    .bind(args.translation_id, args.page_index, args.bubble_idx, args.edited_text)
    .run();
}

export async function getTranslationEdits(
  db: D1Database,
  translationId: number,
): Promise<TranslationEditRow[]> {
  const { results } = await db
    .prepare(
      `SELECT page_index, bubble_idx, edited_text, edited_at
       FROM translation_edits WHERE translation_id = ?
       ORDER BY page_index, bubble_idx`,
    )
    .bind(translationId)
    .all<TranslationEditRow>();
  return results;
}

export async function deleteTranslationEdit(
  db: D1Database,
  translationId: number,
  pageIndex: number,
  bubbleIdx: number,
): Promise<boolean> {
  const result = await db
    .prepare("DELETE FROM translation_edits WHERE translation_id = ? AND page_index = ? AND bubble_idx = ?")
    .bind(translationId, pageIndex, bubbleIdx)
    .run();
  return (result.meta.changes ?? 0) > 0;
}

// ── Translation detail with joined context ──────────────────────────

export interface TranslationDetail {
  id:               number;
  work_id:          number;
  work_chapter_id:  number;
  chapter_id:       number;       // draft's pixel chapter
  material_id:      number;
  owner_id:         number;
  target_lang:      string;
  draft_id:         number | null;
  state:            DraftState;
  archive_url:      string | null;
  has_edits:        boolean;
  chapter_number:   string | null;
  chapter_label:    string | null;
  material_title:   string | null;
  shared:           boolean;
  created_at:       string | null;
  updated_at:       string | null;
}

export async function getTranslationWithDetails(
  db: D1Database,
  translationId: number,
  userId: number,
): Promise<TranslationDetail | null> {
  const row = await db
    .prepare(
      `SELECT
         t.id,
         wc.work_id,
         t.work_chapter_id,
         c.id AS chapter_id,
         c.material_id,
         t.owner_id,
         t.target_lang,
         t.draft_id,
         d.state,
         CASE WHEN t.archive_key IS NOT NULL THEN t.archive_key
              WHEN d.archive_key IS NOT NULL THEN d.archive_key
              ELSE NULL END AS archive_url,
         (SELECT COUNT(*) > 0 FROM translation_edits e WHERE e.translation_id = t.id) AS has_edits,
         wc.number_norm AS chapter_number,
         c.label AS chapter_label,
         m.title AS material_title,
         t.shared,
         t.created_at,
         t.updated_at
       FROM translations t
       JOIN translation_drafts d ON d.id = t.draft_id
       JOIN chapters c ON c.id = d.chapter_id
       JOIN work_chapters wc ON wc.id = t.work_chapter_id
       JOIN materials m ON m.id = c.material_id
       WHERE t.id = ?
         AND (t.owner_id = ? OR t.shared = 1)
         AND t.takedown_at IS NULL`,
    )
    .bind(translationId, userId)
    .first<TranslationDetail>();

  if (!row) return null;

  // Normalize booleans from D1 integer columns
  row.has_edits = Boolean(row.has_edits);
  row.shared = Boolean(row.shared);

  return row;
}

/** Redo a translation: reset draft to pending, clear error.
 *  Returns the updated draft state. */
export async function redoTranslation(
  db: D1Database,
  translationId: number,
  userId: number,
): Promise<{ draft_id: number; state: DraftState } | null> {
  const t = await db
    .prepare(
      `SELECT t.draft_id, d.state, d.chapter_id, d.source_lang, d.target_lang,
              d.glossary_fp, d.llm_model
       FROM translations t
       JOIN translation_drafts d ON d.id = t.draft_id
       WHERE t.id = ? AND t.owner_id = ? AND t.takedown_at IS NULL`,
    )
    .bind(translationId, userId)
    .first<{
      draft_id: number; state: DraftState; chapter_id: number;
      source_lang: string; target_lang: string;
      glossary_fp: string; llm_model: string;
    }>();

  if (!t) return null;

  // Only allow redo on error/blocked/done states
  if (t.state !== "error" && t.state !== "blocked" && t.state !== "done") {
    return { draft_id: t.draft_id, state: t.state };
  }

  await db
    .prepare(
      `UPDATE translation_drafts
       SET state = 'pending', error_message = NULL,
           progress_stage = NULL, progress_index = NULL, progress_total = NULL,
           updated_at = datetime('now')
       WHERE id = ?`,
    )
    .bind(t.draft_id)
    .run();

  return { draft_id: t.draft_id, state: "pending" };
}
