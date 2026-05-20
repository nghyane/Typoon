/**
 * Chapters store — D1 queries for chapters table.
 *
 * Replaces postgres.py chapter section.
 */

import type { D1Database } from "@cloudflare/workers-types";
import { NotFoundError, ConflictError } from "./db";

export interface ChapterRow {
  id:              number;
  material_id:     number;
  work_chapter_id: number;
  position:        number;
  label:           string | null;
  upstream_url:    string | null;
  source_lang:     string | null;
  prepared_prefix: string | null;
  masks_prefix:    string | null;
  page_count:      number;
  created_at:      string;
  updated_at:      string;
}

// ── Read ─────────────────────────────────────────────────────────────

export async function getChapter(
  db: D1Database,
  chapter_id: number,
): Promise<ChapterRow | null> {
  return db.prepare("SELECT * FROM chapters WHERE id = ?")
    .bind(chapter_id)
    .first<ChapterRow>();
}

export async function listChapters(
  db: D1Database,
  material_id: number,
): Promise<ChapterRow[]> {
  const { results } = await db
    .prepare("SELECT * FROM chapters WHERE material_id = ? ORDER BY position ASC")
    .bind(material_id)
    .all<ChapterRow>();
  return results;
}

// ── Write ─────────────────────────────────────────────────────────────

interface CreateChapterArgs {
  material_id:     number;
  work_chapter_id: number;
  position:        number;
  label?:          string;
  upstream_url?:   string;
  source_lang?:    string;
}

export async function createChapter(
  db: D1Database,
  args: CreateChapterArgs,
): Promise<ChapterRow> {
  const row = await db
    .prepare(
      `INSERT INTO chapters (material_id, work_chapter_id, position, label, upstream_url, source_lang)
       VALUES (?, ?, ?, ?, ?, ?)
       RETURNING *`,
    )
    .bind(
      args.material_id,
      args.work_chapter_id,
      args.position,
      args.label       ?? null,
      args.upstream_url ?? null,
      args.source_lang  ?? null,
    )
    .first<ChapterRow>();

  if (!row) throw new Error("Chapter insert returned no row");
  return row;
}

export async function setChapterPrepared(
  db: D1Database,
  chapter_id: number,
  prepared_prefix: string,
  page_count: number,
): Promise<void> {
  await db
    .prepare(
      `UPDATE chapters
       SET prepared_prefix = ?, page_count = ?, updated_at = datetime('now')
       WHERE id = ?`,
    )
    .bind(prepared_prefix, page_count, chapter_id)
    .run();
}

export async function setChapterMasks(
  db: D1Database,
  chapter_id: number,
  masks_prefix: string,
): Promise<void> {
  await db
    .prepare("UPDATE chapters SET masks_prefix = ?, updated_at = datetime('now') WHERE id = ?")
    .bind(masks_prefix, chapter_id)
    .run();
}

export async function deleteChapter(
  db: D1Database,
  chapter_id: number,
): Promise<boolean> {
  const result = await db
    .prepare("DELETE FROM chapters WHERE id = ?")
    .bind(chapter_id)
    .run();
  return (result.meta.changes ?? 0) > 0;
}
