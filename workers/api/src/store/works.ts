/**
 * Works store — D1 queries for logical works, redirects, and Logical chapters.
 */

import type { D1Database } from "@cloudflare/workers-types";
import { NotFoundError } from "./db";

export interface WorkRow {
  id:         number;
  cross_refs: string | null; // JSON string
  created_at: string;
  updated_at: string;
}

export interface WorkChapterRow {
  id:          number;
  work_id:     number;
  number_norm: string;
  label:       string | null;
  created_at:  string;
}

export async function getWork(
  db: D1Database,
  workId: number,
): Promise<WorkRow | null> {
  return db
    .prepare("SELECT * FROM works WHERE id = ?")
    .bind(workId)
    .first<WorkRow>();
}

export async function getWorkRedirect(
  db: D1Database,
  oldId: number,
): Promise<number | null> {
  const row = await db
    .prepare("SELECT new_id FROM work_redirects WHERE old_id = ?")
    .bind(oldId)
    .first<{ new_id: number }>();
  return row ? row.new_id : null;
}

export async function createWork(
  db: D1Database,
  crossRefs: Record<string, any> | null = null,
): Promise<WorkRow> {
  const row = await db
    .prepare("INSERT INTO works (cross_refs) VALUES (?) RETURNING *")
    .bind(crossRefs ? JSON.stringify(crossRefs) : null)
    .first<WorkRow>();

  if (!row) throw new Error("Work insert failed");
  return row;
}

export async function createBlankWork(
  db: D1Database,
  args: {
    user_id:     number;
    title:       string;
    cover_url?:  string | null;
    target_lang: string;
  },
): Promise<{ work_id: number; entry_id: number }> {
  // Step 1: Create Work
  const work = await createWork(db);
  const workId = work.id;

  // Step 2: Create placeholder upload-origin Material
  const material = await db
    .prepare(
      `INSERT INTO materials (imported_by, origin, work_id, title, cover_url, languages)
       VALUES (?, 'upload', ?, ?, ?, ?)
       RETURNING id`,
    )
    .bind(
      args.user_id,
      workId,
      args.title,
      args.cover_url ?? null,
      JSON.stringify(["vi"]),
    )
    .first<{ id: number }>();

  if (!material) throw new Error("Blank work material creation failed");

  // Step 3: Create Library Entry
  const entry = await db
    .prepare(
      `INSERT INTO library_entries (user_id, work_id, target_lang, status)
       VALUES (?, ?, ?, 'reading')
       RETURNING id`,
    )
    .bind(args.user_id, workId, args.target_lang)
    .first<{ id: number }>();

  if (!entry) throw new Error("Blank work library entry creation failed");

  return { work_id: workId, entry_id: entry.id };
}

export async function deleteUserWork(
  db: D1Database,
  workId: number,
  userId: number,
): Promise<void> {
  // Assert no source materials exist for this work
  const sourceCheck = await db
    .prepare("SELECT id FROM materials WHERE work_id = ? AND origin = 'source' LIMIT 1")
    .bind(workId)
    .first();

  if (sourceCheck) {
    throw new Error("Cannot delete a work with source-backed materials");
  }

  // Perform cascade deletion
  await db.prepare("DELETE FROM works WHERE id = ?").bind(workId).run();
}

export interface WorkChapterTranslation {
  id:                  number;
  target_lang:         string;
  source_lang:         string | null;
  owner_id:            number;
  creator_name:        string | null;
  state:               string;
  error_message:       string | null;
  shared:              boolean;
  draft_id:            number | null;
  draft_chapter_id:    number | null;
  draft_material_id:   number | null;
  uses_default_render: boolean;
  updated_at:          string | null;
}

export interface UploadingChapter {
  chapter_id:  number;
  material_id: number;
  source_lang: string | null;
  uploaded_by: number;
  created_at:  string | null;
}

export interface LogicalChapter {
  id:                 number;
  number_norm:        string;
  label:              string | null;
  translations:       WorkChapterTranslation[];
  uploading_chapters: UploadingChapter[];
}

export async function listWorkChaptersWithTranslations(
  db: D1Database,
  workId: number,
  viewerId: number,
): Promise<LogicalChapter[]> {
  // SQLite Left Join mapping mirroring Postgres
  const { results } = await db
    .prepare(
      `SELECT
          wc.id            AS work_chapter_id,
          wc.number_norm,
          wc.label,
          wc.created_at    AS wc_created_at,
          -- translation columns (NULL when no translation)
          t.id             AS translation_id,
          t.target_lang,
          t.owner_id,
          t.shared,
          (t.archive_key IS NULL AND t.draft_id IS NOT NULL) AS uses_default_render,
          CASE
              WHEN d.state IS NULL THEN 'done'
              WHEN d.state = 'done' AND (t.archive_key IS NOT NULL OR d.archive_key IS NOT NULL) THEN 'done'
              WHEN d.state = 'done' THEN 'running'
              ELSE d.state
          END AS state,
          d.error_message  AS error_message,
          t.draft_id,
          d.source_lang    AS draft_source_lang,
          d.chapter_id     AS draft_chapter_id,
          c_draft.material_id AS draft_material_id,
          u.display_name   AS creator_name,
          t.updated_at     AS t_updated_at,
          -- upload-pending columns (NULL when no pending upload)
          c_up.id          AS up_chapter_id,
          c_up.material_id AS up_material_id,
          c_up.source_lang AS up_source_lang,
          m_up.imported_by AS up_uploaded_by,
          c_up.created_at  AS up_created_at
       FROM work_chapters wc
       LEFT JOIN translations t
         ON t.work_chapter_id = wc.id
        AND t.takedown_at IS NULL
        AND (t.shared = 1 OR t.owner_id = ?)
       LEFT JOIN translation_drafts d ON d.id = t.draft_id
       LEFT JOIN chapters c_draft ON c_draft.id = d.chapter_id
       LEFT JOIN users u ON u.id = t.owner_id
       -- pending upload chapters: upload-origin, not yet prepared,
       -- belonging to the viewer's upload material on this work.
       LEFT JOIN chapters c_up
         ON c_up.work_chapter_id = wc.id
        AND c_up.prepared_prefix IS NULL
       LEFT JOIN materials m_up
         ON m_up.id = c_up.material_id
        AND m_up.origin = 'upload'
        AND m_up.work_id = wc.work_id
        AND m_up.imported_by = ?
       WHERE wc.work_id = ?
       ORDER BY wc.id, t.created_at DESC, c_up.created_at DESC`,
    )
    .bind(viewerId, viewerId, workId)
    .all<any>();

  const byWc = new Map<number, LogicalChapter>();

  for (const r of results) {
    const wcId = r.work_chapter_id;
    let chapter = byWc.get(wcId);
    if (!chapter) {
      chapter = {
        id:                 wcId,
        number_norm:        r.number_norm,
        label:              r.label,
        translations:       [],
        uploading_chapters: [],
      };
      byWc.set(wcId, chapter);
    }

    // --- translation ---
    if (r.translation_id !== null && r.translation_id !== undefined) {
      const tr: WorkChapterTranslation = {
        id:                  Number(r.translation_id),
        target_lang:         r.target_lang,
        source_lang:         r.draft_source_lang,
        owner_id:            Number(r.owner_id),
        creator_name:        r.creator_name,
        state:               r.state,
        error_message:       r.error_message,
        shared:              r.shared === 1,
        draft_id:            r.draft_id ? Number(r.draft_id) : null,
        draft_chapter_id:    r.draft_chapter_id ? Number(r.draft_chapter_id) : null,
        draft_material_id:   r.draft_material_id ? Number(r.draft_material_id) : null,
        uses_default_render: r.uses_default_render === 1,
        updated_at:          r.t_updated_at,
      };

      if (!chapter.translations.some(x => x.id === tr.id)) {
        chapter.translations.push(tr);
      }
    }

    // --- pending upload ---
    if (r.up_chapter_id !== null && r.up_chapter_id !== undefined && r.up_uploaded_by !== null && r.up_uploaded_by !== undefined) {
      const up: UploadingChapter = {
        chapter_id:  Number(r.up_chapter_id),
        material_id: Number(r.up_material_id),
        source_lang: r.up_source_lang,
        uploaded_by: Number(r.up_uploaded_by),
        created_at:  r.up_created_at,
      };

      if (!chapter.uploading_chapters.some(x => x.chapter_id === up.chapter_id)) {
        chapter.uploading_chapters.push(up);
      }
    }
  }

  // Parse natural float numbers and sort logically descending (similar to Python logic)
  const tryFloat = (val: string): number | null => {
    const f = parseFloat(val);
    return isNaN(f) ? null : f;
  };

  return Array.from(byWc.values()).sort((a, b) => {
    const na = tryFloat(a.number_norm);
    const nb = tryFloat(b.number_norm);
    if (na !== null && nb !== null) {
      return nb - na; // descending norm
    }
    if (na !== null) return -1;
    if (nb !== null) return 1;
    return b.number_norm.localeCompare(a.number_norm);
  });
}

export async function findOrCreateWorkChapter(
  db: D1Database,
  args: {
    work_id:     number;
    number_norm: string;
    label?:      string | null;
  },
): Promise<number> {
  const existing = await db
    .prepare("SELECT id FROM work_chapters WHERE work_id = ? AND number_norm = ?")
    .bind(args.work_id, args.number_norm)
    .first<{ id: number }>();

  if (existing) return existing.id;

  const result = await db
    .prepare(
      `INSERT INTO work_chapters (work_id, number_norm, label)
       VALUES (?, ?, ?)
       ON CONFLICT (work_id, number_norm) DO UPDATE SET id=id
       RETURNING id`,
    )
    .bind(args.work_id, args.number_norm, args.label ?? null)
    .first<{ id: number }>();

  if (!result) {
    const fallback = await db
      .prepare("SELECT id FROM work_chapters WHERE work_id = ? AND number_norm = ?")
      .bind(args.work_id, args.number_norm)
      .first<{ id: number }>();
    if (!fallback) throw new Error("Work chapter creation failed");
    return fallback.id;
  }
  return result.id;
}

