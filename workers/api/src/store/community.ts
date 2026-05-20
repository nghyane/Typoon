/**
 * Community store — D1 queries for the shared translation feed.
 *
 * Cursor-based pagination on (t.updated_at, t.id) descending.
 */

import type { D1Database } from "@cloudflare/workers-types";
import { decodeCursor, encodeCursor, parseCursorQuery } from "../middleware/pagination";

export interface CommunityFeedEntry {
  translation_id:   number;
  chapter_id:       number;
  chapter_number:   string;
  chapter_label:    string | null;
  work_id:          number;
  material_id:      number;
  title:            string;
  cover:            string | null;
  target_lang:      string;
  creator_id:       number | null;
  creator_name:     string | null;
  created_at:       string | null;
  archive_url:      string | null;
  chapters_in_feed: number;
}

export interface FeedRow {
  translation_id:   number;
  chapter_id:       number;
  chapter_number:   string;
  chapter_label:    string | null;
  work_id:          number;
  material_id:      number;
  title:            string;
  cover:            string | null;
  target_lang:      string;
  creator_id:       number | null;
  creator_name:     string | null;
  created_at:       string | null;
  archive_url:      string | null;
  updated_at:       string;
}

export interface FeedResult {
  items:       CommunityFeedEntry[];
  next_cursor: string | null;
}

export async function listCommunityFeed(
  db: D1Database,
  params: { cursor?: string | null; limit?: number },
): Promise<FeedResult> {
  const { cursor, limit } = parseCursorQuery(params.cursor, params.limit != null ? String(params.limit) : null, 20, 50);

  const { sql: cursorSql, params: cursorParams } = cursor
    ? { sql: " AND (t.updated_at, t.id) < (?, ?)", params: [cursor.ts, cursor.id] }
    : { sql: "", params: [] };

  // Main feed query — shared translations only, ordered by recency
  const { results } = await db
    .prepare(
      `SELECT
         t.id              AS translation_id,
         c.id              AS chapter_id,
         wc.number_norm    AS chapter_number,
         c.label           AS chapter_label,
         wc.work_id,
         c.material_id,
         m.title,
         m.cover_url       AS cover,
         t.target_lang,
         t.owner_id        AS creator_id,
         u.display_name    AS creator_name,
          t.created_at,
          t.updated_at,
          CASE WHEN t.archive_key IS NOT NULL THEN t.archive_key
              WHEN d.archive_key IS NOT NULL THEN d.archive_key
              ELSE NULL END AS archive_url
       FROM translations t
       JOIN translation_drafts d ON d.id = t.draft_id
       JOIN chapters c ON c.id = d.chapter_id
       JOIN work_chapters wc ON wc.id = t.work_chapter_id
       JOIN materials m ON m.id = c.material_id
       LEFT JOIN users u ON u.id = t.owner_id
       WHERE t.shared = 1
         AND t.takedown_at IS NULL
         AND d.state = 'done'
         AND (t.archive_key IS NOT NULL OR d.archive_key IS NOT NULL)
         ${cursorSql}
       ORDER BY t.updated_at DESC, t.id DESC
       LIMIT ?`,
    )
    .bind(...cursorParams, limit)
    .all<FeedRow>();

  if (!results || results.length === 0) {
    return { items: [], next_cursor: null };
  }

  // Enrich with chapters_in_feed count per work
  const workIds = [...new Set(results.map(r => r.work_id))];
  const placeholders = workIds.map(() => "?").join(",");

  const { results: counts } = await db
    .prepare(
      `SELECT wc.work_id, COUNT(DISTINCT t.id) AS chapters_in_feed
       FROM translations t
       JOIN translation_drafts d ON d.id = t.draft_id
       JOIN chapters c ON c.id = d.chapter_id
       JOIN work_chapters wc ON wc.id = t.work_chapter_id
       WHERE t.shared = 1
         AND t.takedown_at IS NULL
         AND d.state = 'done'
         AND wc.work_id IN (${placeholders})
       GROUP BY wc.work_id`,
    )
    .bind(...workIds)
    .all<{ work_id: number; chapters_in_feed: number }>();

  const countMap = new Map(counts.map(c => [c.work_id, c.chapters_in_feed]));

  const items: CommunityFeedEntry[] = results.map(r => ({
    translation_id:   r.translation_id,
    chapter_id:       r.chapter_id,
    chapter_number:   r.chapter_number,
    chapter_label:    r.chapter_label,
    work_id:          r.work_id,
    material_id:      r.material_id,
    title:            r.title,
    cover:            r.cover,
    target_lang:      r.target_lang,
    creator_id:       r.creator_id,
    creator_name:     r.creator_name,
    created_at:       r.created_at,
    archive_url:      r.archive_url,
    chapters_in_feed: countMap.get(r.work_id) ?? 1,
  }));

  const last = results[results.length - 1];
  const next_cursor = !last || items.length < limit
    ? null
    : encodeCursor({ id: last.translation_id, ts: last.updated_at });

  return { items, next_cursor };
}
