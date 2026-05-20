/**
 * Glossary store — D1 queries for user glossary terms.
 */

import type { D1Database } from "@cloudflare/workers-types";

export interface GlossaryTermRow {
  id:          number;
  owner_id:    number;
  source_lang: string;
  target_lang: string;
  source_term: string;
  target_term: string;
  notes:       string | null;
}

export async function listUserGlossary(
  db: D1Database,
  userId: number,
  sourceLang?: string | null,
  targetLang?: string | null,
): Promise<GlossaryTermRow[]> {
  let query = "SELECT * FROM user_glossary WHERE owner_id = ?";
  const params: any[] = [userId];

  if (sourceLang) {
    query += " AND source_lang = ?";
    params.push(sourceLang);
  }
  if (targetLang) {
    query += " AND target_lang = ?";
    params.push(targetLang);
  }
  query += " ORDER BY source_term ASC";

  const { results } = await db.prepare(query).bind(...params).all<GlossaryTermRow>();
  return results;
}

export async function upsertUserGlossaryTerm(
  db: D1Database,
  args: {
    user_id:     number;
    source_lang: string;
    target_lang: string;
    source_term: string;
    target_term: string;
    notes?:      string | null;
  },
): Promise<number> {
  const row = await db
    .prepare(
      `INSERT INTO user_glossary (owner_id, source_lang, target_lang, source_term, target_term, notes)
       VALUES (?, ?, ?, ?, ?, ?)
       ON CONFLICT (owner_id, source_lang, target_lang, source_term) DO UPDATE
       SET target_term = excluded.target_term, notes = excluded.notes
       RETURNING id`,
    )
    .bind(
      args.user_id,
      args.source_lang,
      args.target_lang,
      args.source_term.trim(),
      args.target_term.trim(),
      args.notes ?? null,
    )
    .first<{ id: number }>();

  if (!row) throw new Error("Upsert glossary term failed");
  return row.id;
}

export async function deleteUserGlossaryTerm(
  db: D1Database,
  userId: number,
  termId: number,
): Promise<boolean> {
  const result = await db
    .prepare("DELETE FROM user_glossary WHERE id = ? AND owner_id = ?")
    .bind(termId, userId)
    .run();

  return (result.meta?.changes ?? 0) > 0;
}
