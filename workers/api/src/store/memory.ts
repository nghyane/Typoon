/**
 * Memory store — D1 queries for translator memory and briefs.
 */

import type { D1Database } from "@cloudflare/workers-types";

export interface TranslatorMemoryRow {
  id:              number;
  user_id:         number;
  material_id:     number;
  source_lang:     string;
  target_lang:     string;
  characters:      any; // parsed JSON
  world:           any; // parsed JSON
  style:           any; // parsed JSON
  glossary:        any; // parsed JSON
  style_refs:      any; // parsed JSON
  last_chapter_id: number | null;
  created_at:      string;
  updated_at:      string;
}

export interface MemoryBriefRow {
  chapter_id:  number;
  position:    number;
  number:      string;
  label:       string | null;
  brief_json:  any; // parsed JSON
  summary:     string | null;
  created_at:  string;
  updated_at:  string;
}

export async function getTranslatorMemory(
  db: D1Database,
  args: {
    user_id:     number;
    material_id: number;
    target_lang: string;
  },
): Promise<TranslatorMemoryRow | null> {
  const row = await db
    .prepare(
      `SELECT * FROM translator_memory
       WHERE user_id = ? AND material_id = ? AND target_lang = ?`,
    )
    .bind(args.user_id, args.material_id, args.target_lang)
    .first<any>();

  if (!row) return null;

  return {
    ...row,
    characters: typeof row.characters === "string" ? JSON.parse(row.characters) : row.characters,
    world:      typeof row.world === "string" ? JSON.parse(row.world) : row.world,
    style:      typeof row.style === "string" ? JSON.parse(row.style) : row.style,
    glossary:   typeof row.glossary === "string" ? JSON.parse(row.glossary) : row.glossary,
    style_refs: typeof row.style_refs === "string" ? JSON.parse(row.style_refs) : row.style_refs,
  };
}

export async function upsertTranslatorMemory(
  db: D1Database,
  args: {
    user_id:     number;
    material_id: number;
    source_lang: string;
    target_lang: string;
    characters?: any[] | null;
    world?:      Record<string, any> | null;
    style?:      Record<string, any> | null;
    glossary?:   any[] | null;
    style_refs?: any[] | null;
  },
): Promise<TranslatorMemoryRow> {
  // Read existing
  const existing = await getTranslatorMemory(db, {
    user_id:     args.user_id,
    material_id: args.material_id,
    target_lang: args.target_lang,
  });

  const finalChars = args.characters !== undefined && args.characters !== null ? args.characters : (existing?.characters ?? []);
  const finalWorld = args.world !== undefined && args.world !== null ? args.world : (existing?.world ?? {});
  const finalStyle = args.style !== undefined && args.style !== null ? args.style : (existing?.style ?? {});
  const finalGloss = args.glossary !== undefined && args.glossary !== null ? args.glossary : (existing?.glossary ?? []);
  const finalRefs  = args.style_refs !== undefined && args.style_refs !== null ? args.style_refs : (existing?.style_refs ?? []);

  const row = await db
    .prepare(
      `INSERT INTO translator_memory (
         user_id, material_id, source_lang, target_lang,
         characters, world, style, glossary, style_refs
       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT (user_id, material_id, target_lang) DO UPDATE SET
         source_lang = excluded.source_lang,
         characters = excluded.characters,
         world = excluded.world,
         style = excluded.style,
         glossary = excluded.glossary,
         style_refs = excluded.style_refs,
         updated_at = datetime('now')
       RETURNING *`,
    )
    .bind(
      args.user_id,
      args.material_id,
      args.source_lang,
      args.target_lang,
      JSON.stringify(finalChars),
      JSON.stringify(finalWorld),
      JSON.stringify(finalStyle),
      JSON.stringify(finalGloss),
      JSON.stringify(finalRefs),
    )
    .first<any>();

  if (!row) throw new Error("Upsert translator memory failed");

  return {
    ...row,
    characters: typeof row.characters === "string" ? JSON.parse(row.characters) : row.characters,
    world:      typeof row.world === "string" ? JSON.parse(row.world) : row.world,
    style:      typeof row.style === "string" ? JSON.parse(row.style) : row.style,
    glossary:   typeof row.glossary === "string" ? JSON.parse(row.glossary) : row.glossary,
    style_refs: typeof row.style_refs === "string" ? JSON.parse(row.style_refs) : row.style_refs,
  };
}

export async function deleteTranslatorMemory(
  db: D1Database,
  args: {
    user_id:     number;
    material_id: number;
    target_lang: string;
  },
): Promise<boolean> {
  const result = await db
    .prepare("DELETE FROM translator_memory WHERE user_id = ? AND material_id = ? AND target_lang = ?")
    .bind(args.user_id, args.material_id, args.target_lang)
    .run();

  return (result.meta?.changes ?? 0) > 0;
}

export async function listRecentMemoryBriefs(
  db: D1Database,
  args: {
    memory_id:         number;
    before_chapter_id?: number | null;
    limit?:            number;
  },
): Promise<MemoryBriefRow[]> {
  const limit = args.limit ?? 5;
  let query = `
    SELECT b.chapter_id, c.position, wc.number_norm AS number, c.label,
           b.brief_json, b.summary, b.created_at, b.updated_at
    FROM   translator_memory_briefs b
    JOIN   chapters c ON c.id = b.chapter_id
    JOIN   work_chapters wc ON wc.id = c.work_chapter_id
    WHERE  b.memory_id = ?
  `;
  const params: any[] = [args.memory_id];

  if (args.before_chapter_id !== undefined && args.before_chapter_id !== null) {
    params.push(args.before_chapter_id);
    query += ` AND c.position < (SELECT position FROM chapters WHERE id = ?)`;
  }

  query += ` ORDER BY c.position DESC LIMIT ?`;
  params.push(limit);

  const { results } = await db.prepare(query).bind(...params).all<any>();

  return results.map(r => ({
    ...r,
    brief_json: typeof r.brief_json === "string" ? JSON.parse(r.brief_json) : r.brief_json,
  }));
}
