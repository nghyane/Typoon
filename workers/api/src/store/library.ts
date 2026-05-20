/**
 * Library store — D1 queries for library entries, materials bookmarks, and reading history.
 */

import type { D1Database } from "@cloudflare/workers-types";
import { NotFoundError } from "./db";

export interface LibraryMaterialLink {
  material_id: number;
  link_origin: "auto" | "manual";
  linked_at:   string;
}

export interface LibraryEntryRow {
  id:          number;
  user_id:     number;
  work_id:     number;
  target_lang: string;
  status:      "reading" | "plan" | "done" | "dropped";
  created_at:  string;
  updated_at:  string;
}

export interface LibraryEntryDetail extends LibraryEntryRow {
  title:               string;
  cover:               string | null;
  materials:           LibraryMaterialLink[];
  translation_summary: {
    pending: number;
    running: number;
    done:    number;
    error:   number;
  };
}

// ── Title / Cover display resolvers ──────────────────────────────────

function normLang(l: any): string | null {
  if (!l) return null;
  const parts = String(l).toLowerCase().split(/[-_]/);
  const s = parts[0]?.trim() || "";
  return s || null;
}

function resolveWorkTitle(materials: any[], target: string | null): string {
  if (!materials || materials.length === 0) return "";
  if (target) {
    for (const m of materials) {
      let locale: any = {};
      if (typeof m.title_locale === "string") {
        try { locale = JSON.parse(m.title_locale) || {}; } catch {}
      } else {
        locale = m.title_locale || {};
      }
      const t = String(locale[target] || "").trim();
      if (t) return t;
    }
    for (const m of materials) {
      let langs: string[] = [];
      if (typeof m.languages === "string") {
        try { langs = JSON.parse(m.languages) || []; } catch {}
      } else {
        langs = m.languages || [];
      }
      if (langs.some(l => normLang(l) === target)) {
        const t = String(m.title || "").trim();
        if (t) return t;
      }
    }
  }
  for (const m of materials) {
    const t = String(m.title_native || "").trim();
    if (t) return t;
  }
  return String(materials[0].title || "").trim();
}

function resolveWorkCover(materials: any[], target: string | null): string | null {
  if (!materials || materials.length === 0) return null;

  const sourceMats = materials.filter(m => m.origin === "source");
  const otherMats  = materials.filter(m => m.origin !== "source");

  if (target) {
    for (const m of sourceMats) {
      let langs: string[] = [];
      if (typeof m.languages === "string") {
        try { langs = JSON.parse(m.languages) || []; } catch {}
      } else {
        langs = m.languages || [];
      }
      if (langs.some(l => normLang(l) === target) && m.cover_url) {
        return m.cover_url;
      }
    }
  }

  for (const m of sourceMats) {
    if (m.cover_url) return m.cover_url;
  }

  for (const m of otherMats) {
    if (m.cover_url) return m.cover_url;
  }

  return null;
}

export async function resolveWorkDisplay(
  db: D1Database,
  rows: any[],
  viewerId: number,
): Promise<void> {
  if (!rows || rows.length === 0) return;
  const workIds = Array.from(new Set(rows.map(r => Number(r.work_id)).filter(Boolean)));
  if (workIds.length === 0) return;

  // Fetch materials for these works
  const { results: matRows } = await db
    .prepare(
      `SELECT id, work_id, origin, title, cover_url, title_native, languages, title_locale
       FROM materials WHERE work_id IN (${workIds.map(() => "?").join(",")})`,
    )
    .bind(...workIds)
    .all<any>();

  // Fetch viewer prefs
  const viewer = await db
    .prepare("SELECT preferred_target_lang FROM users WHERE id = ?")
    .bind(viewerId)
    .first<{ preferred_target_lang: string | null }>();

  const target = normLang(viewer?.preferred_target_lang);

  const byWork = new Map<number, any[]>();
  for (const m of matRows) {
    let list = byWork.get(m.work_id);
    if (!list) {
      list = [];
      byWork.set(m.work_id, list);
    }
    list.push(m);
  }

  for (const row of rows) {
    const mats = byWork.get(Number(row.work_id)) || [];
    row.title = resolveWorkTitle(mats, target);
    row.cover = resolveWorkCover(mats, target);
  }
}

// ── Core store functions ─────────────────────────────────────────────

export async function findEntryForWork(
  db: D1Database,
  userId: number,
  workId: number,
): Promise<LibraryEntryRow | null> {
  return db
    .prepare("SELECT * FROM library_entries WHERE user_id = ? AND work_id = ?")
    .bind(userId, workId)
    .first<LibraryEntryRow>();
}

export async function getLibraryEntry(
  db: D1Database,
  entryId: number,
  userId: number,
): Promise<LibraryEntryDetail | null> {
  const entry = await db
    .prepare("SELECT * FROM library_entries WHERE id = ? AND user_id = ?")
    .bind(entryId, userId)
    .first<LibraryEntryRow>();

  if (!entry) return null;

  const { results: links } = await db
    .prepare(
      `SELECT material_id, link_origin, linked_at
       FROM library_materials WHERE entry_id = ?`,
    )
    .bind(entryId)
    .all<any>();

  const { results: summaryRows } = await db
    .prepare(
      `SELECT td.state, COUNT(*) AS n
       FROM library_materials lm
       JOIN chapters c          ON c.material_id = lm.material_id
       JOIN translations t      ON t.work_chapter_id = c.work_chapter_id AND t.owner_id = ?
       JOIN translation_drafts td ON td.id = t.draft_id
       WHERE lm.entry_id = ? AND t.takedown_at IS NULL
       GROUP BY td.state`,
    )
    .bind(userId, entryId)
    .all<{ state: string; n: number }>();

  const summary = { pending: 0, running: 0, done: 0, error: 0 };
  for (const r of summaryRows) {
    const s = r.state;
    if (s === "pending" || s === "running" || s === "done" || s === "error") {
      summary[s] = Number(r.n);
    }
  }

  const detail: any = {
    ...entry,
    materials: links.map(l => ({
      material_id: Number(l.material_id),
      link_origin: l.link_origin,
      linked_at:   l.linked_at,
    })),
    translation_summary: summary,
  };

  await resolveWorkDisplay(db, [detail], userId);
  return detail as LibraryEntryDetail;
}

export async function listLibraryEntries(
  db: D1Database,
  userId: number,
  status?: string | null,
): Promise<LibraryEntryDetail[]> {
  let query = "SELECT * FROM library_entries WHERE user_id = ?";
  const params: any[] = [userId];

  if (status) {
    query += " AND status = ?";
    params.push(status);
  } else {
    query += " AND status <> 'dropped'";
  }
  query += " ORDER BY updated_at DESC";

  const { results: entries } = await db.prepare(query).bind(...params).all<LibraryEntryRow>();
  if (entries.length === 0) return [];

  const entryIds = entries.map(e => e.id);

  // Bulk fetch links
  const { results: allLinks } = await db
    .prepare(
      `SELECT entry_id, material_id, link_origin, linked_at
       FROM library_materials WHERE entry_id IN (${entryIds.map(() => "?").join(",")})`,
    )
    .bind(...entryIds)
    .all<any>();

  // Bulk fetch translation status counts
  const { results: allSummaries } = await db
    .prepare(
      `SELECT lm.entry_id, td.state, COUNT(*) AS n
       FROM library_materials lm
       JOIN chapters c          ON c.material_id = lm.material_id
       JOIN translations t      ON t.work_chapter_id = c.work_chapter_id AND t.owner_id = ?
       JOIN translation_drafts td ON td.id = t.draft_id
       WHERE lm.entry_id IN (${entryIds.map(() => "?").join(",")})
         AND t.takedown_at IS NULL
       GROUP BY lm.entry_id, td.state`,
    )
    .bind(userId, ...entryIds)
    .all<{ entry_id: number; state: string; n: number }>();

  const linksByEntry = new Map<number, LibraryMaterialLink[]>();
  for (const l of allLinks) {
    let list = linksByEntry.get(l.entry_id);
    if (!list) {
      list = [];
      linksByEntry.set(l.entry_id, list);
    }
    list.push({
      material_id: Number(l.material_id),
      link_origin: l.link_origin,
      linked_at:   l.linked_at,
    });
  }

  const summariesByEntry = new Map<number, any>();
  for (const entryId of entryIds) {
    summariesByEntry.set(entryId, { pending: 0, running: 0, done: 0, error: 0 });
  }

  for (const r of allSummaries) {
    const sum = summariesByEntry.get(r.entry_id);
    if (sum) {
      const s = r.state;
      if (s === "pending" || s === "running" || s === "done" || s === "error") {
        sum[s] = Number(r.n);
      }
    }
  }

  const details: any[] = entries.map(e => ({
    ...e,
    materials: linksByEntry.get(e.id) || [],
    translation_summary: summariesByEntry.get(e.id) || { pending: 0, running: 0, done: 0, error: 0 },
  }));

  await resolveWorkDisplay(db, details, userId);
  return details as LibraryEntryDetail[];
}

export async function createLibraryEntry(
  db: D1Database,
  args: {
    user_id:     number;
    work_id:     number;
    target_lang: string;
    materials:   Array<[number, "auto" | "manual"]>;
    status:      string;
  },
): Promise<number> {
  const entry = await db
    .prepare(
      `INSERT INTO library_entries (user_id, work_id, target_lang, status)
       VALUES (?, ?, ?, ?)
       RETURNING id`,
    )
    .bind(args.user_id, args.work_id, args.target_lang, args.status)
    .first<{ id: number }>();

  if (!entry) throw new Error("Failed to create library entry");
  const entryId = entry.id;

  for (const [matId, origin] of args.materials) {
    await linkMaterialToEntry(db, {
      entry_id:    entryId,
      material_id: matId,
      link_origin: origin,
      voter_id:    args.user_id,
    });
  }

  return entryId;
}

export async function updateLibraryEntry(
  db: D1Database,
  entryId: number,
  userId: number,
  args: {
    status?:      string | null;
    target_lang?: string | null;
  },
): Promise<void> {
  let query = "UPDATE library_entries SET updated_at = datetime('now')";
  const params: any[] = [];

  if (args.status !== undefined && args.status !== null) {
    query += `, status = ?`;
    params.push(args.status);
  }
  if (args.target_lang !== undefined && args.target_lang !== null) {
    query += `, target_lang = ?`;
    params.push(args.target_lang);
  }

  query += " WHERE id = ? AND user_id = ?";
  params.push(entryId, userId);

  await db.prepare(query).bind(...params).run();
}

export async function deleteLibraryEntry(
  db: D1Database,
  entryId: number,
  userId: number,
): Promise<boolean> {
  const result = await db
    .prepare("DELETE FROM library_entries WHERE id = ? AND user_id = ?")
    .bind(entryId, userId)
    .run();

  return (result.meta?.changes ?? 0) > 0;
}

export async function linkMaterialToEntry(
  db: D1Database,
  args: {
    entry_id:    number;
    material_id: number;
    link_origin: "auto" | "manual";
    voter_id:    number;
  },
): Promise<void> {
  const entry = await db
    .prepare("SELECT user_id FROM library_entries WHERE id = ?")
    .bind(args.entry_id)
    .first<{ user_id: number }>();

  if (!entry) throw new NotFoundError("Library entry not found");

  const { results: existing } = await db
    .prepare("SELECT material_id FROM library_materials WHERE entry_id = ?")
    .bind(args.entry_id)
    .all<{ material_id: number }>();

  await db
    .prepare(
      `INSERT INTO library_materials (entry_id, material_id, user_id, link_origin)
       VALUES (?, ?, ?, ?)
       ON CONFLICT (entry_id, material_id) DO NOTHING`,
    )
    .bind(args.entry_id, args.material_id, entry.user_id, args.link_origin)
    .run();

  // Cast link votes between this new material and all existing sibling materials in the entry
  for (const row of existing) {
    const other = row.material_id;
    if (other === args.material_id) continue;
    const [a, b] = other < args.material_id ? [other, args.material_id] : [args.material_id, other];

    await db
      .prepare(
        `INSERT INTO material_link_votes (material_a_id, material_b_id, voter_id, vote)
         VALUES (?, ?, ?, 1)
         ON CONFLICT (material_a_id, material_b_id, voter_id)
         DO UPDATE SET vote=1, voted_at=datetime('now')`,
      )
      .bind(a, b, args.voter_id)
      .run();
  }
}

export async function unlinkMaterialFromEntry(
  db: D1Database,
  args: {
    entry_id:    number;
    material_id: number;
    voter_id:    number;
  },
): Promise<void> {
  const { results: siblings } = await db
    .prepare("SELECT material_id FROM library_materials WHERE entry_id = ? AND material_id != ?")
    .bind(args.entry_id, args.material_id)
    .all<{ material_id: number }>();

  await db
    .prepare("DELETE FROM library_materials WHERE entry_id = ? AND material_id = ?")
    .bind(args.entry_id, args.material_id)
    .run();

  for (const s of siblings) {
    const [a, b] = s.material_id < args.material_id ? [s.material_id, args.material_id] : [args.material_id, s.material_id];
    await db
      .prepare("DELETE FROM material_link_votes WHERE material_a_id = ? AND material_b_id = ? AND voter_id = ?")
      .bind(a, b, args.voter_id)
      .run();
  }
}

export interface RecentReadRow {
  work_id:          number;
  material_id:      number;
  title:            string;
  cover:            string | null;
  work_chapter_id:  number;
  chapter_number:   string;
  chapter_label:    string | null;
  translation_id:   number | null;
  last_read_at:     string | null;
}

export async function recordReading(
  db: D1Database,
  args: {
    user_id:          number;
    work_chapter_id:  number;
    last_material_id: number;
    translation_id:   number | null;
  },
): Promise<void> {
  await db
    .prepare(
      `INSERT INTO reading_history (user_id, work_chapter_id, last_material_id, translation_id, last_read_at)
       VALUES (?, ?, ?, ?, datetime('now'))
       ON CONFLICT (user_id, work_chapter_id) DO UPDATE SET
         last_material_id = COALESCE(EXCLUDED.last_material_id, reading_history.last_material_id),
         translation_id   = COALESCE(EXCLUDED.translation_id, reading_history.translation_id),
         last_read_at     = datetime('now')`,
    )
    .bind(args.user_id, args.work_chapter_id, args.last_material_id, args.translation_id)
    .run();
}

export async function listRecentReads(
  db: D1Database,
  args: {
    user_id: number;
    limit:   number;
  },
): Promise<RecentReadRow[]> {
  const { results: rows } = await db
    .prepare(
      `SELECT
         m.work_id         AS work_id,
         m.id              AS material_id,
         wc.id             AS work_chapter_id,
         wc.number_norm    AS chapter_number,
         wc.label          AS chapter_label,
         rh.translation_id,
         rh.last_read_at
       FROM reading_history rh
       JOIN work_chapters wc ON wc.id = rh.work_chapter_id
       JOIN materials m ON m.id = rh.last_material_id
       WHERE rh.user_id = ?
         AND rh.last_material_id IS NOT NULL
         AND rh.last_read_at = (
             SELECT MAX(rh2.last_read_at)
             FROM reading_history rh2
             JOIN work_chapters wc2 ON wc2.id = rh2.work_chapter_id
             JOIN materials m2 ON m2.id = rh2.last_material_id
             WHERE rh2.user_id = rh.user_id
               AND m2.work_id = m.work_id
         )
       ORDER BY rh.last_read_at DESC
       LIMIT ?`,
    )
    .bind(args.user_id, args.limit)
    .all<any>();

  if (rows.length === 0) return [];

  const out: RecentReadRow[] = rows.map(r => ({
    work_id:          Number(r.work_id),
    material_id:      Number(r.material_id),
    title:            "",
    cover:            null,
    work_chapter_id:  Number(r.work_chapter_id),
    chapter_number:   r.chapter_number,
    chapter_label:    r.chapter_label,
    translation_id:   r.translation_id ? Number(r.translation_id) : null,
    last_read_at:     r.last_read_at,
  }));

  await resolveWorkDisplay(db, out, args.user_id);
  return out;
}

