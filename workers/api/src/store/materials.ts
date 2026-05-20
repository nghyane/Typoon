/**
 * Materials store — D1 queries for manga materials, linking votes, and split operations.
 */

import type { D1Database } from "@cloudflare/workers-types";
import { ConflictError, NotFoundError, ForbiddenError } from "./db";

export interface MaterialRow {
  id:            number;
  imported_by:   number | null;
  origin:        "source" | "extension" | "upload";
  work_id:       number;
  source:        string | null;
  upstream_ref:  string | null;
  title:         string;
  cover_url:     string | null;
  description:   string | null;
  author:        string | null;
  status:        string | null;
  languages:     string; // JSON string[]
  title_native:  string | null;
  title_alt:     string | null; // JSON string[]
  cross_refs:    string | null; // JSON object
  title_locale:  string | null; // JSON object
  start_year:    number | null;
  nsfw:          number; // 0=false 1=true
  created_at:    string;
  updated_at:    string;
}

export async function getMaterial(
  db: D1Database,
  id: number,
): Promise<MaterialRow | null> {
  return db
    .prepare("SELECT * FROM materials WHERE id = ?")
    .bind(id)
    .first<MaterialRow>();
}

export async function listMaterialsForWork(
  db: D1Database,
  workId: number,
): Promise<MaterialRow[]> {
  const { results } = await db
    .prepare("SELECT * FROM materials WHERE work_id = ? ORDER BY created_at ASC, id ASC")
    .bind(workId)
    .all<MaterialRow>();
  return results;
}

export async function getLinkVote(
  db: D1Database,
  voterId: number,
  materialAId: number,
  materialBId: number,
): Promise<number | null> {
  const [a, b] = materialAId < materialBId ? [materialAId, materialBId] : [materialBId, materialAId];
  const row = await db
    .prepare("SELECT vote FROM material_link_votes WHERE material_a_id = ? AND material_b_id = ? AND voter_id = ?")
    .bind(a, b, voterId)
    .first<{ vote: number }>();
  return row ? row.vote : null;
}

export async function getSplitVote(
  db: D1Database,
  voterId: number,
  materialId: number,
): Promise<number | null> {
  const row = await db
    .prepare("SELECT vote FROM material_split_votes WHERE material_id = ? AND voter_id = ?")
    .bind(materialId, voterId)
    .first<{ vote: number }>();
  return row ? row.vote : null;
}

export async function getSplitScore(
  db: D1Database,
  materialId: number,
): Promise<{ score: number; total: number }> {
  const row = await db
    .prepare(
      `SELECT COALESCE(SUM(vote), 0) AS score, COUNT(*) AS total
       FROM material_split_votes WHERE material_id = ?`,
    )
    .bind(materialId)
    .first<{ score: number; total: number }>();
  return {
    score: row ? Number(row.score) : 0,
    total: row ? Number(row.total) : 0,
  };
}

export async function getRecentForceLink(
  db: D1Database,
  actorId: number,
  materialId: number,
  windowMinutes: number,
): Promise<{ id: number; material_a_id: number; material_b_id: number; target_work_id: number; created_at: string } | null> {
  // SQLite handles date operations differently, we pass the window filter in SQL or calculate in JS.
  // Using SQLite date modifiers: datetime('now', '-' || windowMinutes || ' minutes')
  const row = await db
    .prepare(
      `SELECT id, material_a_id, material_b_id, target_work_id, created_at
       FROM material_link_actions
       WHERE actor_id = ?
         AND kind = 'force_link'
         AND reversed_at IS NULL
         AND created_at > datetime('now', '-' || ? || ' minutes')
         AND (material_a_id = ? OR material_b_id = ?)
       ORDER BY created_at DESC
       LIMIT 1`,
    )
    .bind(actorId, windowMinutes, materialId, materialId)
    .first<any>();
  return row ?? null;
}

export async function logForceAction(
  db: D1Database,
  args: {
    actor_id:       number;
    kind:           "force_link" | "force_unlink";
    material_a_id:  number;
    material_b_id:  number | null;
    target_work_id: number;
  },
): Promise<number> {
  const row = await db
    .prepare(
      `INSERT INTO material_link_actions (actor_id, kind, material_a_id, material_b_id, target_work_id)
       VALUES (?, ?, ?, ?, ?) RETURNING id`,
    )
    .bind(args.actor_id, args.kind, args.material_a_id, args.material_b_id, args.target_work_id)
    .first<{ id: number }>();
  if (!row) throw new Error("Log force action failed");
  return row.id;
}

// ── Merge Works Inline logic ───────────────────────────────────────────

function crossRefsConflict(a: Record<string, any>, b: Record<string, any>): boolean {
  if (!a || !b) return false;
  for (const k of Object.keys(a)) {
    if (k in b) {
      const va = a[k];
      const vb = b[k];
      if (va === null || va === undefined || vb === null || vb === undefined) continue;
      if (String(va).trim() !== String(vb).trim()) return true;
    }
  }
  return false;
}

export interface LinkVoteResult {
  vote:              number;
  score:             number;
  merged:            boolean;
  canonical_work_id: number | null;
  blocked_reason:    string | null;
}

export async function castLinkVoteWithMerge(
  db: D1Database,
  args: {
    voter_id:      number;
    material_a_id: number;
    material_b_id: number;
    vote:          number;
    threshold:     number;
    force_merge?:  boolean;
  },
): Promise<LinkVoteResult> {
  if (args.vote !== -1 && args.vote !== 1) throw new Error("vote must be ±1");
  if (args.material_a_id === args.material_b_id) throw new Error("cannot vote on a pair with itself");

  const [a_id, b_id] = args.material_a_id < args.material_b_id ? [args.material_a_id, args.material_b_id] : [args.material_b_id, args.material_a_id];

  // Cast vote
  await db
    .prepare(
      `INSERT INTO material_link_votes (material_a_id, material_b_id, voter_id, vote)
       VALUES (?, ?, ?, ?)
       ON CONFLICT (material_a_id, material_b_id, voter_id) DO UPDATE
         SET vote = excluded.vote, voted_at = datetime('now')`,
    )
    .bind(a_id, b_id, args.voter_id, args.vote)
    .run();

  // Get score
  const scoreRow = await db
    .prepare("SELECT COALESCE(SUM(vote), 0) AS score FROM material_link_votes WHERE material_a_id = ? AND material_b_id = ?")
    .bind(a_id, b_id)
    .first<{ score: number }>();
  const score = scoreRow ? Number(scoreRow.score) : 0;

  const result: LinkVoteResult = {
    vote:              args.vote,
    score:             score,
    merged:            false,
    canonical_work_id: null,
    blocked_reason:    null,
  };

  if (!args.force_merge && score < args.threshold) {
    return result;
  }

  // Fetch current work assignments
  const mats = await db
    .prepare("SELECT id, work_id FROM materials WHERE id IN (?, ?)")
    .bind(a_id, b_id)
    .all<{ id: number; work_id: number }>();

  const byId = new Map(mats.results.map(r => [r.id, r.work_id]));
  const work_a = byId.get(a_id);
  const work_b = byId.get(b_id);

  if (work_a === undefined || work_b === undefined) return result;
  if (work_a === work_b) {
    result.canonical_work_id = work_a;
    result.blocked_reason = "same_work";
    return result;
  }

  // Cross refs conflict check
  const works = await db
    .prepare("SELECT id, cross_refs FROM works WHERE id IN (?, ?)")
    .bind(work_a, work_b)
    .all<{ id: number; cross_refs: string | null }>();

  const refsByWork = new Map<number, Record<string, any>>();
  for (const w of works.results) {
    refsByWork.set(w.id, w.cross_refs ? JSON.parse(w.cross_refs) : {});
  }

  if (crossRefsConflict(refsByWork.get(work_a) || {}, refsByWork.get(work_b) || {})) {
    result.blocked_reason = "cross_refs_conflict";
    return result;
  }

  // Perform merge
  const canonical = Math.min(work_a, work_b);
  const doomed    = Math.max(work_a, work_b);

  await mergeWorks(db, canonical, doomed);

  result.merged = true;
  result.canonical_work_id = canonical;
  return result;
}

async function mergeWorks(db: D1Database, canonical: number, doomed: number): Promise<void> {
  // 1. Move materials
  await db.prepare("UPDATE materials SET work_id = ? WHERE work_id = ?").bind(canonical, doomed).run();

  // 2. Reconcile work_chapters one-by-one
  const doomedChs = await db
    .prepare("SELECT id, number_norm FROM work_chapters WHERE work_id = ?")
    .bind(doomed)
    .all<{ id: number; number_norm: string }>();

  for (const wc of doomedChs.results) {
    const doomed_wc = wc.id;
    const norm      = wc.number_norm;

    const existing = await db
      .prepare("SELECT id FROM work_chapters WHERE work_id = ? AND number_norm = ?")
      .bind(canonical, norm)
      .first<{ id: number }>();

    if (!existing) {
      await db.prepare("UPDATE work_chapters SET work_id = ? WHERE id = ?").bind(canonical, doomed_wc).run();
      continue;
    }

    const target_wc = existing.id;

    // Re-point chapters
    await db.prepare("UPDATE chapters SET work_chapter_id = ? WHERE work_chapter_id = ?").bind(target_wc, doomed_wc).run();

    // Delete duplicate colliding translations
    await db
      .prepare(
        `DELETE FROM translations
         WHERE work_chapter_id = ?
           AND EXISTS (
             SELECT 1 FROM translations t2
             WHERE t2.work_chapter_id = ?
               AND t2.owner_id = translations.owner_id
               AND t2.target_lang = translations.target_lang
           )`,
      )
      .bind(doomed_wc, target_wc)
      .run();

    // Re-point translations
    await db.prepare("UPDATE translations SET work_chapter_id = ? WHERE work_chapter_id = ?").bind(target_wc, doomed_wc).run();

    // Re-point reading history
    await db
      .prepare(
        `DELETE FROM reading_history
         WHERE work_chapter_id = ?
           AND user_id IN (
             SELECT user_id FROM reading_history WHERE work_chapter_id = ?
           )`,
      )
      .bind(doomed_wc, target_wc)
      .run();

    await db.prepare("UPDATE reading_history SET work_chapter_id = ? WHERE work_chapter_id = ?").bind(target_wc, doomed_wc).run();

    // Delete doomed work_chapter
    await db.prepare("DELETE FROM work_chapters WHERE id = ?").bind(doomed_wc).run();
  }

  // 3. Merge cross refs
  const works = await db
    .prepare("SELECT id, cross_refs FROM works WHERE id IN (?, ?)")
    .bind(canonical, doomed)
    .all<{ id: number; cross_refs: string | null }>();

  let canonicalRefs: Record<string, any> = {};
  let doomedRefs: Record<string, any> = {};
  for (const w of works.results) {
    if (w.id === canonical && w.cross_refs) canonicalRefs = JSON.parse(w.cross_refs);
    if (w.id === doomed && w.cross_refs) doomedRefs = JSON.parse(w.cross_refs);
  }

  const mergedRefs = { ...doomedRefs, ...canonicalRefs };
  await db
    .prepare("UPDATE works SET cross_refs = ?, updated_at = datetime('now') WHERE id = ?")
    .bind(JSON.stringify(mergedRefs), canonical)
    .run();

  // 4. Collapse duplicate library entries
  const dupEntries = await db
    .prepare(
      `SELECT user_id, GROUP_CONCAT(id) AS ids
       FROM library_entries
       WHERE work_id = ?
       GROUP BY user_id
       HAVING COUNT(*) > 1`,
    )
    .bind(canonical)
    .all<{ user_id: number; ids: string }>();

  for (const dup of dupEntries.results) {
    const ids = dup.ids.split(",").map(Number).sort((x, y) => x - y);
    const keeper = ids[0];
    const losers = ids.slice(1);

    for (const loser of losers) {
      await db.prepare("UPDATE library_materials SET entry_id = ? WHERE entry_id = ?").bind(keeper, loser).run();
      await db.prepare("DELETE FROM library_entries WHERE id = ?").bind(loser).run();
    }
  }

  // 5. Log the redirect
  await db
    .prepare(
      `INSERT INTO work_redirects (old_id, new_id) VALUES (?, ?)
       ON CONFLICT (old_id) DO UPDATE SET new_id = excluded.new_id, merged_at = datetime('now')`,
    )
    .bind(doomed, canonical)
    .run();

  // 6. Delete doomed work
  await db.prepare("DELETE FROM works WHERE id = ?").bind(doomed).run();
}

// ── Split Material Inline logic ────────────────────────────────────────

export interface SplitVoteResult {
  vote:           number;
  score:          number;
  split:          boolean;
  new_work_id:    number | null;
  blocked_reason: string | null;
}

export async function castSplitVoteWithSplit(
  db: D1Database,
  args: {
    voter_id:    number;
    material_id: number;
    vote:        number;
    threshold:   number;
    force_split?: boolean;
  },
): Promise<SplitVoteResult> {
  if (args.vote !== -1 && args.vote !== 1) throw new Error("vote must be ±1");

  await db
    .prepare(
      `INSERT INTO material_split_votes (material_id, voter_id, vote)
       VALUES (?, ?, ?)
       ON CONFLICT (material_id, voter_id) DO UPDATE
         SET vote = excluded.vote, voted_at = datetime('now')`,
    )
    .bind(args.material_id, args.voter_id, args.vote)
    .run();

  const scoreRow = await db
    .prepare("SELECT COALESCE(SUM(vote), 0) AS score FROM material_split_votes WHERE material_id = ?")
    .bind(args.material_id)
    .first<{ score: number }>();
  const score = scoreRow ? Number(scoreRow.score) : 0;

  const result: SplitVoteResult = {
    vote:           args.vote,
    score:          score,
    split:          false,
    new_work_id:    null,
    blocked_reason: null,
  };

  if (!args.force_split && score < args.threshold) {
    return result;
  }

  const mat = await db.prepare("SELECT work_id FROM materials WHERE id = ?").bind(args.material_id).first<{ work_id: number }>();
  if (!mat) {
    result.blocked_reason = "material_gone";
    return result;
  }
  const work_id = mat.work_id;

  const sib = await db.prepare("SELECT COUNT(*) AS n FROM materials WHERE work_id = ?").bind(work_id).first<{ n: number }>();
  if (!sib || sib.n < 2) {
    result.blocked_reason = "solo_member";
    return result;
  }

  // Move material to a new work
  const newWorkRow = await db.prepare("INSERT INTO works (cross_refs) VALUES (NULL) RETURNING id").first<{ id: number }>();
  if (!newWorkRow) throw new Error("New work insert failed");

  await splitMaterialToWork(db, args.material_id, newWorkRow.id);

  // Clear split votes
  await db.prepare("DELETE FROM material_split_votes WHERE material_id = ?").bind(args.material_id).run();

  result.split = true;
  result.new_work_id = newWorkRow.id;
  return result;
}

async function splitMaterialToWork(db: D1Database, materialId: number, newWorkId: number): Promise<void> {
  // 1. Re-home the material
  await db.prepare("UPDATE materials SET work_id = ? WHERE id = ?").bind(newWorkId, materialId).run();

  // 2. Fetch and re-home chapters
  const oldChs = await db
    .prepare(
      `SELECT c.id AS chapter_id, c.work_chapter_id AS old_wc_id, wc.number_norm, wc.label
       FROM chapters c
       JOIN work_chapters wc ON wc.id = c.work_chapter_id
       WHERE c.material_id = ?`,
    )
    .bind(materialId)
    .all<{ chapter_id: number; old_wc_id: number; number_norm: string; label: string | null }>();

  for (const r of oldChs.results) {
    const chapter_id = r.chapter_id;
    const old_wc_id  = r.old_wc_id;

    // Create a new work_chapter row
    const newWcRow = await db
      .prepare("INSERT INTO work_chapters (work_id, number_norm, label) VALUES (?, ?, ?) RETURNING id")
      .bind(newWorkId, r.number_norm, r.label)
      .first<{ id: number }>();

    if (!newWcRow) throw new Error("New work_chapter insert failed");
    const new_wc_id = newWcRow.id;

    // Re-point chapter
    await db.prepare("UPDATE chapters SET work_chapter_id = ? WHERE id = ?").bind(new_wc_id, chapter_id).run();

    // Re-point translations
    await db
      .prepare(
        `UPDATE translations SET work_chapter_id = ?
         WHERE work_chapter_id = ? AND id IN (
           SELECT t.id FROM translations t
           JOIN translation_drafts d ON d.id = t.draft_id
           JOIN chapters cc ON cc.id = d.chapter_id
           WHERE cc.material_id = ?
         )`,
      )
      .bind(new_wc_id, old_wc_id, materialId)
      .run();

    // Re-point reading history
    await db
      .prepare("UPDATE reading_history SET work_chapter_id = ? WHERE work_chapter_id = ? AND last_material_id = ?")
      .bind(new_wc_id, old_wc_id, materialId)
      .run();

    // Clean orphaned old work_chapter
    const leftover = await db.prepare("SELECT 1 FROM chapters WHERE work_chapter_id = ? LIMIT 1").bind(old_wc_id).first();
    if (!leftover) {
      await db.prepare("DELETE FROM work_chapters WHERE id = ?").bind(old_wc_id).run();
    }
  }
}

export async function forceUnlinkMaterial(
  db: D1Database,
  actorId: number,
  materialId: number,
): Promise<{ new_work_id: number; previous_work_id: number }> {
  const mat = await db.prepare("SELECT work_id FROM materials WHERE id = ?").bind(materialId).first<{ work_id: number }>();
  if (!mat) throw new NotFoundError("Material not found");
  const previous_work_id = mat.work_id;

  const sib = await db.prepare("SELECT COUNT(*) AS n FROM materials WHERE work_id = ?").bind(previous_work_id).first<{ n: number }>();
  if (!sib || sib.n < 2) throw new Error("solo_member");

  const newWorkRow = await db.prepare("INSERT INTO works (cross_refs) VALUES (NULL) RETURNING id").first<{ id: number }>();
  if (!newWorkRow) throw new Error("New work insert failed");
  const new_work_id = newWorkRow.id;

  await splitMaterialToWork(db, materialId, new_work_id);

  // Clear split votes
  await db.prepare("DELETE FROM material_split_votes WHERE material_id = ?").bind(materialId).run();

  // Log action
  await logForceAction(db, {
    actor_id:       actorId,
    kind:           "force_unlink",
    material_a_id:  materialId,
    material_b_id:  null,
    target_work_id: new_work_id,
  });

  // Reverse original force_link action
  await db
    .prepare(
      `UPDATE material_link_actions
       SET reversed_at = datetime('now')
       WHERE actor_id = ?
         AND kind = 'force_link'
         AND reversed_at IS NULL
         AND (material_a_id = ? OR material_b_id = ?)`,
    )
    .bind(actorId, materialId, materialId)
    .run();

  return { new_work_id, previous_work_id };
}

// ── Link suggestions & similarity fuzzy-matching ─────────────────────

export async function listWorkLinkSuggestions(
  db: D1Database,
  workId: number,
): Promise<any[]> {
  const { results } = await db
    .prepare(
      `WITH own_ms AS (
           SELECT id FROM materials WHERE work_id = ?
       ),
       voted AS (
           SELECT
             CASE WHEN v.material_a_id IN (SELECT id FROM own_ms)
                  THEN v.material_b_id
                  ELSE v.material_a_id
             END AS candidate_id,
             CASE WHEN v.material_a_id IN (SELECT id FROM own_ms)
                  THEN v.material_a_id
                  ELSE v.material_b_id
             END AS own_id,
             v.vote
           FROM material_link_votes v
           WHERE v.material_a_id IN (SELECT id FROM own_ms)
              OR v.material_b_id IN (SELECT id FROM own_ms)
       ),
       agg AS (
           SELECT candidate_id,
                  SUM(vote) AS score,
                  COUNT(*) AS total,
                  MAX(own_id) AS own_id
           FROM voted
           GROUP BY candidate_id
       )
       SELECT
           m.id          AS candidate_material_id,
           m.title       AS candidate_title,
           m.source      AS candidate_source,
           m.cover_url   AS candidate_cover,
           m.work_id     AS candidate_work_id,
           agg.score     AS score,
           agg.total     AS total,
           agg.own_id    AS own_material_id
       FROM agg
       JOIN materials m ON m.id = agg.candidate_id
       WHERE m.work_id != ?
         AND agg.score > 0
       ORDER BY agg.score DESC, m.id ASC`,
    )
    .bind(workId, workId)
    .all<any>();
  return results;
}

// Levenshtein fuzzy string distance helper
function levenshtein(a: string, b: string): number {
  const m = a.length;
  const n = b.length;
  let prev = Array.from({ length: n + 1 }, (_, i) => i);
  let curr = new Array<number>(n + 1);

  for (let i = 1; i <= m; i++) {
    curr[0] = i;
    for (let j = 1; j <= n; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      curr[j] = Math.min(
        (curr[j - 1] ?? 0) + 1, // left
        (prev[j] ?? 0) + 1,    // above
        (prev[j - 1] ?? 0) + cost // diagonal
      );
    }
    prev = [...curr];
  }
  return prev[n] ?? 0;
}

function titleSimilarity(a: string, b: string): number {
  const len = Math.max(a.length, b.length);
  if (len === 0) return 1.0;
  const dist = levenshtein(a.toLowerCase().trim(), b.toLowerCase().trim());
  return 1.0 - dist / len;
}

export async function listWorkLinkCandidates(
  db: D1Database,
  workId: number,
  limit: number = 10,
  threshold: number = 0.45,
): Promise<any[]> {
  // Fetch own materials
  const ownMats = await listMaterialsForWork(db, workId);
  if (ownMats.length === 0) return [];

  // Compile own titles list
  const ownTitles: { own_id: number; title: string; is_native: boolean }[] = [];
  for (const m of ownMats) {
    ownTitles.push({ own_id: m.id, title: m.title, is_native: false });
    if (m.title_native) {
      ownTitles.push({ own_id: m.id, title: m.title_native, is_native: true });
    }
    if (m.title_alt) {
      try {
        const alts = JSON.parse(m.title_alt) as string[];
        for (const alt of alts) {
          ownTitles.push({ own_id: m.id, title: alt, is_native: false });
        }
      } catch {}
    }
  }

  // Fetch candidate materials outside of this work, excluding existing redirect links
  const { results: candidates } = await db
    .prepare(
      `SELECT id, title, title_native, title_alt, cover_url, source, work_id, languages FROM materials
       WHERE work_id != ?
         AND work_id NOT IN (
           SELECT new_id FROM work_redirects WHERE old_id = ?
           UNION ALL
           SELECT old_id FROM work_redirects WHERE new_id = ?
         )`,
    )
    .bind(workId, workId, workId)
    .all<any>();

  // Rate and match candidates in JavaScript
  const matches: any[] = [];
  for (const cand of candidates) {
    let maxSim = 0;
    const firstOwn = ownTitles[0];
    let matchingOwnId = firstOwn ? firstOwn.own_id : 0;
    let exactNative = false;
    let altOverlap = false;

    // Check exact native title agreement
    if (cand.title_native) {
      const cNat = cand.title_native.toLowerCase().trim();
      const hasExactNative = ownTitles.some(ot => ot.is_native && ot.title.toLowerCase().trim() === cNat);
      if (hasExactNative) {
        exactNative = true;
      }
    }

    // Check alt titles overlap
    if (cand.title_alt) {
      try {
        const cAlts = JSON.parse(cand.title_alt) as string[];
        const hasAltOverlap = ownTitles.some(ot => cAlts.some(ca => ca.toLowerCase().trim() === ot.title.toLowerCase().trim()));
        if (hasAltOverlap) {
          altOverlap = true;
        }
      } catch {}
    }

    // Compute maximum Levenshtein similarity against own titles
    for (const ot of ownTitles) {
      const sim = titleSimilarity(cand.title, ot.title);
      if (sim > maxSim) {
        maxSim = sim;
        matchingOwnId = ot.own_id;
      }
      if (cand.title_native) {
        const simNat = titleSimilarity(cand.title_native, ot.title);
        if (simNat > maxSim) {
          maxSim = simNat;
          matchingOwnId = ot.own_id;
        }
      }
    }

    let finalScore = maxSim;
    let reason = "title_trgm";

    if (exactNative) {
      finalScore = 0.95;
      reason = "title_native_exact";
    } else if (altOverlap) {
      finalScore = Math.max(maxSim, 0.80);
      reason = "title_alt_overlap";
    }

    if (finalScore >= threshold || exactNative || altOverlap) {
      matches.push({
        candidate_material_id: cand.id,
        candidate_title:       cand.title,
        candidate_source:      cand.source,
        candidate_cover:       cand.cover_url,
        candidate_work_id:     cand.work_id,
        own_material_id:       matchingOwnId,
        score:                 finalScore,
        reason:                reason,
      });
    }
  }

  // Sort logically and slice
  return matches.sort((a, b) => b.score - a.score).slice(0, limit);
}

export async function getOrCreateUploadMaterial(
  db: D1Database,
  args: {
    work_id:     number;
    imported_by: number;
  },
): Promise<number> {
  const existing = await db
    .prepare(
      `SELECT id FROM materials
       WHERE work_id = ? AND imported_by = ? AND origin = 'upload'`,
    )
    .bind(args.work_id, args.imported_by)
    .first<{ id: number }>();

  if (existing) {
    return existing.id;
  }

  // Find some sibling title or fallback to "Tải lên"
  const titleRow = await db
    .prepare(
      `SELECT COALESCE(
         (SELECT title FROM materials
          WHERE work_id = ?
          ORDER BY (origin = 'source') DESC, id ASC LIMIT 1),
         'Tải lên'
       ) AS title`,
    )
    .bind(args.work_id)
    .first<{ title: string }>();

  const title = titleRow?.title ?? "Tải lên";

  const row = await db
    .prepare(
      `INSERT INTO materials (imported_by, origin, work_id, title, languages)
       VALUES (?, 'upload', ?, ?, ?)
       RETURNING id`,
    )
    .bind(args.imported_by, args.work_id, title, JSON.stringify(["vi"]))
    .first<{ id: number }>();

  if (!row) throw new Error("Failed to create upload material");
  return row.id;
}

export async function deleteUserUploadMaterial(
  db: D1Database,
  args: {
    work_id: number;
    user_id: number;
  },
): Promise<boolean> {
  const result = await db
    .prepare(
      `DELETE FROM materials
       WHERE work_id = ? AND imported_by = ? AND origin = 'upload'`,
    )
    .bind(args.work_id, args.user_id)
    .run();

  return (result.meta?.changes ?? 0) > 0;
}

// ── Material CRUD ───────────────────────────────────────────────────

export interface ImportMaterialArgs {
  imported_by:   number;
  origin:        "source" | "extension" | "upload";
  work_id:       number;
  source:        string | null;
  upstream_ref:  string | null;
  title:         string;
  cover_url?:    string | null;
  description?:  string | null;
  author?:       string | null;
  status?:       string | null;
  languages?:    string[];
  title_native?: string | null;
  title_alt?:    string[];
  cross_refs?:   Record<string, unknown> | null;
  nsfw?:         boolean;
}

export async function importMaterial(
  db: D1Database,
  args: ImportMaterialArgs,
): Promise<MaterialRow> {
  const row = await db
    .prepare(
      `INSERT INTO materials (
         imported_by, origin, work_id, source, upstream_ref,
         title, cover_url, description, author, status,
         languages, title_native, title_alt, cross_refs, nsfw
       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
       RETURNING *`,
    )
    .bind(
      args.imported_by,
      args.origin,
      args.work_id,
      args.source ?? null,
      args.upstream_ref ?? null,
      args.title,
      args.cover_url ?? null,
      args.description ?? null,
      args.author ?? null,
      args.status ?? null,
      JSON.stringify(args.languages ?? []),
      args.title_native ?? null,
      args.title_alt ? JSON.stringify(args.title_alt) : null,
      args.cross_refs ? JSON.stringify(args.cross_refs) : null,
      args.nsfw ? 1 : 0,
    )
    .first<MaterialRow>();

  if (!row) throw new Error("Material import failed");
  return row;
}

export interface UpdateMaterialArgs {
  title?:        string;
  cover_url?:    string | null;
  description?:  string | null;
  author?:       string | null;
  status?:       string | null;
  title_native?: string | null;
  title_alt?:    string[];
  cross_refs?:   Record<string, unknown> | null;
  title_locale?: Record<string, string> | null;
  start_year?:   number | null;
  nsfw?:         boolean;
}

export async function updateMaterial(
  db: D1Database,
  materialId: number,
  args: UpdateMaterialArgs,
): Promise<MaterialRow> {
  // Build dynamic SET clause — only update provided fields
  const sets: string[] = [];
  const params: (string | number | null)[] = [];

  if (args.title !== undefined)       { sets.push("title = ?");       params.push(args.title); }
  if (args.cover_url !== undefined)   { sets.push("cover_url = ?");   params.push(args.cover_url); }
  if (args.description !== undefined) { sets.push("description = ?"); params.push(args.description); }
  if (args.author !== undefined)      { sets.push("author = ?");      params.push(args.author); }
  if (args.status !== undefined)      { sets.push("status = ?");      params.push(args.status); }
  if (args.title_native !== undefined){ sets.push("title_native = ?");params.push(args.title_native); }
  if (args.title_alt !== undefined)   { sets.push("title_alt = ?");   params.push(JSON.stringify(args.title_alt)); }
  if (args.cross_refs !== undefined)  { sets.push("cross_refs = ?");  params.push(args.cross_refs ? JSON.stringify(args.cross_refs) : null); }
  if (args.title_locale !== undefined){ sets.push("title_locale = ?");params.push(args.title_locale ? JSON.stringify(args.title_locale) : null); }
  if (args.start_year !== undefined)  { sets.push("start_year = ?");  params.push(args.start_year); }
  if (args.nsfw !== undefined)        { sets.push("nsfw = ?");        params.push(args.nsfw ? 1 : 0); }

  if (sets.length === 0) {
    // Nothing to update — just return current row
    const existing = await getMaterial(db, materialId);
    if (!existing) throw new NotFoundError("Material not found");
    return existing;
  }

  sets.push("updated_at = datetime('now')");
  params.push(materialId);

  const row = await db
    .prepare(`UPDATE materials SET ${sets.join(", ")} WHERE id = ? RETURNING *`)
    .bind(...params)
    .first<MaterialRow>();

  if (!row) throw new NotFoundError("Material not found");
  return row;
}

export async function deleteMaterial(
  db: D1Database,
  materialId: number,
  userId: number,
): Promise<boolean> {
  const mat = await getMaterial(db, materialId);
  if (!mat) return false;

  // Only upload/extension origin materials can be deleted by their importer
  if (mat.origin === "source") {
    throw new ForbiddenError("Cannot delete source-backed materials");
  }
  if (mat.imported_by !== userId) {
    throw new ForbiddenError("Only the importer can delete this material");
  }

  const result = await db
    .prepare("DELETE FROM materials WHERE id = ?")
    .bind(materialId)
    .run();

  return (result.meta?.changes ?? 0) > 0;
}

