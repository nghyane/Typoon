/**
 * Work routes — global identity hub for cross-source manga pages.
 */

import { Hono } from "hono";
import type { Env, ContextVars } from "../types";
import { getWork, createBlankWork, deleteUserWork, listWorkChaptersWithTranslations } from "../store/works";
import {
  listMaterialsForWork,
  getLinkVote,
  getSplitVote,
  getSplitScore,
  getRecentForceLink,
  logForceAction,
  castLinkVoteWithMerge,
  castSplitVoteWithSplit,
  forceUnlinkMaterial,
  listWorkLinkSuggestions,
  listWorkLinkCandidates,
  getOrCreateUploadMaterial,
  deleteUserUploadMaterial,
  getMaterial,
} from "../store/materials";
import { findEntryForWork } from "../store/library";
import { NotFoundError, ConflictError } from "../store/db";

const router = new Hono<{ Bindings: Env; Variables: ContextVars }>();

const LINK_MERGE_THRESHOLD = 2;
const SPLIT_THRESHOLD      = 2;
const FORCE_UNDO_WINDOW_MIN = 10;

// Helper to assert work exists
async function requireWork(db: any, id: number) {
  const work = await getWork(db, id);
  if (!work) throw new NotFoundError("Work not found");
  return work;
}

// Helper to assert material exists
async function requireMaterial(db: any, id: number) {
  const mat = await getMaterial(db, id);
  if (!mat) throw new NotFoundError("Material not found");
  return mat;
}

async function resolveLinkPair(
  workId: number,
  targetMaterialId: number,
  ownMaterialId: number | undefined | null,
  db: any,
): Promise<[number, number]> {
  await requireWork(db, workId);
  const target = await requireMaterial(db, targetMaterialId);

  let ownId: number;
  if (ownMaterialId !== undefined && ownMaterialId !== null) {
    const own = await requireMaterial(db, ownMaterialId);
    if (Number(own.work_id) !== workId) {
      const err = new Error("own_material_id does not belong to this work");
      (err as any).status = 400;
      throw err;
    }
    ownId = Number(own.id);
  } else {
    const mats = await listMaterialsForWork(db, workId);
    const firstMat = mats[0];
    if (!firstMat) {
      const err = new Error("Work has no materials");
      (err as any).status = 409;
      throw err;
    }
    ownId = Number(firstMat.id);
  }

  if (Number(target.id) === ownId) {
    const err = new Error("Cannot link a material to itself");
    (err as any).status = 400;
    throw err;
  }

  if (Number(target.work_id) === workId) {
    const err = new Error("target material already belongs to this work");
    (err as any).status = 409;
    throw err;
  }

  return [ownId, Number(target.id)];
}

// ── GET /:work_id ─────────────────────────────────────────────────────

router.get("/:work_id", async (ctx) => {
  const userId = ctx.get("userId");
  const workId = Number(ctx.req.param("work_id"));

  const db = ctx.env.DB;
  const work = await requireWork(db, workId);
  const materials = await listMaterialsForWork(db, workId);
  const chapters = await listWorkChaptersWithTranslations(db, workId, userId);
  const entry = await findEntryForWork(db, userId, workId);

  return ctx.json({
    work: {
      id:         Number(work.id),
      cross_refs: work.cross_refs ? JSON.parse(work.cross_refs) : null,
      created_at: work.created_at,
      updated_at: work.updated_at,
    },
    materials: materials.map(m => ({
      ...m,
      languages:  m.languages ? JSON.parse(m.languages) : [],
      title_alt:  m.title_alt ? JSON.parse(m.title_alt) : [],
      cross_refs: m.cross_refs ? JSON.parse(m.cross_refs) : null,
      title_locale: m.title_locale ? JSON.parse(m.title_locale) : null,
    })),
    chapters: chapters.map(c => ({
      id:                 c.id,
      number_norm:        c.number_norm,
      label:              c.label,
      translations:       c.translations,
      uploading_chapters: c.uploading_chapters,
    })),
    viewer_entry: entry ? {
      entry_id:    Number(entry.id),
      status:      entry.status,
      target_lang: entry.target_lang,
    } : null,
  });
});

// ── POST ──────────────────────────────────────────────────────────────

router.post("", async (ctx) => {
  const userId = ctx.get("userId");
  const body = await ctx.req.json<{
    title:       string;
    cover_url?:  string | null;
    target_lang?: string;
  }>();

  if (!body.title) {
    return ctx.json({ error: "title is required" }, 400);
  }

  const targetLang = body.target_lang ?? "vi";
  const { work_id } = await createBlankWork(ctx.env.DB, {
    user_id:     userId,
    title:       body.title,
    cover_url:   body.cover_url,
    target_lang: targetLang,
  });

  // Return the full detail of the newly created Work
  const work = await requireWork(ctx.env.DB, work_id);
  const materials = await listMaterialsForWork(ctx.env.DB, work_id);
  const entry = await findEntryForWork(ctx.env.DB, userId, work_id);

  return ctx.json({
    work: {
      id:         Number(work.id),
      cross_refs: null,
      created_at: work.created_at,
      updated_at: work.updated_at,
    },
    materials: materials.map(m => ({
      ...m,
      languages:  ["vi"],
      title_alt:  [],
      cross_refs: null,
      title_locale: null,
    })),
    chapters: [],
    viewer_entry: entry ? {
      entry_id:    Number(entry.id),
      status:      entry.status,
      target_lang: entry.target_lang,
    } : null,
  }, 201);
});

// ── DELETE /:work_id ──────────────────────────────────────────────────

router.delete("/:work_id", async (ctx) => {
  const userId = ctx.get("userId");
  const workId = Number(ctx.req.param("work_id"));

  try {
    await deleteUserWork(ctx.env.DB, workId, userId);
    return ctx.body(null, 204);
  } catch (err: any) {
    return ctx.json({ error: err.message }, err.message.includes("source-backed") ? 403 : 400);
  }
});

// ── DELETE /:work_id/my-upload ────────────────────────────────────────

router.delete("/:work_id/my-upload", async (ctx) => {
  const userId = ctx.get("userId");
  const workId = Number(ctx.req.param("work_id"));

  await deleteUserUploadMaterial(ctx.env.DB, { work_id: workId, user_id: userId });
  return ctx.body(null, 204);
});

// ── POST /:work_id/upload-init ────────────────────────────────────────

router.post("/:work_id/upload-init", async (ctx) => {
  const userId = ctx.get("userId");
  const workId = Number(ctx.req.param("work_id"));
  const body = await ctx.req.json<{
    byte_size: number;
  }>();

  if (!body.byte_size || body.byte_size <= 0) {
    return ctx.json({ error: "byte_size ge 1 required" }, 400);
  }

  await requireWork(ctx.env.DB, workId);

  const materialId = await getOrCreateUploadMaterial(ctx.env.DB, {
    work_id:     workId,
    imported_by: userId,
  });

  const MAX_PART_SIZE = 100_000_000;
  const partCount = Math.ceil(body.byte_size / MAX_PART_SIZE);
  const tmp_key = `raw/tmp/${userId}/${crypto.randomUUID()}.zip`;

  // Start R2 Multipart
  const upload = await ctx.env.R2.createMultipartUpload(tmp_key);

  const parts = [];
  for (let i = 1; i <= partCount; i++) {
    parts.push({ number: i, url: "" });
  }

  return ctx.json({
    material_id: materialId,
    tmp_id:      tmp_key,
    upload_id:   upload.uploadId,
    parts:       parts,
    part_size:   MAX_PART_SIZE,
    expires_in:  3600,
  });
});

// ── GET /:work_id/link-suggestions ────────────────────────────────────

router.get("/:work_id/link-suggestions", async (ctx) => {
  const userId = ctx.get("userId");
  const workId = Number(ctx.req.param("work_id"));

  await requireWork(ctx.env.DB, workId);

  const votedRows = await listWorkLinkSuggestions(ctx.env.DB, workId);
  const votedKeys = new Set(votedRows.map(r => Number(r.candidate_material_id)));
  const rankedRows = await listWorkLinkCandidates(ctx.env.DB, workId);

  const out = [];

  // 1. Existing votes
  for (const r of votedRows) {
    const ownId = Number(r.own_material_id);
    const candId = Number(r.candidate_material_id);
    const viewerVote = await getLinkVote(ctx.env.DB, userId, ownId, candId);
    out.push({
      kind: "voted",
      candidate_material_id: candId,
      candidate_title:       r.candidate_title,
      candidate_source:      r.candidate_source,
      candidate_cover:       r.candidate_cover,
      candidate_work_id:     Number(r.candidate_work_id),
      own_material_id:       ownId,
      score:                 Number(r.score),
      total_votes:           Number(r.total),
      viewer_vote:           viewerVote,
    });
  }

  // 2. Ranked candidates from title similarity
  for (const r of rankedRows) {
    const candId = Number(r.candidate_material_id);
    if (votedKeys.has(candId)) continue;
    const ownId = Number(r.own_material_id);
    const viewerVote = await getLinkVote(ctx.env.DB, userId, ownId, candId);
    out.push({
      kind: "ranked",
      candidate_material_id: candId,
      candidate_title:       r.candidate_title,
      candidate_source:      r.candidate_source,
      candidate_cover:       r.candidate_cover,
      candidate_work_id:     Number(r.candidate_work_id),
      own_material_id:       ownId,
      confidence:            Number(r.score),
      reason:                r.reason,
      viewer_vote:           viewerVote,
    });
  }

  return ctx.json(out);
});

// ── POST /:work_id/link-vote ──────────────────────────────────────────

router.post("/:work_id/link-vote", async (ctx) => {
  const userId = ctx.get("userId");
  const workId = Number(ctx.req.param("work_id"));
  const body = await ctx.req.json<{
    target_material_id: number;
    vote:               -1 | 1;
    own_material_id?:    number | null;
  }>();

  if (!body.target_material_id || (body.vote !== -1 && body.vote !== 1)) {
    return ctx.json({ error: "Invalid parameters" }, 400);
  }

  try {
    const [ownId, targetId] = await resolveLinkPair(workId, body.target_material_id, body.own_material_id, ctx.env.DB);
    const result = await castLinkVoteWithMerge(ctx.env.DB, {
      voter_id:      userId,
      material_a_id: ownId,
      material_b_id: targetId,
      vote:          body.vote,
      threshold:     LINK_MERGE_THRESHOLD,
    });

    return ctx.json({
      vote:              Number(result.vote),
      score:             Number(result.score),
      merged:            Boolean(result.merged),
      canonical_work_id: result.canonical_work_id ? Number(result.canonical_work_id) : null,
      blocked_reason:    result.blocked_reason,
    });
  } catch (err: any) {
    return ctx.json({ error: err.message }, err.status || 500);
  }
});

// ── POST /:work_id/propose-link ───────────────────────────────────────

router.post("/:work_id/propose-link", async (ctx) => {
  const userId = ctx.get("userId");
  const workId = Number(ctx.req.param("work_id"));
  const body = await ctx.req.json<{
    target_material_id: number;
    own_material_id?:    number | null;
  }>();

  if (!body.target_material_id) {
    return ctx.json({ error: "target_material_id required" }, 400);
  }

  try {
    const [ownId, targetId] = await resolveLinkPair(workId, body.target_material_id, body.own_material_id, ctx.env.DB);
    const result = await castLinkVoteWithMerge(ctx.env.DB, {
      voter_id:      userId,
      material_a_id: ownId,
      material_b_id: targetId,
      vote:          1,
      threshold:     LINK_MERGE_THRESHOLD,
    });

    return ctx.json({
      vote:              Number(result.vote),
      score:             Number(result.score),
      merged:            Boolean(result.merged),
      canonical_work_id: result.canonical_work_id ? Number(result.canonical_work_id) : null,
      blocked_reason:    result.blocked_reason,
    });
  } catch (err: any) {
    return ctx.json({ error: err.message }, err.status || 500);
  }
});

// ── POST /:work_id/force-link ─────────────────────────────────────────

router.post("/:work_id/force-link", async (ctx) => {
  const userId = ctx.get("userId");
  const workId = Number(ctx.req.param("work_id"));
  const body = await ctx.req.json<{
    target_material_id: number;
    own_material_id?:    number | null;
  }>();

  if (!body.target_material_id) {
    return ctx.json({ error: "target_material_id required" }, 400);
  }

  try {
    const [ownId, targetId] = await resolveLinkPair(workId, body.target_material_id, body.own_material_id, ctx.env.DB);
    const result = await castLinkVoteWithMerge(ctx.env.DB, {
      voter_id:      userId,
      material_a_id: ownId,
      material_b_id: targetId,
      vote:          1,
      threshold:     LINK_MERGE_THRESHOLD,
      force_merge:   true,
    });

    if (result.merged && result.canonical_work_id !== undefined && result.canonical_work_id !== null) {
      await logForceAction(ctx.env.DB, {
        actor_id:       userId,
        kind:           "force_link",
        material_a_id:  ownId,
        material_b_id:  targetId,
        target_work_id: Number(result.canonical_work_id),
      });
    }

    return ctx.json({
      vote:              Number(result.vote),
      score:             Number(result.score),
      merged:            Boolean(result.merged),
      canonical_work_id: result.canonical_work_id ? Number(result.canonical_work_id) : null,
      blocked_reason:    result.blocked_reason,
    });
  } catch (err: any) {
    return ctx.json({ error: err.message }, err.status || 500);
  }
});

// Helper for splits
async function requireMember(workId: number, materialId: number, db: any) {
  await requireWork(db, workId);
  const material = await requireMaterial(db, materialId);
  if (Number(material.work_id) !== workId) {
    const err = new Error("material does not belong to this work");
    (err as any).status = 400;
    throw err;
  }
  return material;
}

// ── POST /:work_id/split-vote ─────────────────────────────────────────

router.post("/:work_id/split-vote", async (ctx) => {
  const userId = ctx.get("userId");
  const workId = Number(ctx.req.param("work_id"));
  const body = await ctx.req.json<{
    material_id: number;
    vote:        -1 | 1;
  }>();

  if (!body.material_id || (body.vote !== -1 && body.vote !== 1)) {
    return ctx.json({ error: "Invalid parameters" }, 400);
  }

  try {
    await requireMember(workId, body.material_id, ctx.env.DB);
    const result = await castSplitVoteWithSplit(ctx.env.DB, {
      voter_id:    userId,
      material_id: body.material_id,
      vote:        body.vote,
      threshold:   SPLIT_THRESHOLD,
    });

    return ctx.json({
      vote:           Number(result.vote),
      score:          Number(result.score),
      split:          Boolean(result.split),
      new_work_id:    result.new_work_id ? Number(result.new_work_id) : null,
      blocked_reason: result.blocked_reason,
    });
  } catch (err: any) {
    return ctx.json({ error: err.message }, err.status || 500);
  }
});

// ── POST /:work_id/force-unlink ───────────────────────────────────────

router.post("/:work_id/force-unlink", async (ctx) => {
  const userId = ctx.get("userId");
  const workId = Number(ctx.req.param("work_id"));
  const body = await ctx.req.json<{
    material_id: number;
  }>();

  if (!body.material_id) {
    return ctx.json({ error: "material_id required" }, 400);
  }

  try {
    await requireMember(workId, body.material_id, ctx.env.DB);
    const recent = await getRecentForceLink(ctx.env.DB, userId, body.material_id, FORCE_UNDO_WINDOW_MIN);
    if (!recent) {
      return ctx.json({
        error: "Không có liên kết gần đây để hoàn tác. Dùng 'Báo nhầm nguồn' để đề xuất tách."
      }, 403);
    }

    try {
      const res = await forceUnlinkMaterial(ctx.env.DB, userId, body.material_id);
      return ctx.json({
        vote:           1,
        score:          0,
        split:          true,
        new_work_id:    Number(res.new_work_id),
        blocked_reason: null,
      });
    } catch (err: any) {
      if (err.message === "solo_member") {
        return ctx.json({
          vote:           1,
          score:          0,
          split:          false,
          new_work_id:    null,
          blocked_reason: "solo_member",
        });
      }
      throw err;
    }
  } catch (err: any) {
    return ctx.json({ error: err.message }, err.status || 500);
  }
});

// ── GET /:work_id/members ─────────────────────────────────────────────

router.get("/:work_id/members", async (ctx) => {
  const userId = ctx.get("userId");
  const workId = Number(ctx.req.param("work_id"));

  await requireWork(ctx.env.DB, workId);
  const materials = await listMaterialsForWork(ctx.env.DB, workId);

  const out = [];
  for (const m of materials) {
    const mid = Number(m.id);
    const viewerVote = await getSplitVote(ctx.env.DB, userId, mid);
    const score = await getSplitScore(ctx.env.DB, mid);
    const recent = await getRecentForceLink(ctx.env.DB, userId, mid, FORCE_UNDO_WINDOW_MIN);

    let undoExpires: string | null = null;
    if (recent) {
      // Add minutes in JS:
      const created = new Date(recent.created_at + "Z"); // SQLite timestamps are UTC ISO strings without 'Z' usually, but in WAL they are UTC
      const expires = new Date(created.getTime() + FORCE_UNDO_WINDOW_MIN * 60 * 1000);
      undoExpires = expires.toISOString().replace(/\.\d+Z$/, "Z");
    }

    out.push({
      material_id:                mid,
      title:                      m.title,
      cover_url:                  m.cover_url,
      source:                     m.source,
      languages:                  m.languages ? JSON.parse(m.languages) : [],
      title_native:               m.title_native,
      title_locale:               m.title_locale ? JSON.parse(m.title_locale) : null,
      viewer_split_vote:          viewerVote,
      pending_split_score:        Number(score.score),
      pending_split_threshold:    SPLIT_THRESHOLD,
      force_link_undo_expires_at: undoExpires,
    });
  }

  return ctx.json(out);
});

export default router;
