/**
 * Translations routes — spawn, list, detail, bubbles, edits, redo, delete.
 *
 * POST   /api/translations              → spawn translation
 * GET    /api/translations              → list my translations
 * GET    /api/translations/:id          → translation detail
 * GET    /api/translations/:id/bubbles  → bubble edit data
 * PUT    /api/translations/:id/edits    → upsert bubble text
 * DELETE /api/translations/:id/edits/:page/:bubble → remove edit
 * POST   /api/translations/:id/redo     → re-run pipeline
 * DELETE /api/translations/:id          → delete translation
 * WS     /api/translations/:id/ws       → progress stream
 */

import { Hono } from "hono";
import type { Env, ContextVars, SpawnTranslateResult, ApiMyTranslation, ApiTranslation, ApiBubbleEdit } from "../types";
import { findOrCreateDraft, computeGlossaryFp } from "../store/drafts";
import {
  listMyTranslations,
  getTranslationWithDetails,
  getTranslationEdits,
  upsertTranslationEdit,
  deleteTranslationEdit,
  redoTranslation,
  deleteTranslation,
  type MyTranslationRow,
} from "../store/translations";
import { getChapter } from "../store/chapters";

type AppEnv = { Bindings: Env; Variables: ContextVars };
const router = new Hono<AppEnv>();

const DEFAULT_LLM = "gemini-2.5-flash";

// ── Helpers ─────────────────────────────────────────────────────────

/** Spawn pipeline for a draft. Throws if the pipeline rejects. */
async function spawnPipeline(
  env: Env,
  args: {
    chapter_id:  number;
    draft_id:    number;
    source_lang: string;
    target_lang: string;
    glossary_fp: string;
  },
): Promise<void> {
  const canonicalKey = `raw/${args.chapter_id}/source.zip`;
  const params = {
    chapter_id:  args.chapter_id,
    draft_id:    args.draft_id,
    zip_key:     canonicalKey,
    source_lang: args.source_lang,
    target_lang: args.target_lang,
    glossary_fp: args.glossary_fp,
  };
  const res = await env.PIPELINE.fetch("http://pipeline/start", {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify(params),
  });
  if (!res.ok) {
    throw new Error(`Failed to spawn pipeline: ${await res.text()}`);
  }
}

// ── POST / — spawn translation ──────────────────────────────────────

router.post("", async (ctx) => {
  const userId = ctx.get("userId");
  const body = await ctx.req.json<{
    chapter_id:  number;
    target_lang: string;
  }>();

  if (!body.chapter_id || !body.target_lang) {
    return ctx.json(
      { error: { code: "bad_request", message: "chapter_id and target_lang required" } },
      400,
    );
  }

  const chapter = await getChapter(ctx.env.DB, body.chapter_id);
  if (!chapter) {
    return ctx.json(
      { error: { code: "not_found", message: "Chapter not found" } },
      404,
    );
  }

  const sourceLang = chapter.source_lang ?? "ja";
  const glossaryFp = await computeGlossaryFp(ctx.env.DB, userId, sourceLang, body.target_lang);

  const { draft, cache_hit } = await findOrCreateDraft(ctx.env.DB, {
    chapter_id:  body.chapter_id,
    source_lang: sourceLang,
    target_lang: body.target_lang,
    glossary_fp: glossaryFp,
    llm_model:   DEFAULT_LLM,
    created_by:  userId,
  });

  // Create per-user translation row
  let translationId: number | null = null;
  const translation = await ctx.env.DB
    .prepare(
      `INSERT INTO translations (work_chapter_id, owner_id, target_lang, draft_id, shared)
       VALUES (?, ?, ?, ?, 1)
       ON CONFLICT (work_chapter_id, owner_id, draft_id) DO UPDATE
         SET shared = translations.shared
       RETURNING id`,
    )
    .bind(chapter.work_chapter_id, userId, body.target_lang, draft.id)
    .first<{ id: number }>();

  if (translation) {
    translationId = translation.id;
  } else {
    // Fallback: row already existed
    const existing = await ctx.env.DB
      .prepare(
        `SELECT id FROM translations
         WHERE work_chapter_id = ? AND owner_id = ? AND draft_id = ?`,
      )
      .bind(chapter.work_chapter_id, userId, draft.id)
      .first<{ id: number }>();

    if (!existing) throw new Error("Failed to resolve translation_id");
    translationId = existing.id;
  }

  // Spawn pipeline only on cache miss with pending draft
  if (!cache_hit && draft.state === "pending") {
    await spawnPipeline(ctx.env, {
      chapter_id:  body.chapter_id,
      draft_id:    draft.id,
      source_lang: sourceLang,
      target_lang: body.target_lang,
      glossary_fp: glossaryFp,
    });

    // Record quota consume
    await ctx.env.DB
      .prepare(
        `INSERT INTO chapter_consumes (user_id, translation_id, kind)
         VALUES (?, ?, 'draft_create')`,
      )
      .bind(userId, translationId)
      .run();
  }

  const result: SpawnTranslateResult = {
    translation_id: translationId,
    draft_id:       draft.id,
    state:          draft.state,
    cache_hit,
    chapter_id:     body.chapter_id,
  };

  return ctx.json(result, cache_hit ? 200 : 201);
});

// ── GET / — list my translations ────────────────────────────────────

router.get("", async (ctx) => {
  const userId = ctx.get("userId");
  const rows = await listMyTranslations(ctx.env.DB, userId);

  const items: ApiMyTranslation[] = rows.map((r: MyTranslationRow) => ({
    translation_id:       r.translation_id,
    target_lang:          r.target_lang,
    state:                r.state as ApiMyTranslation["state"],
    has_archive:          r.has_archive === 1,
    updated_at:           r.updated_at,
    chapter_id:           r.chapter_id,
    chapter_number:       r.chapter_number,
    chapter_label:        r.chapter_label,
    chapter_position:     r.chapter_position,
    chapter_upstream_url: r.chapter_upstream_url,
    material_id:          r.material_id,
    material_title:       r.material_title,
    material_cover:       r.material_cover,
    material_source:      r.material_source,
    material_upstream_ref: r.material_upstream_ref,
  }));

  return ctx.json(items);
});

// ── GET /:id — translation detail ───────────────────────────────────

router.get("/:id", async (ctx) => {
  const userId = ctx.get("userId");
  const id = Number(ctx.req.param("id"));
  if (!id || isNaN(id)) {
    return ctx.json(
      { error: { code: "bad_request", message: "Invalid translation ID" } },
      400,
    );
  }

  const detail = await getTranslationWithDetails(ctx.env.DB, id, userId);
  if (!detail) {
    return ctx.json(
      { error: { code: "not_found", message: "Translation not found" } },
      404,
    );
  }

  const result: ApiTranslation = {
    id:               detail.id,
    work_id:          detail.work_id,
    work_chapter_id:  detail.work_chapter_id,
    chapter_id:       detail.chapter_id,
    material_id:      detail.material_id,
    owner_id:         detail.owner_id,
    target_lang:      detail.target_lang,
    draft_id:         detail.draft_id,
    state:            detail.state,
    archive_url:      detail.archive_url,
    has_edits:        detail.has_edits,
    chapter_number:   detail.chapter_number,
    chapter_label:    detail.chapter_label,
    material_title:   detail.material_title,
    shared:           detail.shared,
    created_at:       detail.created_at,
    updated_at:       detail.updated_at,
  };

  return ctx.json(result);
});

// ── GET /:id/bubbles — bubble edit data ─────────────────────────────

router.get("/:id/bubbles", async (ctx) => {
  const userId = ctx.get("userId");
  const id = Number(ctx.req.param("id"));
  if (!id || isNaN(id)) {
    return ctx.json(
      { error: { code: "bad_request", message: "Invalid translation ID" } },
      400,
    );
  }

  // Verify access
  const detail = await getTranslationWithDetails(ctx.env.DB, id, userId);
  if (!detail) {
    return ctx.json(
      { error: { code: "not_found", message: "Translation not found" } },
      404,
    );
  }

  const edits = await getTranslationEdits(ctx.env.DB, id);

  // Build bubble list from edits + draft data
  // For now return edits as bubbles — full bubble data comes from draft archive
  const bubbles: ApiBubbleEdit[] = edits.map(e => ({
    page_index:  e.page_index,
    bubble_idx:  e.bubble_idx,
    source_text: "",     // from draft archive
    draft_text:  "",     // from draft archive
    edited_text: e.edited_text,
    kind:        "dialogue",
  }));

  return ctx.json(bubbles);
});

// ── PUT /:id/edits — upsert bubble text ─────────────────────────────

router.put("/:id/edits", async (ctx) => {
  const userId = ctx.get("userId");
  const id = Number(ctx.req.param("id"));
  if (!id || isNaN(id)) {
    return ctx.json(
      { error: { code: "bad_request", message: "Invalid translation ID" } },
      400,
    );
  }

  // Verify ownership
  const detail = await getTranslationWithDetails(ctx.env.DB, id, userId);
  if (!detail || detail.owner_id !== userId) {
    return ctx.json(
      { error: { code: "forbidden", message: "Only the owner can edit this translation" } },
      403,
    );
  }

  const body = await ctx.req.json<{
    page_index:  number;
    bubble_idx:  number;
    edited_text: string;
  }>();

  if (body.page_index == null || body.bubble_idx == null || body.edited_text == null) {
    return ctx.json(
      { error: { code: "bad_request", message: "page_index, bubble_idx, edited_text required" } },
      400,
    );
  }

  await upsertTranslationEdit(ctx.env.DB, {
    translation_id: id,
    page_index:     body.page_index,
    bubble_idx:     body.bubble_idx,
    edited_text:    body.edited_text,
  });

  return ctx.body(null, 204);
});

// ── DELETE /:id/edits/:page/:bubble ─────────────────────────────────

router.delete("/:id/edits/:page/:bubble", async (ctx) => {
  const userId = ctx.get("userId");
  const id = Number(ctx.req.param("id"));
  const page = Number(ctx.req.param("page"));
  const bubble = Number(ctx.req.param("bubble"));

  if (!id || isNaN(id) || isNaN(page) || isNaN(bubble)) {
    return ctx.json(
      { error: { code: "bad_request", message: "Invalid parameters" } },
      400,
    );
  }

  // Verify ownership
  const detail = await getTranslationWithDetails(ctx.env.DB, id, userId);
  if (!detail || detail.owner_id !== userId) {
    return ctx.json(
      { error: { code: "forbidden", message: "Only the owner can edit this translation" } },
      403,
    );
  }

  await deleteTranslationEdit(ctx.env.DB, id, page, bubble);
  return ctx.body(null, 204);
});

// ── POST /:id/redo — re-run pipeline ────────────────────────────────

router.post("/:id/redo", async (ctx) => {
  const userId = ctx.get("userId");
  const id = Number(ctx.req.param("id"));
  if (!id || isNaN(id)) {
    return ctx.json(
      { error: { code: "bad_request", message: "Invalid translation ID" } },
      400,
    );
  }

  const result = await redoTranslation(ctx.env.DB, id, userId);
  if (!result) {
    return ctx.json(
      { error: { code: "not_found", message: "Translation not found" } },
      404,
    );
  }

  // If draft was reset to pending, spawn pipeline
  if (result.state === "pending") {
    const draft = await ctx.env.DB
      .prepare(
        `SELECT d.chapter_id, d.source_lang, d.target_lang, d.glossary_fp
         FROM translation_drafts d WHERE d.id = ?`,
      )
      .bind(result.draft_id)
      .first<{ chapter_id: number; source_lang: string; target_lang: string; glossary_fp: string }>();

    if (draft) {
      await spawnPipeline(ctx.env, {
        chapter_id:  draft.chapter_id,
        draft_id:    result.draft_id,
        source_lang: draft.source_lang,
        target_lang: draft.target_lang,
        glossary_fp: draft.glossary_fp,
      });

      const spawnResult: SpawnTranslateResult = {
        translation_id: id,
        draft_id:       result.draft_id,
        state:          result.state,
        cache_hit:      true,
        chapter_id:     draft.chapter_id,
      };

      return ctx.json(spawnResult, 201);
    }
  }

  return ctx.json({
    translation_id: id,
    draft_id:       result.draft_id,
    state:          result.state,
    cache_hit:      true,
    chapter_id:     0,
  }, 201);
});

// ── DELETE /:id — delete translation ────────────────────────────────

router.delete("/:id", async (ctx) => {
  const userId = ctx.get("userId");
  const id = Number(ctx.req.param("id"));
  if (!id || isNaN(id)) {
    return ctx.json(
      { error: { code: "bad_request", message: "Invalid translation ID" } },
      400,
    );
  }

  // Verify ownership
  const t = await ctx.env.DB
    .prepare(
      `SELECT id FROM translations WHERE id = ? AND owner_id = ? AND takedown_at IS NULL`,
    )
    .bind(id, userId)
    .first();

  if (!t) {
    return ctx.json(
      { error: { code: "not_found", message: "Translation not found" } },
      404,
    );
  }

  await deleteTranslation(ctx.env.DB, id);
  return ctx.body(null, 204);
});

// ── WS /:id/ws — progress stream ────────────────────────────────────

router.get("/:id/ws", async (ctx) => {
  const userId = ctx.get("userId");
  const id = Number(ctx.req.param("id"));
  if (!id || isNaN(id)) {
    return ctx.json(
      { error: { code: "bad_request", message: "Invalid translation ID" } },
      400,
    );
  }

  // Verify access
  const t = await ctx.env.DB
    .prepare(
      `SELECT draft_id FROM translations
       WHERE id = ? AND (owner_id = ? OR shared = 1) AND takedown_at IS NULL`,
    )
    .bind(id, userId)
    .first<{ draft_id: number }>();

  if (!t) {
    return ctx.json(
      { error: { code: "not_found", message: "Translation not found" } },
      404,
    );
  }

  if (ctx.req.header("Upgrade") !== "websocket") {
    return ctx.json(
      { error: { code: "upgrade_required", message: "WebSocket upgrade required" } },
      426,
    );
  }

  const stubId = ctx.env.STATUS_DO.idFromName(String(t.draft_id));
  const stub = ctx.env.STATUS_DO.get(stubId);
  return stub.fetch(ctx.req.raw);
});

export default router;
