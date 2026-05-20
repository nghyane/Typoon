import { Hono } from "hono";
import type { Env, ContextVars } from "../types";
import {
  getTranslatorMemory,
  upsertTranslatorMemory,
  deleteTranslatorMemory,
  listRecentMemoryBriefs,
} from "../store/memory";
import { getMaterial } from "../store/materials";
import { NotFoundError } from "../store/db";

const router = new Hono<{ Bindings: Env; Variables: ContextVars }>();

// Helper to assert material exists
async function requireMaterial(db: any, id: number) {
  const mat = await getMaterial(db, id);
  if (!mat) throw new NotFoundError("Material not found");
  return mat;
}

// ── GET /:material_id/memory ──────────────────────────────────────────

router.get("/:material_id/memory", async (ctx) => {
  const userId = ctx.get("userId");
  const materialId = Number(ctx.req.param("material_id"));
  const targetLang = ctx.req.query("target_lang");

  if (!targetLang || targetLang.length < 2 || targetLang.length > 8) {
    return ctx.json({ error: "target_lang of length 2-8 is required" }, 400);
  }

  await requireMaterial(ctx.env.DB, materialId);

  const row = await getTranslatorMemory(ctx.env.DB, {
    user_id: userId,
    material_id: materialId,
    target_lang: targetLang,
  });

  return ctx.json(row);
});

// ── PUT /:material_id/memory ──────────────────────────────────────────

router.put("/:material_id/memory", async (ctx) => {
  const userId = ctx.get("userId");
  const materialId = Number(ctx.req.param("material_id"));
  const body = await ctx.req.json<{
    source_lang?: string | null;
    target_lang:  string;
    characters?:  any[] | null;
    world?:       Record<string, any> | null;
    style?:       Record<string, any> | null;
    glossary?:    any[] | null;
    style_refs?:  any[] | null;
  }>();

  if (!body.target_lang) {
    return ctx.json({ error: "target_lang is required" }, 400);
  }

  await requireMaterial(ctx.env.DB, materialId);

  const existing = await getTranslatorMemory(ctx.env.DB, {
    user_id: userId,
    material_id: materialId,
    target_lang: body.target_lang,
  });

  if (!existing && !body.source_lang) {
    return ctx.json({ error: "source_lang is required when creating memory for the first time" }, 400);
  }

  const sourceLang = body.source_lang || existing!.source_lang;

  const row = await upsertTranslatorMemory(ctx.env.DB, {
    user_id: userId,
    material_id: materialId,
    source_lang: sourceLang,
    target_lang: body.target_lang,
    characters: body.characters,
    world: body.world,
    style: body.style,
    glossary: body.glossary,
    style_refs: body.style_refs,
  });

  return ctx.json(row);
});

// ── DELETE /:material_id/memory ───────────────────────────────────────

router.delete("/:material_id/memory", async (ctx) => {
  const userId = ctx.get("userId");
  const materialId = Number(ctx.req.param("material_id"));
  const targetLang = ctx.req.query("target_lang");

  if (!targetLang || targetLang.length < 2 || targetLang.length > 8) {
    return ctx.json({ error: "target_lang of length 2-8 is required" }, 400);
  }

  await requireMaterial(ctx.env.DB, materialId);

  await deleteTranslatorMemory(ctx.env.DB, {
    user_id: userId,
    material_id: materialId,
    target_lang: targetLang,
  });

  return ctx.body(null, 204);
});

// ── GET /:material_id/memory/briefs ───────────────────────────────────

router.get("/:material_id/memory/briefs", async (ctx) => {
  const userId = ctx.get("userId");
  const materialId = Number(ctx.req.param("material_id"));
  const targetLang = ctx.req.query("target_lang");
  const beforeChapterId = ctx.req.query("before_chapter_id") ? Number(ctx.req.query("before_chapter_id")) : null;
  const limit = Math.min(50, Math.max(1, Number(ctx.req.query("limit") ?? 5)));

  if (!targetLang) {
    return ctx.json({ error: "target_lang is required" }, 400);
  }

  await requireMaterial(ctx.env.DB, materialId);

  const mem = await getTranslatorMemory(ctx.env.DB, {
    user_id: userId,
    material_id: materialId,
    target_lang: targetLang,
  });

  if (!mem) return ctx.json([]);

  const rows = await listRecentMemoryBriefs(ctx.env.DB, {
    memory_id: mem.id,
    before_chapter_id: beforeChapterId,
    limit,
  });

  return ctx.json(rows);
});

export default router;
