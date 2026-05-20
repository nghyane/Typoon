/**
 * Library routes — per-user, per-Work bookmark.
 */

import { Hono } from "hono";
import type { Env, ContextVars } from "../types";
import {
  listLibraryEntries,
  getLibraryEntry,
  findEntryForWork,
  createLibraryEntry,
  updateLibraryEntry,
  deleteLibraryEntry,
  linkMaterialToEntry,
  unlinkMaterialFromEntry,
} from "../store/library";
import { getMaterial } from "../store/materials";
import { NotFoundError } from "../store/db";

const router = new Hono<{ Bindings: Env; Variables: ContextVars }>();

// Helper to assert material exists
async function requireMaterial(db: any, id: number) {
  const mat = await getMaterial(db, id);
  if (!mat) throw new NotFoundError("Material not found");
  return mat;
}

// Helper to assert library entry exists and belongs to the caller
async function requireLibraryEntry(db: any, entryId: number, userId: number) {
  const entry = await getLibraryEntry(db, entryId, userId);
  if (!entry) throw new NotFoundError("Library entry not found");
  return entry;
}

// ── GET / ─────────────────────────────────────────────────────────────

router.get("", async (ctx) => {
  const userId = ctx.get("userId");
  const status = ctx.req.query("status"); // reading | plan | done | dropped

  const entries = await listLibraryEntries(ctx.env.DB, userId, status);
  return ctx.json(entries);
});

// ── GET /entry/:entry_id ──────────────────────────────────────────────

router.get("/entry/:entry_id", async (ctx) => {
  const userId = ctx.get("userId");
  const entryId = Number(ctx.req.param("entry_id"));

  const entry = await requireLibraryEntry(ctx.env.DB, entryId, userId);
  return ctx.json(entry);
});

// ── POST /entry ───────────────────────────────────────────────────────

router.post("/entry", async (ctx) => {
  const userId = ctx.get("userId");
  const body = await ctx.req.json<{
    material_id: number;
    target_lang?: string;
    status?:     "reading" | "plan" | "done" | "dropped";
  }>();

  if (!body.material_id) {
    return ctx.json({ error: "material_id required" }, 400);
  }

  const targetLang = body.target_lang ?? "vi";
  const status = body.status ?? "reading";

  const mat = await requireMaterial(ctx.env.DB, body.material_id);
  const workId = Number(mat.work_id);

  const existing = await findEntryForWork(ctx.env.DB, userId, workId);
  let entryId: number;

  if (existing) {
    entryId = Number(existing.id);
    await linkMaterialToEntry(ctx.env.DB, {
      entry_id:    entryId,
      material_id: body.material_id,
      link_origin: "manual",
      voter_id:    userId,
    });
  } else {
    entryId = await createLibraryEntry(ctx.env.DB, {
      user_id:     userId,
      work_id:     workId,
      target_lang: targetLang,
      materials:   [[body.material_id, "manual"]],
      status:      status,
    });
  }

  const entry = await requireLibraryEntry(ctx.env.DB, entryId, userId);
  return ctx.json(entry);
});

// ── PATCH /entry/:entry_id ────────────────────────────────────────────

router.patch("/entry/:entry_id", async (ctx) => {
  const userId = ctx.get("userId");
  const entryId = Number(ctx.req.param("entry_id"));
  const body = await ctx.req.json<{
    status?:      "reading" | "plan" | "done" | "dropped" | null;
    target_lang?: string | null;
  }>();

  await requireLibraryEntry(ctx.env.DB, entryId, userId);

  await updateLibraryEntry(ctx.env.DB, entryId, userId, {
    status:      body.status,
    target_lang: body.target_lang,
  });

  const entry = await requireLibraryEntry(ctx.env.DB, entryId, userId);
  return ctx.json(entry);
});

// ── DELETE /entry/:entry_id ───────────────────────────────────────────

router.delete("/entry/:entry_id", async (ctx) => {
  const userId = ctx.get("userId");
  const entryId = Number(ctx.req.param("entry_id"));

  const ok = await deleteLibraryEntry(ctx.env.DB, entryId, userId);
  if (!ok) {
    return ctx.json({ error: "Library entry not found" }, 404);
  }
  return ctx.body(null, 204);
});

// ── POST /entry/:entry_id/link ────────────────────────────────────────

router.post("/entry/:entry_id/link", async (ctx) => {
  const userId = ctx.get("userId");
  const entryId = Number(ctx.req.param("entry_id"));
  const body = await ctx.req.json<{
    material_id: number;
    link_origin?: string;
  }>();

  if (!body.material_id) {
    return ctx.json({ error: "material_id required" }, 400);
  }

  const linkOrigin = body.link_origin ?? "manual";
  if (linkOrigin !== "auto" && linkOrigin !== "manual") {
    return ctx.json({ error: "invalid link_origin" }, 400);
  }

  await requireLibraryEntry(ctx.env.DB, entryId, userId);
  await requireMaterial(ctx.env.DB, body.material_id);

  await linkMaterialToEntry(ctx.env.DB, {
    entry_id:    entryId,
    material_id: body.material_id,
    link_origin: linkOrigin,
    voter_id:    userId,
  });

  return ctx.body(null, 204);
});

// ── POST /entry/:entry_id/unlink ──────────────────────────────────────

router.post("/entry/:entry_id/unlink", async (ctx) => {
  const userId = ctx.get("userId");
  const entryId = Number(ctx.req.param("entry_id"));
  const body = await ctx.req.json<{
    material_id: number;
  }>();

  if (!body.material_id) {
    return ctx.json({ error: "material_id required" }, 400);
  }

  await requireLibraryEntry(ctx.env.DB, entryId, userId);

  await unlinkMaterialFromEntry(ctx.env.DB, {
    entry_id:    entryId,
    material_id: body.material_id,
    voter_id:    userId,
  });

  return ctx.body(null, 204);
});

export default router;
