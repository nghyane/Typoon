import { Hono } from "hono";
import type { Env, ContextVars } from "../types";
import { hashApiToken } from "../middleware/auth";
import {
  getUser,
  updateUserPreferences,
  createApiToken,
  listActiveApiTokens,
  revokeApiToken,
} from "../store/users";
import {
  listRecentReads,
  recordReading,
} from "../store/library";
import { getQuotaSnapshot } from "../store/quota";
import { getTranslation } from "../store/translations";
import { getDraft } from "../store/drafts";
import { getChapter } from "../store/chapters";
import { getMaterial } from "../store/materials";
import { findOrCreateWorkChapter } from "../store/works";

const router = new Hono<{ Bindings: Env; Variables: ContextVars }>();

// ── PATCH /preferences ───────────────────────────────────────────────

router.patch("/preferences", async (ctx) => {
  const userId = ctx.get("userId");
  const body = await ctx.req.json<{ preferred_target_lang?: string | null }>();

  if (body.preferred_target_lang !== undefined) {
    await updateUserPreferences(ctx.env.DB, userId, body.preferred_target_lang);
  }

  const user = await getUser(ctx.env.DB, userId);
  if (!user) return ctx.json({ error: "User not found" }, 404);

  return ctx.json({
    id: user.id,
    display_name: user.display_name,
    avatar_url: user.avatar_url,
    email: user.email,
    preferred_target_lang: user.preferred_target_lang,
    roles: ctx.get("jwtRoles") ?? [],
  });
});

// ── GET /recent-reads ────────────────────────────────────────────────

router.get("/recent-reads", async (ctx) => {
  const userId = ctx.get("userId");
  const limit = Math.min(100, Number(ctx.req.query("limit") ?? 30));

  const rows = await listRecentReads(ctx.env.DB, { user_id: userId, limit });
  return ctx.json(rows);
});

// ── POST /reading/translated ─────────────────────────────────────────

router.post("/reading/translated", async (ctx) => {
  const userId = ctx.get("userId");
  const body = await ctx.req.json<{ translation_id: number }>();

  if (!body.translation_id) {
    return ctx.json({ error: "translation_id required" }, 400);
  }

  const trans = await getTranslation(ctx.env.DB, body.translation_id);
  if (!trans) return ctx.json({ error: "Translation not found" }, 404);

  const draft = await getDraft(ctx.env.DB, trans.draft_id);
  if (!draft) return ctx.json({ error: "Translation draft missing" }, 500);

  const chapter = await getChapter(ctx.env.DB, draft.chapter_id);
  if (!chapter) return ctx.json({ error: "Draft chapter missing" }, 500);

  await recordReading(ctx.env.DB, {
    user_id: userId,
    work_chapter_id: trans.work_chapter_id,
    last_material_id: chapter.material_id,
    translation_id: body.translation_id,
  });

  return ctx.body(null, 204);
});

// ── POST /reading/raw ────────────────────────────────────────────────

router.post("/reading/raw", async (ctx) => {
  const userId = ctx.get("userId");
  const body = await ctx.req.json<{
    material_id: number;
    number:      string;
    number_norm: string;
    label?:      string | null;
  }>();

  if (!body.material_id || !body.number_norm) {
    return ctx.json({ error: "material_id and number_norm required" }, 400);
  }

  const material = await getMaterial(ctx.env.DB, body.material_id);
  if (!material) return ctx.json({ error: "Material not found" }, 404);

  const workChapterId = await findOrCreateWorkChapter(ctx.env.DB, {
    work_id: Number(material.work_id),
    number_norm: body.number_norm,
    label: body.label,
  });

  await recordReading(ctx.env.DB, {
    user_id: userId,
    work_chapter_id: workChapterId,
    last_material_id: body.material_id,
    translation_id: null,
  });

  return ctx.body(null, 204);
});

// ── GET /tokens ──────────────────────────────────────────────────────

router.get("/tokens", async (ctx) => {
  const userId = ctx.get("userId");
  const rows = await listActiveApiTokens(ctx.env.DB, userId);
  return ctx.json(
    rows.map(r => ({
      id: r.id,
      name: r.name,
      prefix: r.prefix,
      last_used: r.last_used,
      created_at: r.created_at,
    }))
  );
});

// ── POST /tokens ─────────────────────────────────────────────────────

function generateToken(): { plaintext: string; prefix: string } {
  const bytes = new Uint8Array(24);
  crypto.getRandomValues(bytes);
  let body = btoa(String.fromCharCode(...bytes))
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/, "");
  body = body.slice(0, 32);
  const plaintext = "typ_" + body;
  const prefix = body.slice(0, 8);
  return { plaintext, prefix };
}


router.post("/tokens", async (ctx) => {
  const userId = ctx.get("userId");
  const body = await ctx.req.json<{ name: string }>();

  const name = (body.name ?? "").trim();
  if (!name) return ctx.json({ error: "name required" }, 400);

  const { plaintext, prefix } = generateToken();
  const tokenHash = await hashApiToken(plaintext);

  const row = await createApiToken(ctx.env.DB, {
    user_id: userId,
    name,
    token_hash: tokenHash,
    prefix,
    scopes: [],
  });

  return ctx.json({
    id: row.id,
    name: row.name,
    prefix: row.prefix,
    last_used: row.last_used,
    created_at: row.created_at,
    token: plaintext,
  }, 201);
});

// ── DELETE /tokens/:token_id ──────────────────────────────────────────

router.delete("/tokens/:token_id", async (ctx) => {
  const userId = ctx.get("userId");
  const tokenId = Number(ctx.req.param("token_id"));

  const ok = await revokeApiToken(ctx.env.DB, userId, tokenId);
  if (!ok) return ctx.json({ error: "Token not found" }, 404);

  return ctx.body(null, 204);
});

// ── GET /quota ───────────────────────────────────────────────────────

router.get("/quota", async (ctx) => {
  const userId = ctx.get("userId");
  const roles = ctx.get("jwtRoles") ?? [];
  const adminRoleId = ctx.env.ADMIN_ROLE_ID;

  const snapshot = await getQuotaSnapshot(ctx.env.DB, {
    userId,
    roles,
    adminRoleId,
  });

  return ctx.json(snapshot);
});

export default router;
