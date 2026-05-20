import { Hono } from "hono";
import type { Env, ContextVars } from "../types";
import {
  listUserGlossary,
  upsertUserGlossaryTerm,
  deleteUserGlossaryTerm,
} from "../store/glossary";

const router = new Hono<{ Bindings: Env; Variables: ContextVars }>();

// ── GET / ─────────────────────────────────────────────────────────────

router.get("", async (ctx) => {
  const userId = ctx.get("userId");
  const sourceLang = ctx.req.query("source_lang");
  const targetLang = ctx.req.query("target_lang");

  const rows = await listUserGlossary(ctx.env.DB, userId, sourceLang, targetLang);
  return ctx.json(rows);
});

// ── POST / ────────────────────────────────────────────────────────────

router.post("", async (ctx) => {
  const userId = ctx.get("userId");
  const body = await ctx.req.json<{
    source_lang: string;
    target_lang: string;
    source_term: string;
    target_term: string;
    notes?:       string | null;
  }>();

  const sourceTerm = (body.source_term ?? "").trim();
  const targetTerm = (body.target_term ?? "").trim();

  if (!body.source_lang || !body.target_lang || !sourceTerm || !targetTerm) {
    return ctx.json({ error: "source_lang, target_lang, source_term and target_term required" }, 400);
  }

  const termId = await upsertUserGlossaryTerm(ctx.env.DB, {
    user_id: userId,
    source_lang: body.source_lang,
    target_lang: body.target_lang,
    source_term: sourceTerm,
    target_term: targetTerm,
    notes: body.notes,
  });

  return ctx.json({
    id: termId,
    source_lang: body.source_lang,
    target_lang: body.target_lang,
    source_term: sourceTerm,
    target_term: targetTerm,
    notes: body.notes ?? null,
  }, 201);
});

// ── DELETE /:term_id ──────────────────────────────────────────────────

router.delete("/:term_id", async (ctx) => {
  const userId = ctx.get("userId");
  const termId = Number(ctx.req.param("term_id"));

  const ok = await deleteUserGlossaryTerm(ctx.env.DB, userId, termId);
  if (!ok) return ctx.json({ error: "Term not found" }, 404);

  return ctx.body(null, 204);
});

export default router;
