/**
 * Upload routes — R2 multipart upload for chapter archives.
 *
 * POST /api/uploads/presign   → create multipart upload, return metadata
 * POST /api/uploads/finalize  → complete multipart, create chapter, spawn pipeline
 *
 * Canonical R2 key: raw/{chapter_id}/source.zip
 * Pipeline always receives the canonical key — tmp keys never leave this file.
 */

import { Hono } from "hono";
import type { Env, ContextVars } from "../types";
import { createChapter } from "../store/chapters";
import { findOrCreateDraft, computeGlossaryFp } from "../store/drafts";

type AppEnv = { Bindings: Env; Variables: ContextVars };
const router = new Hono<AppEnv>();

const MAX_PART_SIZE = 100_000_000; // 100 MB per part (R2 max)
const DEFAULT_LLM = "gemini-2.5-flash";

// ── POST /presign ───────────────────────────────────────────────────

router.post("/presign", async (ctx) => {
  const userId = ctx.get("userId");
  const body = await ctx.req.json<{
    material_id: number;
    byte_size: number;
  }>();

  if (!body.material_id || !body.byte_size || body.byte_size <= 0) {
    return ctx.json(
      { error: { code: "bad_request", message: "material_id and byte_size (>= 1) required" } },
      400,
    );
  }

  const tmpKey = `raw/tmp/${userId}/${crypto.randomUUID()}.zip`;
  const partCount = Math.ceil(body.byte_size / MAX_PART_SIZE);
  const upload = await ctx.env.R2.createMultipartUpload(tmpKey);

  // R2 Bindings don't expose presigned PUT URLs.
  // Client uses S3-compatible endpoint with AWS4-HMAC-SHA256 signing:
  //   PUT https://<account>.r2.cloudflarestorage.com/<bucket>/<tmpKey>
  //        ?partNumber=N&uploadId=<upload_id>
  return ctx.json({
    tmp_key:     tmpKey,
    upload_id:   upload.uploadId,
    part_count:  partCount,
    part_size:   MAX_PART_SIZE,
    expires_in:  3600,
  });
});

// ── POST /finalize ──────────────────────────────────────────────────

router.post("/finalize", async (ctx) => {
  const userId = ctx.get("userId");
  const body = await ctx.req.json<{
    material_id:     number;
    tmp_key:         string;
    upload_id:       string;
    parts:           Array<{ number: number; etag: string }>;
    work_chapter_id: number;
    position:        number;
    label?:          string;
    source_lang?:    string;
    target_lang:     string;
    upstream_url?:   string;
  }>();

  const {
    material_id, tmp_key, upload_id, parts,
    work_chapter_id, position, label, source_lang,
    target_lang, upstream_url,
  } = body;

  if (!material_id || !tmp_key || !upload_id || !parts?.length || !work_chapter_id || !target_lang) {
    return ctx.json(
      { error: { code: "bad_request", message: "Missing required fields" } },
      400,
    );
  }

  // 1. Complete R2 multipart upload
  const upload = ctx.env.R2.resumeMultipartUpload(tmp_key, upload_id);
  await upload.complete(parts.map(p => ({ partNumber: p.number, etag: p.etag })));

  // 2. Create chapter row → stable chapter_id
  const chapter = await createChapter(ctx.env.DB, {
    material_id,
    work_chapter_id,
    position,
    label,
    upstream_url,
    source_lang,
  });

  // 3. Move tmp → canonical key: raw/{chapter_id}/source.zip
  const canonicalKey = `raw/${chapter.id}/source.zip`;
  const tmpObj = await ctx.env.R2.get(tmp_key);
  if (!tmpObj) {
    throw new Error(`R2 object ${tmp_key} not found after multipart complete`);
  }
  await ctx.env.R2.put(canonicalKey, tmpObj.body, {
    httpMetadata: { contentType: "application/zip" },
  });
  ctx.executionCtx.waitUntil(ctx.env.R2.delete(tmp_key));

  // 4. Glossary fingerprint
  const srcLang = source_lang ?? "ja";
  const glossaryFp = await computeGlossaryFp(ctx.env.DB, userId, srcLang, target_lang);

  // 5. Find or create draft (cache-pool)
  const { draft, cache_hit } = await findOrCreateDraft(ctx.env.DB, {
    chapter_id:  chapter.id,
    source_lang: srcLang,
    target_lang,
    glossary_fp: glossaryFp,
    llm_model:   DEFAULT_LLM,
    created_by:  userId,
  });

  // 6. Create per-user translation row
  const translation = await ctx.env.DB
    .prepare(
      `INSERT INTO translations (work_chapter_id, owner_id, target_lang, draft_id, shared)
       VALUES (?, ?, ?, ?, 1)
       RETURNING id`,
    )
    .bind(chapter.work_chapter_id, userId, target_lang, draft.id)
    .first<{ id: number }>();

  if (!translation) throw new Error("Translation insert failed");

  // 7. Record quota consume (cache miss = new LLM run)
  if (!cache_hit) {
    await ctx.env.DB
      .prepare(
        `INSERT INTO chapter_consumes (user_id, translation_id, kind)
         VALUES (?, ?, 'draft_create')`,
      )
      .bind(userId, translation.id)
      .run();
  }

  // 8. Spawn pipeline (cache miss only)
  if (!cache_hit) {
    const params = {
      chapter_id:  chapter.id,
      draft_id:    draft.id,
      zip_key:     canonicalKey,
      source_lang: srcLang,
      target_lang,
      glossary_fp: glossaryFp,
    };
    const res = await ctx.env.PIPELINE.fetch("http://pipeline/start", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(params),
    });
    if (!res.ok) {
      throw new Error(`Failed to spawn pipeline: ${await res.text()}`);
    }
  }

  return ctx.json({
    chapter_id:     chapter.id,
    draft_id:       draft.id,
    translation_id: translation.id,
    state:          draft.state,
    cache_hit,
  }, 201);
});

export default router;
