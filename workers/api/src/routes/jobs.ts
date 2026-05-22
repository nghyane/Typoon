/**
 * Jobs routes — translation lifecycle.
 *
 *   POST   /jobs                  init: quota check + R2 multipart + presigned URLs
 *                                  hydrates ctx/{id}/input.json.gz from KV when work_id given
 *   POST   /jobs/:id/start        finalize multipart + spawn pipeline
 *   GET    /jobs/:id              poll status (includes presigned archive + context URLs)
 *   GET    /jobs/:id/download     302 to presigned R2 GET (state='done', kind='translate')
 *   DELETE /jobs/:id              cleanup R2 + DB
 *   GET    /me/jobs               paginated job list (last 7d)
 *   GET    /me/quota              tier-aware quota snapshot
 *
 * Two job kinds:
 *   'translate'  full pipeline → archive.bnl + merged context
 *   'analyze'    brief-only    → no archive, merged context only
 * Both consume 1 quota unit and write KV ctx:{user}:{work_id} on success.
 */

import { Hono } from "hono";

import type { Env, ContextVars, ApiQuota } from "../types";
import { getTier } from "../lib/tiers";
import {
  awsClient, presignR2Url, toApiJob, withSignedUrls, DOWNLOAD_TTL,
} from "../lib/api-job";
import {
  createJob, getJobForUser, listJobsForUser,
  setJobState, deleteJob,
  countActiveJobs, countChaptersThisMonth,
} from "../store/jobs";
import { copyContextFromKvToR2 } from "../store/work-context";

type AppEnv = { Bindings: Env; Variables: ContextVars };
const router = new Hono<AppEnv>();

// ── R2 presigned URL constants ──────────────────────────────────────

const MAX_PART_SIZE       = 100_000_000;  // 100 MB — R2 multipart per-part max
const UPLOAD_TTL          = 3_600;        // presigned URL TTL (seconds)
const PAGE_SIZE_HEURISTIC = 1_000_000;    // 1 MB ~= 1 page (rough)

// ── Helpers ─────────────────────────────────────────────────────────

/** Delete every R2 artifact tied to this job_id. Used by DELETE /jobs/:id
 *  (full removal) and POST /jobs/:id/retry (clean slate before re-upload). */
async function cleanupJobR2(env: Env, jobId: number): Promise<void> {
  const prefixes = [
    `raw/${jobId}/`,       `prepared/${jobId}/`,    `scan/${jobId}/`,
    `mask/${jobId}/`,      `inpaint/${jobId}/`,     `archive/${jobId}/`,
    `ctx/${jobId}/`,       `brief/${jobId}/`,       `storyboard/${jobId}/`,
    `typeset/${jobId}/`,   `translate/${jobId}`,
  ];
  for (const prefix of prefixes) {
    const listed = await env.R2.list({ prefix });
    if (listed.objects.length > 0) {
      await env.R2.delete(listed.objects.map(o => o.key));
    }
  }
}

// ── POST /jobs — init ───────────────────────────────────────────────

router.post("/jobs", async (ctx) => {
  const userId = ctx.get("userId");
  const tier   = getTier(ctx.get("tierId"));
  const body   = await ctx.req.json<{
    byte_size:    number;
    source_lang:  string;
    target_lang?: string;
    /** Composite client-side ID (e.g. "mdex:abc-uuid"). When present,
     *  server pulls ctx:{user_id}:{work_id} from KV into ctx/{job_id}/input.json.gz. */
    work_id?:     string;
    /** 'translate' (default) renders full archive; 'analyze' only refreshes context. */
    kind?:        "translate" | "analyze";
  }>();

  if (!body.byte_size || body.byte_size <= 0 || !body.source_lang) {
    return ctx.json({ error: "byte_size and source_lang required" }, 400);
  }

  const kind = body.kind ?? "translate";
  if (kind !== "translate" && kind !== "analyze") {
    return ctx.json({ error: "kind must be 'translate' or 'analyze'" }, 400);
  }

  const estimatedPages = Math.max(1, Math.ceil(body.byte_size / PAGE_SIZE_HEURISTIC));
  if (estimatedPages > tier.max_pages_per_chapter) {
    return ctx.json({
      error:   `Chapter quá lớn: ước tính ${estimatedPages} trang, tier ${tier.name} cho phép ${tier.max_pages_per_chapter}`,
      tier_id: tier.id,
    }, 400);
  }

  const active = await countActiveJobs(ctx.env.DB, userId);
  if (active >= tier.concurrent_jobs) {
    return ctx.json({
      error:   `Tối đa ${tier.concurrent_jobs} job đồng thời (tier ${tier.name})`,
      tier_id: tier.id,
    }, 429);
  }
  const used = await countChaptersThisMonth(ctx.env.DB, userId);
  if (used >= tier.monthly_chapters) {
    return ctx.json({
      error:   `Đã dùng hết quota: ${used}/${tier.monthly_chapters} chương tháng này (tier ${tier.name})`,
      tier_id: tier.id,
    }, 429);
  }

  // Resolve target_lang
  let targetLang = body.target_lang;
  if (!targetLang) {
    const u = await ctx.env.DB
      .prepare("SELECT preferred_target_lang FROM users WHERE id = ?")
      .bind(userId)
      .first<{ preferred_target_lang: string | null }>();
    targetLang = u?.preferred_target_lang ?? "vi";
  }

  // Reserve job row — gives us a stable job_id for R2 namespacing
  const tmpKey = `raw/_init_${crypto.randomUUID()}.zip`;
  const job = await createJob(ctx.env.DB, {
    user_id:         userId,
    source_lang:     body.source_lang,
    target_lang:     targetLang,
    estimated_pages: estimatedPages,
    zip_key:         tmpKey,
    work_id:         body.work_id ?? null,
    kind,
  });
  const zipKey = `raw/${job.id}/source.zip`;
  await ctx.env.DB
    .prepare("UPDATE jobs SET zip_key = ? WHERE id = ?")
    .bind(zipKey, job.id)
    .run();

  // Hydrate input context from KV if work_id given
  let contextHydrated = false;
  if (body.work_id) {
    const inKey = await copyContextFromKvToR2(ctx.env, userId, body.work_id, job.id);
    if (inKey) {
      contextHydrated = true;
      await ctx.env.DB
        .prepare("UPDATE jobs SET context_in_key = ? WHERE id = ?")
        .bind(inKey, job.id)
        .run();
    }
  }

  // R2 multipart init + presigned PUTs
  const upload    = await ctx.env.R2.createMultipartUpload(zipKey);
  const partCount = Math.ceil(body.byte_size / MAX_PART_SIZE);
  const aws       = awsClient(ctx.env);

  const parts = await Promise.all(
    Array.from({ length: partCount }, (_, i) =>
      presignR2Url(aws, ctx.env, "PUT", zipKey, {
        partNumber: String(i + 1),
        uploadId:   upload.uploadId,
      }).then(url => ({ number: i + 1, url })),
    ),
  );

  await ctx.env.DB
    .prepare(`UPDATE jobs SET workflow_id = ?, state = 'uploading' WHERE id = ?`)
    .bind(`mpu:${upload.uploadId}`, job.id)
    .run();

  return ctx.json({
    job_id:            job.id,
    kind,
    parts,
    part_size:         MAX_PART_SIZE,
    expires_in:        UPLOAD_TTL,
    context_hydrated:  contextHydrated,
  });
});

// ── POST /jobs/:id/start — finalize multipart + spawn pipeline ──────

router.post("/jobs/:id/start", async (ctx) => {
  const userId = ctx.get("userId");
  const id     = Number(ctx.req.param("id"));
  const body   = await ctx.req.json<{
    parts: { number: number; etag: string }[];
  }>();

  if (!body.parts?.length) {
    return ctx.json({ error: "parts required" }, 400);
  }

  const job = await getJobForUser(ctx.env.DB, id, userId);
  if (!job) return ctx.json({ error: "Job not found" }, 404);
  if (job.state !== "uploading") {
    return ctx.json({ error: `Job state ${job.state}, expected uploading` }, 409);
  }
  if (!job.workflow_id?.startsWith("mpu:") || !job.zip_key) {
    return ctx.json({ error: "Job missing multipart handle" }, 500);
  }

  const uploadId = job.workflow_id.slice(4);
  const mpu = ctx.env.R2.resumeMultipartUpload(job.zip_key, uploadId);
  await mpu.complete(body.parts.map(p => ({ partNumber: p.number, etag: p.etag })));

  let pipeRes: Response;
  try {
    pipeRes = await ctx.env.PIPELINE.fetch("http://pipeline/start", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({
        job_id:         job.id,
        kind:           job.kind,
        zip_key:        job.zip_key,
        source_lang:    job.source_lang,
        target_lang:    job.target_lang,
        context_in_key: job.context_in_key ?? undefined,
      }),
    });
  } catch (err) {
    // Pipeline binding unreachable → roll back the committed zip so we
    // don't leak R2 storage. Job goes to 'error' for visibility.
    ctx.executionCtx.waitUntil(ctx.env.R2.delete(job.zip_key));
    await setJobState(ctx.env.DB, job.id, "error", {
      error_message: `Pipeline unreachable: ${String(err)}`,
    });
    return ctx.json({ error: "Pipeline unreachable", detail: String(err) }, 502);
  }

  if (!pipeRes.ok) {
    const detail = await pipeRes.text();
    ctx.executionCtx.waitUntil(ctx.env.R2.delete(job.zip_key));
    await setJobState(ctx.env.DB, job.id, "error", {
      error_message: `Pipeline rejected: ${detail}`,
    });
    return ctx.json({ error: "Pipeline rejected", detail }, 502);
  }

  const { id: workflowId } = await pipeRes.json<{ id: string }>();
  await setJobState(ctx.env.DB, job.id, "pending", {
    workflow_id: workflowId,
    started_at:  true,
  });

  const fresh = await getJobForUser(ctx.env.DB, id, userId);
  return ctx.json(toApiJob(fresh!));
});

// ── GET /jobs/:id ───────────────────────────────────────────────────

router.get("/jobs/:id", async (ctx) => {
  const userId = ctx.get("userId");
  const id     = Number(ctx.req.param("id"));
  const job    = await getJobForUser(ctx.env.DB, id, userId);
  if (!job) return ctx.json({ error: "Job not found" }, 404);

  // Merge error marker: when notifyError fails in the pipeline,
  // it writes an error marker to R2 as fallback.
  if (job.state === "running" || job.state === "pending") {
    const marker = await ctx.env.R2.get(`jobs/${id}/error.json`);
    if (marker) {
      const errData = await marker.json() as {
        error_message: string; stage: string; timestamp: string;
      };
      await setJobState(ctx.env.DB, id, "error", {
        error_message:  errData.error_message,
        progress_stage: errData.stage,
        finished_at:    true,
      });
      await ctx.env.R2.delete(`jobs/${id}/error.json`);
      // Re-fetch to return consistent snapshot
      const updated = await getJobForUser(ctx.env.DB, id, userId);
      if (updated) return ctx.json(await withSignedUrls(ctx.env, updated));
    }
  }

  return ctx.json(await withSignedUrls(ctx.env, job));
});

// ── GET /jobs/:id/download ──────────────────────────────────────────

router.get("/jobs/:id/download", async (ctx) => {
  const userId = ctx.get("userId");
  const id     = Number(ctx.req.param("id"));
  const job    = await getJobForUser(ctx.env.DB, id, userId);
  if (!job) return ctx.json({ error: "Job not found" }, 404);
  if (job.kind !== "translate") {
    return ctx.json({ error: "Job kind 'analyze' has no archive" }, 409);
  }
  if (job.state !== "done" || !job.archive_key) {
    return ctx.json({ error: `Archive not ready (state=${job.state})` }, 409);
  }
  const url = await presignR2Url(
    awsClient(ctx.env), ctx.env, "GET", job.archive_key, {}, DOWNLOAD_TTL,
  );
  return ctx.redirect(url, 302);
});

// ── DELETE /jobs/:id ────────────────────────────────────────────────

router.delete("/jobs/:id", async (ctx) => {
  const userId = ctx.get("userId");
  const id     = Number(ctx.req.param("id"));
  const job    = await getJobForUser(ctx.env.DB, id, userId);
  if (!job) return ctx.json({ error: "Job not found" }, 404);

  await cleanupJobR2(ctx.env, id);

  if (job.state === "uploading" && job.workflow_id?.startsWith("mpu:") && job.zip_key) {
    try {
      const mpu = ctx.env.R2.resumeMultipartUpload(job.zip_key, job.workflow_id.slice(4));
      await mpu.abort();
    } catch { /* idempotent */ }
  }

  await deleteJob(ctx.env.DB, id, userId);
  return ctx.body(null, 204);
});

// ── GET /me/jobs ────────────────────────────────────────────────────

router.get("/me/jobs", async (ctx) => {
  const userId = ctx.get("userId");
  const limit  = Math.min(100, Number(ctx.req.query("limit") ?? 50));
  const rows   = await listJobsForUser(ctx.env.DB, userId, limit);
  const out    = await Promise.all(rows.map(r => withSignedUrls(ctx.env, r)));
  return ctx.json(out);
});

// ── GET /me/quota ───────────────────────────────────────────────────

router.get("/me/quota", async (ctx) => {
  const userId = ctx.get("userId");
  const tier   = getTier(ctx.get("tierId"));
  const used   = await countChaptersThisMonth(ctx.env.DB, userId);
  const active = await countActiveJobs(ctx.env.DB, userId);
  const now    = new Date();
  const reset  = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth() + 1, 1));

  const out: ApiQuota = {
    tier: {
      id:                    tier.id,
      name:                  tier.name,
      monthly_chapters:      tier.monthly_chapters,
      max_pages_per_chapter: tier.max_pages_per_chapter,
      concurrent_jobs:       tier.concurrent_jobs,
      sync_quota_bytes:      tier.sync_quota_bytes,
      can_use_api_tokens:    tier.can_use_api_tokens,
    },
    used_chapters: used,
    active_jobs:   active,
    reset_at:      reset.toISOString(),
  };
  return ctx.json(out);
});

export default router;
