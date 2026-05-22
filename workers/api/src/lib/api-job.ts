/**
 * Shared helpers for converting D1 `jobs` rows into the wire-shape `ApiJob`.
 *
 * Two consumers:
 *   - `routes/jobs.ts`        — REST handlers (GET /jobs/:id, GET /me/jobs).
 *   - `do/user-events.ts`     — DO that pushes per-user job updates over WS
 *                                and snapshots the current state on connect.
 *   - `rpc/pipeline-callback.ts` — RPC handlers that broadcast a fresh
 *                                  `ApiJob` after each state mutation.
 *
 * Keeping the conversion in one place means the wire shape (especially
 * presigned URLs + context version) stays consistent across pull, push,
 * and replay.
 */

import { AwsClient } from "aws4fetch";

import type { Env, ApiJob } from "../types";
import type { JobRow } from "../store/jobs";
import { getContextWithMeta } from "../store/work-context";

/** Presigned download URL TTL (1 h). The client refetches its job before
 *  this expires, so a stale URL never reaches the user. */
export const DOWNLOAD_TTL = 3_600;

export function awsClient(env: Env): AwsClient {
  return new AwsClient({
    accessKeyId:     env.R2_ACCESS_KEY_ID,
    secretAccessKey: env.R2_SECRET_ACCESS_KEY,
    region:          "auto",
    service:         "s3",
  });
}

export async function presignR2Url(
  aws:    AwsClient,
  env:    Env,
  method: "PUT" | "GET",
  key:    string,
  query:  Record<string, string> = {},
  ttl:    number = 3_600,
): Promise<string> {
  const url = new URL(
    `https://${env.R2_ACCOUNT_ID}.r2.cloudflarestorage.com/${env.R2_BUCKET_NAME}/${key}`,
  );
  for (const [k, v] of Object.entries(query)) url.searchParams.set(k, v);
  url.searchParams.set("X-Amz-Expires", String(ttl));
  const signed = await aws.sign(new Request(url, { method }), {
    aws: { signQuery: true },
  });
  return signed.url;
}

export function toApiJob(row: JobRow): ApiJob {
  return {
    id:              row.id,
    state:           row.state,
    kind:            row.kind,
    work_id:         row.work_id,
    source_lang:     row.source_lang,
    target_lang:     row.target_lang,
    progress_stage:  row.progress_stage,
    progress_index:  row.progress_index,
    progress_total:  row.progress_total,
    page_count:      row.page_count,
    estimated_pages: row.estimated_pages,
    archive_url:     null,
    context_out_url: null,
    context_version: null,
    error_message:   row.error_message,
    created_at:      row.created_at,
    started_at:      row.started_at,
    finished_at:     row.finished_at,
    expires_at:      row.expires_at,
  };
}

/** Hydrate presigned download URLs + KV context version for terminal jobs.
 *  No-op for non-`done` rows; safe to call on every push. */
export async function withSignedUrls(env: Env, job: JobRow): Promise<ApiJob> {
  const out = toApiJob(job);
  if (job.state !== "done") return out;

  const aws = awsClient(env);
  if (job.kind === "translate" && job.archive_key) {
    out.archive_url = await presignR2Url(aws, env, "GET", job.archive_key, {}, DOWNLOAD_TTL);
  }
  if (job.context_out_key) {
    out.context_out_url = await presignR2Url(
      aws, env, "GET", job.context_out_key, {}, DOWNLOAD_TTL,
    );
  }
  if (job.work_id) {
    const meta = await getContextWithMeta(env, job.user_id, job.work_id);
    out.context_version = meta.metadata?.version ?? null;
  }
  return out;
}
