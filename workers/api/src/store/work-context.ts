/**
 * Work-context KV store.
 *
 * KV layout
 *   key:      ctx:{user_id}:{work_id}
 *   value:    gzip(JSON.stringify(WorkContext))   — server treats as opaque bytes
 *   metadata: { version: number; updated_at: string }
 *   TTL:      180 days (refreshed on every write/read)
 *
 * Server never parses the JSON — pipeline brief stage reads it from R2,
 * uses it as seed, and writes a fresh blob back. This module just shuttles
 * bytes between KV and R2.
 */

import type { Env } from "../types";

const CTX_TTL_SECONDS = 180 * 86_400;   // 180 days

export interface ContextMetadata {
  version:    number;
  updated_at: string;
}

export function ctxKey(user_id: number, work_id: string): string {
  return `ctx:${user_id}:${work_id}`;
}

/** Pull current context bytes from KV into a fresh R2 key tied to this job.
 *  Returns the R2 key on hit, null on miss. */
export async function copyContextFromKvToR2(
  env:     Env,
  user_id: number,
  work_id: string,
  job_id:  number,
): Promise<string | null> {
  const stream = await env.WORK_CONTEXTS.get(ctxKey(user_id, work_id), "stream");
  if (!stream) return null;

  const r2Key = `ctx/${job_id}/input.json.gz`;
  await env.R2.put(r2Key, stream, {
    httpMetadata: {
      contentType:     "application/json",
      contentEncoding: "gzip",
    },
  });
  return r2Key;
}

/** Copy the brief-stage output blob from R2 → KV with bumped version. */
export async function copyContextFromR2ToKv(
  env:     Env,
  user_id: number,
  work_id: string,
  r2Key:   string,
): Promise<number> {
  const obj = await env.R2.get(r2Key);
  if (!obj) throw new Error(`R2 object missing: ${r2Key}`);
  const bytes = await obj.arrayBuffer();

  const existing = await env.WORK_CONTEXTS.getWithMetadata<ContextMetadata>(
    ctxKey(user_id, work_id),
  );
  const nextVersion = (existing.metadata?.version ?? 0) + 1;

  await env.WORK_CONTEXTS.put(ctxKey(user_id, work_id), bytes, {
    metadata:      { version: nextVersion, updated_at: new Date().toISOString() },
    expirationTtl: CTX_TTL_SECONDS,
  });
  return nextVersion;
}

export async function getContextWithMeta(
  env:     Env,
  user_id: number,
  work_id: string,
): Promise<{ stream: ReadableStream | null; metadata: ContextMetadata | null }> {
  const result = await env.WORK_CONTEXTS.getWithMetadata<ContextMetadata>(
    ctxKey(user_id, work_id), "stream",
  );
  return {
    stream:   result.value,
    metadata: result.metadata ?? null,
  };
}

export async function putContext(
  env:        Env,
  user_id:    number,
  work_id:    string,
  bytes:      ArrayBuffer | ReadableStream,
  base_version: number | null,
): Promise<{ version: number } | { conflict: number }> {
  const existing = await env.WORK_CONTEXTS.getWithMetadata<ContextMetadata>(
    ctxKey(user_id, work_id),
  );
  const cur = existing.metadata?.version ?? 0;

  if (base_version !== null && base_version !== cur) {
    return { conflict: cur };
  }

  const nextVersion = cur + 1;
  await env.WORK_CONTEXTS.put(ctxKey(user_id, work_id), bytes, {
    metadata:      { version: nextVersion, updated_at: new Date().toISOString() },
    expirationTtl: CTX_TTL_SECONDS,
  });
  return { version: nextVersion };
}

export async function deleteContext(
  env: Env, user_id: number, work_id: string,
): Promise<void> {
  await env.WORK_CONTEXTS.delete(ctxKey(user_id, work_id));
}
