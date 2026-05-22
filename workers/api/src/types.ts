/**
 * Env — Cloudflare bindings for typoon-api Worker.
 *
 * Keep in sync with wrangler.toml + .dev.vars + worker-configuration.d.ts.
 */

import type { UserEventsDO } from "./do/user-events";
import type { PipelineCallback } from "./rpc/pipeline-callback";

export interface Env {
  // D1 — primary datastore
  DB: D1Database;

  // R2 — blob storage (raw uploads + pipeline intermediate + rendered archives)
  R2: R2Bucket;

  // KV — persistent per-(user × work) translation context.
  // Key: `ctx:{user_id}:{work_id}`  Value: gzip(JSON.stringify(WorkContext))
  // Metadata: { version: number, updated_at: string }  TTL: 180 days (refresh on access).
  WORK_CONTEXTS: KVNamespace;

  // Durable Objects — one DO per user, multiplexes job events.
  USER_EVENTS_DO: DurableObjectNamespace<UserEventsDO>;

  // Service binding — invoke ChapterPipeline workflow
  PIPELINE: Fetcher;

  // Secrets
  JWT_SECRET:            string;
  DISCORD_CLIENT_ID:     string;
  DISCORD_CLIENT_SECRET: string;
  DISCORD_GUILD_ID?:     string;
  ADMIN_ROLE_ID?:        string;
  /** JSON: `{"<discord_role_id>": "<tier_id>"}` — maps Discord roles to tiers. */
  DISCORD_ROLE_TIER_MAP?: string;
  // R2 S3-compatible credentials (presigned upload URLs)
  R2_ACCESS_KEY_ID:      string;
  R2_SECRET_ACCESS_KEY:  string;
  R2_ACCOUNT_ID:         string;
  R2_BUCKET_NAME:        string;

  // Vars
  DISCORD_API: string;
}

// ── Domain literals ─────────────────────────────────────────────────

export type JobState =
  | "init"       // created, R2 multipart open, awaiting parts
  | "uploading"  // client uploading parts
  | "pending"    // /start called, workflow scheduled
  | "running"    // workflow active
  | "done"       // archive ready
  | "error"      // failed
  | "expired";   // R2 blobs gone (7d cleanup)

export type PipelineStage = "prepare" | "scan" | "brief" | "translate" | "typeset" | "finalize";

export type ReportStatus = "open" | "reviewing" | "resolved" | "dismissed";

// ── JWT payload ─────────────────────────────────────────────────────

export interface JwtPayload {
  sub:      string;    // user_id as string (RFC 7519)
  iss:      string;
  iat:      number;
  exp:      number;
  /** Discord guild role IDs at time of issuance. Used for is_admin gating. */
  roles:    string[];
  /** Tier ID at time of issuance. Auth middleware passes this through; tier
   *  config is looked up from getTier() so quota changes ship without a JWT
   *  re-issue. */
  tier_id:  string;
}

// ── Hono context variables ──────────────────────────────────────────

export interface ContextVars {
  userId:   number;
  jwtRoles: string[];
  /** Tier ID from JWT claims; resolve via getTier() in route handlers. */
  tierId:   string;
}

// ── Wire types — server → client (SessionUser) ──────────────────────

export interface ApiTierInfo {
  id:                    string;
  name:                  string;
  monthly_chapters:      number;
  max_pages_per_chapter: number;
  concurrent_jobs:       number;
  sync_quota_bytes:      number;
  can_use_api_tokens:    boolean;
}

export interface ApiSessionUser {
  id:                    number;
  display_name:          string;
  avatar_url:            string | null;
  email:                 string | null;
  is_admin:              boolean;
  preferred_target_lang: string | null;
  tier:                  ApiTierInfo;
}

// ── Wire types — jobs ───────────────────────────────────────────────

export interface ApiJobInitPart {
  number: number;
  url:    string;
}

export interface ApiJobInit {
  job_id:     number;
  parts:      ApiJobInitPart[];
  part_size:  number;
  expires_in: number;          // presigned URL TTL (seconds)
}

export interface ApiJob {
  id:              number;
  state:           JobState;
  kind:            "translate" | "analyze";
  work_id:         string | null;
  source_lang:     string;
  target_lang:     string;
  progress_stage:  string | null;
  progress_index:  number | null;
  progress_total:  number | null;
  page_count:      number | null;
  estimated_pages: number | null;
  /** Presigned GET URL for the output archive; null when state != 'done'
   *  or when kind = 'analyze' (no archive produced). */
  archive_url:     string | null;
  /** Presigned GET URL for merged WorkContext (gzip+JSON); null when state != 'done'. */
  context_out_url: string | null;
  /** Current KV version after pipeline merge — useful for client cache invalidation. */
  context_version: number | null;
  error_message:   string | null;
  created_at:      string;
  started_at:      string | null;
  finished_at:     string | null;
  expires_at:      string;
}

// ── Wire types — quota ──────────────────────────────────────────────

export interface ApiQuota {
  tier:           ApiTierInfo;
  used_chapters:  number;
  active_jobs:    number;
  reset_at:       string;       // ISO datetime, 1st of next month UTC
}

// ── Pipeline callback args ──────────────────────────────────────────

export interface FinalizeArgs {
  job_id:           number;
  /** Absent when kind='analyze'. */
  archive_key?:     string;
  page_count:       number;
  context_out_key?: string;
}
