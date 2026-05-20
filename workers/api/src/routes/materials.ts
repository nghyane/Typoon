/**
 * Materials routes — import, edit metadata, delete, cover upload.
 *
 * POST   /api/materials/import          → import from source
 * PATCH  /api/materials/:id             → edit metadata
 * DELETE /api/materials/:id             → delete (upload/extension owner only)
 * POST   /api/materials/:id/cover       → upload cover image
 */

import { Hono } from "hono";
import type { Env, ContextVars, ApiMaterial } from "../types";
import {
  getMaterial,
  importMaterial,
  updateMaterial,
  deleteMaterial,
  type MaterialRow,
} from "../store/materials";

type AppEnv = { Bindings: Env; Variables: ContextVars };
const router = new Hono<AppEnv>();

// ── Helpers ─────────────────────────────────────────────────────────

/** Check if user can mutate a material (importer or admin). */
function canMutateMaterial(
  userId: number,
  jwtRoles: string[] | undefined,
  material: { origin: string; imported_by: number | null },
  adminRoleId: string | undefined,
): boolean {
  if (material.origin !== "source") return true;
  if (material.imported_by === userId) return true;
  return (jwtRoles ?? []).includes(adminRoleId ?? "");
}

// ── Serializer ──────────────────────────────────────────────────────

function toApiMaterial(row: MaterialRow): ApiMaterial {
  return {
    id:            row.id,
    origin:        row.origin,
    work_id:       row.work_id,
    source:        row.source,
    upstream_ref:  row.upstream_ref,
    title:         row.title,
    cover_url:     row.cover_url,
    description:   row.description,
    author:        row.author,
    status:        row.status,
    languages:     row.languages ? JSON.parse(row.languages) : [],
    title_native:  row.title_native,
    title_alt:     row.title_alt ? JSON.parse(row.title_alt) : [],
    cross_refs:    row.cross_refs ? JSON.parse(row.cross_refs) : null,
    title_locale:  row.title_locale ? JSON.parse(row.title_locale) : null,
    start_year:    row.start_year,
    nsfw:          row.nsfw === 1,
    imported_by:   row.imported_by,
    created_at:    row.created_at,
    updated_at:    row.updated_at,
  };
}

// ── POST /import ────────────────────────────────────────────────────

router.post("/import", async (ctx) => {
  const userId = ctx.get("userId");
  const body = await ctx.req.json<{
    work_id:       number;
    source:        string;
    upstream_ref:  string;
    title:         string;
    cover_url?:    string | null;
    description?:  string | null;
    author?:       string | null;
    status?:       string | null;
    languages?:    string[];
    title_native?: string | null;
    title_alt?:    string[];
    cross_refs?:   Record<string, unknown>;
    nsfw?:         boolean;
  }>();

  if (!body.work_id || !body.source || !body.upstream_ref || !body.title) {
    return ctx.json(
      { error: { code: "bad_request", message: "work_id, source, upstream_ref, title required" } },
      400,
    );
  }

  const row = await importMaterial(ctx.env.DB, {
    imported_by:  userId,
    origin:       "source",
    work_id:      body.work_id,
    source:       body.source,
    upstream_ref: body.upstream_ref,
    title:        body.title,
    cover_url:    body.cover_url,
    description:  body.description,
    author:       body.author,
    status:       body.status,
    languages:    body.languages,
    title_native: body.title_native,
    title_alt:    body.title_alt,
    cross_refs:   body.cross_refs,
    nsfw:         body.nsfw,
  });

  return ctx.json(toApiMaterial(row), 201);
});

// ── PATCH /:id ──────────────────────────────────────────────────────

router.patch("/:id", async (ctx) => {
  const userId = ctx.get("userId");
  const id = Number(ctx.req.param("id"));
  if (!id || isNaN(id)) {
    return ctx.json(
      { error: { code: "bad_request", message: "Invalid material ID" } },
      400,
    );
  }

  const existing = await getMaterial(ctx.env.DB, id);
  if (!existing) {
    return ctx.json(
      { error: { code: "not_found", message: "Material not found" } },
      404,
    );
  }

  // Source-backed: only admin or importer can patch
  if (!canMutateMaterial(userId, ctx.get("jwtRoles"), existing, ctx.env.ADMIN_ROLE_ID)) {
    return ctx.json(
      { error: { code: "forbidden", message: "Only the importer can edit source-backed materials" } },
      403,
    );
  }

  const body = await ctx.req.json<{
    title?:        string;
    cover_url?:    string | null;
    description?:  string | null;
    author?:       string | null;
    status?:       string | null;
    title_native?: string | null;
    title_alt?:    string[];
    cross_refs?:   Record<string, unknown> | null;
    title_locale?: Record<string, string> | null;
    start_year?:   number | null;
    nsfw?:         boolean;
  }>();

  const row = await updateMaterial(ctx.env.DB, id, body);
  return ctx.json(toApiMaterial(row));
});

// ── DELETE /:id ─────────────────────────────────────────────────────

router.delete("/:id", async (ctx) => {
  const userId = ctx.get("userId");
  const id = Number(ctx.req.param("id"));
  if (!id || isNaN(id)) {
    return ctx.json(
      { error: { code: "bad_request", message: "Invalid material ID" } },
      400,
    );
  }

  const ok = await deleteMaterial(ctx.env.DB, id, userId);
  if (!ok) {
    return ctx.json(
      { error: { code: "not_found", message: "Material not found" } },
      404,
    );
  }

  return ctx.body(null, 204);
});

// ── POST /:id/cover ─────────────────────────────────────────────────

const ALLOWED_COVER_MIMES = new Set(["image/jpeg", "image/png", "image/webp"]);
const MAX_COVER_SIZE = 2 * 1024 * 1024; // 2 MiB

router.post("/:id/cover", async (ctx) => {
  const userId = ctx.get("userId");
  const id = Number(ctx.req.param("id"));
  if (!id || isNaN(id)) {
    return ctx.json(
      { error: { code: "bad_request", message: "Invalid material ID" } },
      400,
    );
  }

  const existing = await getMaterial(ctx.env.DB, id);
  if (!existing) {
    return ctx.json(
      { error: { code: "not_found", message: "Material not found" } },
      404,
    );
  }

  // Ownership check
  if (!canMutateMaterial(userId, ctx.get("jwtRoles"), existing, ctx.env.ADMIN_ROLE_ID)) {
    return ctx.json(
      { error: { code: "forbidden", message: "Only the importer can upload covers for source-backed materials" } },
      403,
    );
  }

  const body = await ctx.req.parseBody();
  const file = body["file"];
  if (!file || typeof file === "string") {
    return ctx.json(
      { error: { code: "bad_request", message: "file field required" } },
      400,
    );
  }

  if (!ALLOWED_COVER_MIMES.has(file.type)) {
    return ctx.json(
      { error: { code: "bad_request", message: "Only JPEG, PNG, WebP allowed" } },
      400,
    );
  }

  if (file.size > MAX_COVER_SIZE) {
    return ctx.json(
      { error: { code: "bad_request", message: "Cover must be under 2 MiB" } },
      400,
    );
  }

  // Upload to R2
  const ext = file.type.split("/")[1] ?? "bin";
  const key = `covers/${id}/${crypto.randomUUID()}.${ext}`;
  const bytes = await file.arrayBuffer();
  await ctx.env.R2.put(key, bytes, {
    httpMetadata: { contentType: file.type },
  });

  // Update material cover_url
  const coverUrl = `/r2/${key}`;
  const row = await updateMaterial(ctx.env.DB, id, { cover_url: coverUrl });

  return ctx.json(toApiMaterial(row));
});

export default router;
