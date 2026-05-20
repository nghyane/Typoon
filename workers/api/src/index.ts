/**
 * Typoon-API Worker Entrypoint.
 *
 * Architecture:
 *   - Public routes:  GET /api/auth/*  (OAuth config, Discord exchange)
 *   - Protected routes: all others under /api require a valid JWT
 *   - Worker routes: /api/blobs require "worker" scope (checked in blobs.ts)
 */

import { Hono }         from "hono";
import { cors }         from "hono/cors";
import type { Env, ContextVars } from "./types";
import { requireUser }  from "./middleware/auth";
import { TranslationStatusDO } from "./do/translation-status";
import { PipelineCallback }     from "./rpc/pipeline-callback";
import { NotFoundError, ConflictError, ForbiddenError, RateLimitError } from "./store/db";

// Route imports
import authRouter       from "./routes/auth";
import materialsRouter  from "./routes/materials";
import translationsRouter from "./routes/translations";
import communityRouter  from "./routes/community";
import uploadRouter     from "./routes/upload";
import libraryRouter    from "./routes/library";
import workRouter       from "./routes/work";
import meRouter         from "./routes/me";
import glossaryRouter   from "./routes/glossary";
import memoryRouter     from "./routes/memory";
import { reportsRouter, adminRouter } from "./routes/admin";
import blobsRouter      from "./routes/blobs";

type AppEnv = { Bindings: Env; Variables: ContextVars };

const app = new Hono<AppEnv>();

// ── CORS ─────────────────────────────────────────────────────────────
//
// Discord Activities run inside an iframe at {clientId}.discordsays.com.
// All network traffic is proxied through Discord's CSP proxy, so the
// Origin header will be https://{clientId}.discordsays.com.
// We allow any *.discordsays.com origin — Discord's CSP already
// isolates each Activity to its own proxy domain.

const TRUSTED_ORIGINS = new Set([
  "https://typoon.app",
  "https://www.typoon.app",
]);

app.use("*", cors({
  origin: (origin) => {
    if (!origin) return "*";
    if (TRUSTED_ORIGINS.has(origin)) return origin;
    if (/^https?:\/\/localhost(:\d+)?$/.test(origin)) return origin;
    if (/^https:\/\/[a-z0-9-]+\.discordsays\.com$/.test(origin)) return origin;
    return null;
  },
  allowMethods: ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
  allowHeaders: ["Content-Type", "Authorization"],
  exposeHeaders: ["Content-Length"],
  maxAge:       86400,
  credentials:  true,
}));

// ── API sub-app ───────────────────────────────────────────────────────

const apiApp = new Hono<AppEnv>();

// Public: Discord OAuth — no auth required
apiApp.route("/auth", authRouter);

// Protected: apply auth to everything below this point
apiApp.use("/*", requireUser());

// Core resources
apiApp.route("/materials",   materialsRouter);
apiApp.route("/translations", translationsRouter);
apiApp.route("/community",   communityRouter);
apiApp.route("/uploads",     uploadRouter);
apiApp.route("/library",     libraryRouter);
apiApp.route("/work",        workRouter);

// User features
apiApp.route("/me",          meRouter);
apiApp.route("/glossary",    glossaryRouter);
apiApp.route("/memory",      memoryRouter);

// Admin + reports
apiApp.route("/reports",     reportsRouter);
apiApp.route("/admin",       adminRouter);

// Worker-facing (requires "worker" scope, checked internally)
apiApp.route("/blobs",       blobsRouter);

app.route("/api", apiApp);

// ── Global error handler (Hono native) ──────────────────────────────

app.onError((err, ctx) => {
  if (err instanceof NotFoundError) {
    return ctx.json(
      { error: { code: "not_found", message: err.message } },
      404,
    );
  }
  if (err instanceof ForbiddenError) {
    return ctx.json(
      { error: { code: "forbidden", message: err.message } },
      403,
    );
  }
  if (err instanceof ConflictError) {
    return ctx.json(
      { error: { code: "conflict", message: err.message } },
      409,
    );
  }
  if (err instanceof RateLimitError) {
    const res = ctx.json(
      { error: { code: "rate_limited", message: err.message } },
      429,
    );
    res.headers.set("Retry-After", String(err.retryAfter));
    return res;
  }

  // D1 constraint violations → 409
  if ((err as any)?.code === "SQLITE_CONSTRAINT") {
    return ctx.json(
      { error: { code: "conflict", message: (err as any).message } },
      409,
    );
  }

  // Hono HTTPException passthrough
  if ((err as any)?.status && (err as any)?.message) {
    return ctx.json(
      { error: { code: "http_error", message: (err as any).message } },
      (err as any).status,
    );
  }

  // Fallback — 500, no stack trace to client
  console.error("[unhandled]", err);
  return ctx.json(
    { error: { code: "internal_error", message: "Internal server error" } },
    500,
  );
});

// ── Exports ───────────────────────────────────────────────────────────

export { TranslationStatusDO, PipelineCallback };
export default app;
