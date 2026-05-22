/**
 * typoon-api — Worker entrypoint.
 *
 *   Public routes:
 *     /api/auth/config
 *     /api/auth/discord/exchange
 *     /api/cdn/c/<host><path>   ← CORS proxy for source adapters
 *
 *   Protected routes (require JWT or API token):
 *     /api/auth/me
 *     /api/auth/me/preferences
 *     /api/auth/refresh
 *     /api/jobs/*
 *     /api/me/*
 *     /api/me/events           ← WebSocket: per-user multiplexed job stream
 *     /api/sync
 *     /api/reports
 */

import { Hono } from "hono";
import { cors } from "hono/cors";
import type { Env, ContextVars } from "./types";
import { requireUser } from "./middleware/auth";
import { UserEventsDO } from "./do/user-events";
import { PipelineCallback } from "./rpc/pipeline-callback";
import { NotFoundError, ConflictError, ForbiddenError, RateLimitError } from "./store/db";

import authRouter    from "./routes/auth";
import jobsRouter    from "./routes/jobs";
import worksRouter   from "./routes/works";
import reportsRouter from "./routes/reports";

type AppEnv = { Bindings: Env; Variables: ContextVars };

const app = new Hono<AppEnv>();

// ── CORS ────────────────────────────────────────────────────────────

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
  allowMethods:  ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
  allowHeaders:  ["Content-Type", "Authorization", "X-Proxy-Headers"],
  exposeHeaders: ["Content-Length", "ETag"],
  maxAge:        86400,
  credentials:   true,
}));

// ── /api/* sub-app ──────────────────────────────────────────────────

const apiApp = new Hono<AppEnv>();

// Public auth routes (no JWT required)
apiApp.route("/auth", authRouter);

// Per-user WebSocket — protected via query token because browsers can't
// send Authorization headers on WebSocket upgrades. requireUser() reads
// ?token= as fallback. A single connection per device multiplexes every
// job event for this user (multiple concurrent translations supported).
apiApp.get("/me/events", requireUser(), async (ctx) => {
  if (ctx.req.header("Upgrade") !== "websocket") {
    return ctx.json({ error: "Expected WebSocket upgrade" }, 426);
  }
  const userId = ctx.get("userId");
  const doId   = ctx.env.USER_EVENTS_DO.idFromName(String(userId));
  const stub   = ctx.env.USER_EVENTS_DO.get(doId);
  // DO needs user_id to build the snapshot; pass it explicitly because
  // the DO can't re-run the JWT middleware. idFromName already binds the
  // DO to this user — the uid in the URL is just a fast accessor.
  const url = new URL(ctx.req.url);
  url.searchParams.set("uid", String(userId));
  return stub.fetch(new Request(url, ctx.req.raw));
});

// Protected routes
apiApp.use("/jobs/*",  requireUser());
apiApp.use("/me/*",    requireUser());
apiApp.use("/works/*", requireUser());

apiApp.route("/",        jobsRouter);     // /jobs, /jobs/:id, /me/jobs, /me/quota
apiApp.route("/works",   worksRouter);
apiApp.route("/reports", reportsRouter);

app.route("/api", apiApp);

// ── /cdn/c/<host><path> — CORS proxy ────────────────────────────────
//
// Forwards X-Proxy-Headers (base64url(JSON)) or ?_h= query blob as
// upstream headers verbatim. Denies hop-by-hop and CF-* headers.

const DENYLIST = /^(host|connection|cf-|x-forwarded-|x-proxy-headers)$/i;

app.all("/cdn/c/*", async (ctx) => {
  const url    = new URL(ctx.req.url);
  const after  = url.pathname.slice("/cdn/c/".length);
  const slash  = after.indexOf("/");
  const host   = slash === -1 ? after : after.slice(0, slash);
  const rest   = slash === -1 ? "/"   : after.slice(slash);
  if (!host) return ctx.json({ error: "Missing upstream host" }, 400);

  const params     = new URLSearchParams(url.searchParams);
  const headerBlob = params.get("_h");
  params.delete("_h");
  const qs       = params.toString();
  const upstream = `https://${host}${rest}${qs ? "?" + qs : ""}`;

  const xph = ctx.req.header("X-Proxy-Headers") ?? headerBlob ?? "";
  let extra: Record<string, string> = {};
  if (xph) {
    try {
      const json = atob(xph.replace(/-/g, "+").replace(/_/g, "/"));
      extra = JSON.parse(json);
    } catch { /* malformed — ignore */ }
  }

  const fwd = new Headers();
  for (const [k, v] of Object.entries(ctx.req.raw.headers)) {
    if (!DENYLIST.test(k)) fwd.set(k, v as string);
  }
  for (const [k, v] of Object.entries(extra)) {
    if (!DENYLIST.test(k)) fwd.set(k, v);
  }
  fwd.set("Host", host);

  const res = await fetch(upstream, {
    method:   ctx.req.method,
    headers:  fwd,
    body:     ["GET", "HEAD"].includes(ctx.req.method) ? undefined : ctx.req.raw.body,
    redirect: "follow",
  });

  const resHeaders = new Headers(res.headers);
  resHeaders.set("Access-Control-Allow-Origin", "*");
  resHeaders.delete("Content-Encoding");

  return new Response(res.body, { status: res.status, headers: resHeaders });
});

// ── Global error handler ────────────────────────────────────────────

app.onError((err, ctx) => {
  if (err instanceof NotFoundError)    return ctx.json({ error: { code: "not_found",    message: err.message } }, 404);
  if (err instanceof ForbiddenError)   return ctx.json({ error: { code: "forbidden",    message: err.message } }, 403);
  if (err instanceof ConflictError)    return ctx.json({ error: { code: "conflict",     message: err.message } }, 409);
  if (err instanceof RateLimitError) {
    const res = ctx.json({ error: { code: "rate_limited", message: err.message } }, 429);
    res.headers.set("Retry-After", String(err.retryAfter));
    return res;
  }
  if ((err as any)?.code === "SQLITE_CONSTRAINT") {
    return ctx.json({ error: { code: "conflict", message: (err as any).message } }, 409);
  }
  if ((err as any)?.status && (err as any)?.message) {
    return ctx.json({ error: { code: "http_error", message: (err as any).message } }, (err as any).status);
  }

  console.error("[unhandled]", err);
  return ctx.json({ error: { code: "internal_error", message: "Internal server error" } }, 500);
});

// ── Exports ─────────────────────────────────────────────────────────

export { UserEventsDO, PipelineCallback };
export default app;
