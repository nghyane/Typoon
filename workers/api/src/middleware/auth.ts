/**
 * JWT authentication middleware for Hono.
 *
 * Supports two schemes:
 *   Bearer <jwt>      — Discord OAuth JWT issued by POST /auth/discord/exchange
 *   Bearer <token>    — API token (prefix lookup → bcrypt verify)
 *
 * The middleware sets ctx.var.userId and ctx.var.jwtRoles.
 * Use `requireAdmin` guard for admin-only routes.
 */

import type { Context, MiddlewareHandler, Next } from "hono";
import type { Env, JwtPayload, ContextVars }      from "../types";

const JWT_ALG   = { name: "HMAC", hash: "SHA-256" } as const;
const ISSUER    = "typoon";
const TOKEN_TTL = 60 * 60 * 24 * 7; // 7 days

// ── JWT issue (used by auth route) ──────────────────────────────────

export async function issueJwt(
  userId:  number,
  roles:   string[],
  tier_id: string,
  secret:  string,
): Promise<string> {
  const now     = Math.floor(Date.now() / 1000);
  const payload: JwtPayload = {
    sub:     String(userId),
    iss:     ISSUER,
    iat:     now,
    exp:     now + TOKEN_TTL,
    roles,
    tier_id,
  };

  const key = await importKey(secret, ["sign"]);
  const [header, body] = [
    btoa(JSON.stringify({ alg: "HS256", typ: "JWT" })),
    btoa(JSON.stringify(payload)),
  ].map(b64url);

  const sig = await crypto.subtle.sign(
    JWT_ALG,
    key,
    new TextEncoder().encode(`${header}.${body}`),
  );

  return `${header}.${body}.${b64url(btoa(String.fromCharCode(...new Uint8Array(sig))))}`;
}

// ── JWT verify ───────────────────────────────────────────────────────

export async function verifyJwt(token: string, secret: string): Promise<JwtPayload | null> {
  const parts = token.split(".");
  if (parts.length !== 3) return null;

  const [header, body, sig] = parts as [string, string, string];

  try {
    const key  = await importKey(secret, ["verify"]);
    const valid = await crypto.subtle.verify(
      JWT_ALG,
      key,
      b64decode(sig),
      new TextEncoder().encode(`${header}.${body}`),
    );
    if (!valid) return null;

    const payload = JSON.parse(atob(fromB64url(body))) as JwtPayload;
    const now     = Math.floor(Date.now() / 1000);

    if (payload.iss !== ISSUER)   return null;
    if (payload.exp < now)        return null;

    return payload;
  } catch {
    return null;
  }
}

// ── Hono middleware: require authenticated user ──────────────────────

export function requireUser(): MiddlewareHandler<{ Bindings: Env; Variables: ContextVars }> {
  return async (ctx, next) => {
    const token = extractBearerToken(ctx);
    if (!token) return ctx.json({ error: "Unauthorized" }, 401);

    // Try JWT first
    const payload = await verifyJwt(token, ctx.env.JWT_SECRET);
    if (payload) {
      ctx.set("userId",   Number(payload.sub));
      ctx.set("jwtRoles", payload.roles);
      ctx.set("tierId",   payload.tier_id ?? "free");
      return next();
    }

    // Try API token (prefix is first 8 chars)
    const prefix = token.slice(0, 8);
    const row    = await ctx.env.DB
      .prepare(`SELECT id, user_id, token_hash, scopes FROM api_tokens
                WHERE prefix = ? AND revoked_at IS NULL`)
      .bind(prefix)
      .first<{ id: number; user_id: number; token_hash: string; scopes: string }>();

    if (!row) return ctx.json({ error: "Unauthorized" }, 401);

    const match = await verifyApiToken(token, row.token_hash);
    if (!match)  return ctx.json({ error: "Unauthorized" }, 401);

    // Touch last_used async (fire and forget — don't block the request)
    ctx.executionCtx.waitUntil(
      ctx.env.DB.prepare("UPDATE api_tokens SET last_used = datetime('now') WHERE id = ?")
        .bind(row.id).run(),
    );

    // API tokens carry their owner's current tier from D1 (no JWT re-issue).
    const userRow = await ctx.env.DB
      .prepare(`SELECT tier_id FROM users WHERE id = ?`)
      .bind(row.user_id)
      .first<{ tier_id: string }>();

    ctx.set("userId",   row.user_id);
    ctx.set("jwtRoles", JSON.parse(row.scopes) as string[]);
    ctx.set("tierId",   userRow?.tier_id ?? "free");
    return next();
  };
}

// ── Guard: admin only ────────────────────────────────────────────────

export function requireAdmin(): MiddlewareHandler<{ Bindings: Env; Variables: ContextVars }> {
  return async (ctx, next) => {
    const roles     = ctx.get("jwtRoles") ?? [];
    const adminRole = ctx.env.ADMIN_ROLE_ID;
    if (!adminRole || !roles.includes(adminRole)) {
      return ctx.json({ error: "Forbidden" }, 403);
    }
    return next();
  };
}

// ── Helpers ──────────────────────────────────────────────────────────

function extractBearerToken(ctx: Context): string | null {
  const auth = ctx.req.header("Authorization") ?? "";
  if (auth.startsWith("Bearer ")) return auth.slice(7);
  // Also accept ?token= query param (for EventSource / WebSocket)
  return ctx.req.query("token") ?? null;
}

async function importKey(
  secret: string,
  usages: Array<"sign" | "verify">,
): Promise<CryptoKey> {
  return crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(secret),
    JWT_ALG,
    false,
    usages,
  );
}

// ── API token hashing (shared with me.ts for token creation) ────────

/** SHA-256 digest of the plaintext token stored as hex. */
export async function hashApiToken(plain: string): Promise<string> {
  const buf = await crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(plain),
  );
  return Array.from(new Uint8Array(buf))
    .map(b => b.toString(16).padStart(2, "0")).join("");
}

async function verifyApiToken(plain: string, hash: string): Promise<boolean> {
  const computed = await hashApiToken(plain);
  return computed === hash;
}

function b64url(b64: string): string {
  return b64.replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function fromB64url(s: string): string {
  return s.replace(/-/g, "+").replace(/_/g, "/");
}

function b64decode(s: string): Uint8Array {
  const bin = atob(fromB64url(s));
  return Uint8Array.from(bin, c => c.charCodeAt(0));
}
