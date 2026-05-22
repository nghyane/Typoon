/**
 * Auth routes — Discord OAuth + JWT issuance.
 *
 *   GET   /auth/config           public OAuth config for SPA
 *   POST  /auth/discord/exchange exchange code → JWT + SessionUser
 *   POST  /auth/refresh          re-sync tier from Discord role → reissue JWT
 *   GET   /auth/me               current session (protected)
 *   PATCH /auth/me/preferences   update user preferences (protected)
 */

import { Hono } from "hono";
import { issueJwt, requireUser } from "../middleware/auth";
import { getTier, resolveTierFromRoles, TIERS, type TierConfig } from "../lib/tiers";
import { getUser, updateUserPreferences, upsertDiscordUser } from "../store/users";
import type { Env, ContextVars, ApiSessionUser } from "../types";

const router = new Hono<{ Bindings: Env; Variables: ContextVars }>();

// ── GET /auth/config ────────────────────────────────────────────────

router.get("/config", (ctx) => {
  if (!ctx.env.DISCORD_CLIENT_ID) {
    return ctx.json({ error: "Discord OAuth not configured" }, 503);
  }
  return ctx.json({ discord_client_id: ctx.env.DISCORD_CLIENT_ID });
});

// ── Helpers ─────────────────────────────────────────────────────────

function parseRoleTierMap(env: Env): Record<string, string> {
  if (!env.DISCORD_ROLE_TIER_MAP) return {};
  try { return JSON.parse(env.DISCORD_ROLE_TIER_MAP); }
  catch { return {}; }
}

/** Fetch Discord guild member roles. Returns [] on any failure
 *  (user not in guild, network error, etc) so tier falls back to free. */
async function fetchGuildRoles(
  env: Env, accessToken: string,
): Promise<string[]> {
  if (!env.DISCORD_GUILD_ID) return [];
  try {
    const res = await fetch(
      `${env.DISCORD_API}/users/@me/guilds/${env.DISCORD_GUILD_ID}/member`,
      { headers: { Authorization: `Bearer ${accessToken}` } },
    );
    if (!res.ok) return [];
    const member = await res.json<{ roles: string[] }>();
    return member.roles ?? [];
  } catch {
    return [];
  }
}

function tierToApi(tier: TierConfig) {
  return {
    id:                    tier.id,
    name:                  tier.name,
    monthly_chapters:      tier.monthly_chapters,
    max_pages_per_chapter: tier.max_pages_per_chapter,
    concurrent_jobs:       tier.concurrent_jobs,
    sync_quota_bytes:      tier.sync_quota_bytes,
    can_use_api_tokens:    tier.can_use_api_tokens,
  };
}

function sessionUser(user: Awaited<ReturnType<typeof getUser>>, env: Env, roles: string[]): ApiSessionUser {
  if (!user) throw new Error("user missing");
  const tier = getTier(user.tier_id);
  return {
    id:                    user.id,
    display_name:          user.display_name,
    avatar_url:            user.avatar_url,
    email:                 user.email,
    is_admin:              !!env.ADMIN_ROLE_ID && roles.includes(env.ADMIN_ROLE_ID),
    preferred_target_lang: user.preferred_target_lang,
    tier:                  tierToApi(tier),
  };
}

// ── POST /discord/exchange ──────────────────────────────────────────

router.post("/discord/exchange", async (ctx) => {
  const { code, redirect_uri } = await ctx.req.json<{
    code:         string;
    redirect_uri: string;
  }>();

  if (!code || !redirect_uri) {
    return ctx.json({ error: "code and redirect_uri required" }, 400);
  }

  const tokenRes = await fetch(`${ctx.env.DISCORD_API}/oauth2/token`, {
    method:  "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({
      client_id:     ctx.env.DISCORD_CLIENT_ID,
      client_secret: ctx.env.DISCORD_CLIENT_SECRET,
      grant_type:    "authorization_code",
      code,
      redirect_uri,
    }),
  });

  if (!tokenRes.ok) {
    const body = await tokenRes.text();
    console.error("[auth/exchange] Discord rejected:", tokenRes.status, body);
    return ctx.json({ error: "Discord token exchange failed", detail: body }, 502);
  }

  const { access_token } = await tokenRes.json<{ access_token: string }>();

  // Fetch Discord user
  const userRes = await fetch(`${ctx.env.DISCORD_API}/users/@me`, {
    headers: { Authorization: `Bearer ${access_token}` },
  });
  if (!userRes.ok) return ctx.json({ error: "Discord user fetch failed" }, 502);

  const discordUser = await userRes.json<{
    id:          string;
    username:    string;
    global_name: string | null;
    avatar:      string | null;
    email:       string | null;
  }>();

  // Fetch guild roles → resolve tier
  const roles = await fetchGuildRoles(ctx.env, access_token);
  const tier  = resolveTierFromRoles(roles, parseRoleTierMap(ctx.env));

  const displayName = discordUser.global_name ?? discordUser.username;
  const avatarUrl   = discordUser.avatar
    ? `https://cdn.discordapp.com/avatars/${discordUser.id}/${discordUser.avatar}.png`
    : null;

  const user = await upsertDiscordUser(ctx.env.DB, {
    discord_id:    discordUser.id,
    display_name:  displayName,
    avatar_url:    avatarUrl,
    email:         discordUser.email,
    tier_id:       tier.id,
    discord_roles: roles,
  });

  const jwt = await issueJwt(user.id, roles, tier.id, ctx.env.JWT_SECRET);

  return ctx.json({
    token: jwt,
    user:  sessionUser(user, ctx.env, roles),
  });
});

// ── GET /auth/me (protected) ────────────────────────────────────────

router.get("/me", requireUser(), async (ctx) => {
  const userId = ctx.get("userId");
  const user   = await getUser(ctx.env.DB, userId);
  if (!user) return ctx.json({ error: "User not found" }, 404);
  return ctx.json(sessionUser(user, ctx.env, ctx.get("jwtRoles") ?? []));
});

// ── PATCH /auth/me/preferences (protected) ──────────────────────────

router.patch("/me/preferences", requireUser(), async (ctx) => {
  const userId = ctx.get("userId");
  const body   = await ctx.req.json<{ preferred_target_lang?: string | null }>();

  if (body.preferred_target_lang !== undefined) {
    await updateUserPreferences(ctx.env.DB, userId, body.preferred_target_lang);
  }

  const user = await getUser(ctx.env.DB, userId);
  if (!user) return ctx.json({ error: "User not found" }, 404);
  return ctx.json(sessionUser(user, ctx.env, ctx.get("jwtRoles") ?? []));
});

// ── POST /auth/refresh (protected) ──────────────────────────────────
//
// Re-sync tier from Discord guild role and reissue JWT. Only useful
// when user wants instant tier update after admin grants/revokes a role
// (otherwise wait for JWT TTL = 7d). Caller must supply Discord
// access token in body — we don't store Discord tokens.

router.post("/refresh", requireUser(), async (ctx) => {
  const userId = ctx.get("userId");
  const body   = await ctx.req.json<{ discord_access_token: string }>();
  if (!body.discord_access_token) {
    return ctx.json({ error: "discord_access_token required" }, 400);
  }

  const roles = await fetchGuildRoles(ctx.env, body.discord_access_token);
  const tier  = resolveTierFromRoles(roles, parseRoleTierMap(ctx.env));

  // Update users row with fresh tier
  await ctx.env.DB
    .prepare(
      `UPDATE users
       SET tier_id = ?, tier_synced_at = datetime('now'), discord_roles = ?
       WHERE id = ?`,
    )
    .bind(tier.id, JSON.stringify(roles), userId)
    .run();

  const user = await getUser(ctx.env.DB, userId);
  if (!user) return ctx.json({ error: "User not found" }, 404);

  const jwt = await issueJwt(userId, roles, tier.id, ctx.env.JWT_SECRET);
  return ctx.json({
    token: jwt,
    user:  sessionUser(user, ctx.env, roles),
  });
});

export default router;
