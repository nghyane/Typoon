/**
 * Discord OAuth + JWT issuance route.
 *
 * POST /auth/discord/exchange { code, redirect_uri }
 * GET  /auth/config            → public OAuth config for SPA
 */

import { Hono }     from "hono";
import { issueJwt } from "../middleware/auth";
import type { Env, ContextVars } from "../types";

const router = new Hono<{ Bindings: Env; Variables: ContextVars }>();

// ── GET /auth/config ──────────────────────────────────────────────────

router.get("/config", (ctx) => {
  if (!ctx.env.DISCORD_CLIENT_ID) {
    return ctx.json({ error: "Discord OAuth not configured" }, 503);
  }
  return ctx.json({ client_id: ctx.env.DISCORD_CLIENT_ID });
});

// ── POST /auth/discord/exchange ───────────────────────────────────────

router.post("/discord/exchange", async (ctx) => {
  const { code, redirect_uri } = await ctx.req.json<{
    code:         string;
    redirect_uri: string;
  }>();

  if (!code || !redirect_uri) {
    return ctx.json({ error: "code and redirect_uri required" }, 400);
  }

  const discordApi = ctx.env.DISCORD_API;

  // Exchange code for Discord access token
  const tokenRes = await fetch(`${discordApi}/oauth2/token`, {
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
    return ctx.json({ error: "Discord token exchange failed" }, 502);
  }

  const { access_token } = await tokenRes.json<{ access_token: string }>();

  // Fetch Discord user
  const userRes = await fetch(`${discordApi}/users/@me`, {
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

  // Fetch guild roles (optional — for admin gating)
  let roles: string[] = [];
  if (ctx.env.DISCORD_GUILD_ID) {
    try {
      const memberRes = await fetch(
        `${discordApi}/users/@me/guilds/${ctx.env.DISCORD_GUILD_ID}/member`,
        { headers: { Authorization: `Bearer ${access_token}` } },
      );
      if (memberRes.ok) {
        const member = await memberRes.json<{ roles: string[] }>();
        roles = member.roles;
      }
    } catch { /* guild not joined → roles stays [] */ }
  }

  const displayName = discordUser.global_name ?? discordUser.username;
  const avatarUrl   = discordUser.avatar
    ? `https://cdn.discordapp.com/avatars/${discordUser.id}/${discordUser.avatar}.png`
    : null;

  // Look up existing identity first — identity is the authoritative link
  const existingIdentity = await ctx.env.DB
    .prepare(
      `SELECT i.user_id FROM identities i
       WHERE i.provider = 'discord' AND i.external_id = ?`,
    )
    .bind(discordUser.id)
    .first<{ user_id: number }>();

  let userId: number;

  if (existingIdentity) {
    // Known user — update profile fields
    userId = existingIdentity.user_id;
    await ctx.env.DB
      .prepare(
        `UPDATE users
         SET display_name = ?, avatar_url = ?, last_login_at = datetime('now')
         WHERE id = ?`,
      )
      .bind(displayName, avatarUrl, userId)
      .run();
  } else {
    // New user — insert user row then identity row
    const newUser = await ctx.env.DB
      .prepare(
        `INSERT INTO users (display_name, avatar_url, email, last_login_at)
         VALUES (?, ?, ?, datetime('now'))
         RETURNING id`,
      )
      .bind(displayName, avatarUrl, discordUser.email ?? null)
      .first<{ id: number }>();

    if (!newUser) return ctx.json({ error: "User creation failed" }, 500);
    userId = newUser.id;
  }

  // Upsert identity (handles re-login after identity row was deleted)
  await ctx.env.DB
    .prepare(
      `INSERT INTO identities (user_id, provider, external_id, metadata)
       VALUES (?, 'discord', ?, ?)
       ON CONFLICT(provider, external_id) DO UPDATE
         SET metadata = excluded.metadata`,
    )
    .bind(userId, discordUser.id, JSON.stringify({ username: discordUser.username }))
    .run();

  const jwt = await issueJwt(userId, roles, ctx.env.JWT_SECRET);

  return ctx.json({
    token:        jwt,
    user_id:      userId,
    display_name: displayName,
    avatar_url:   avatarUrl,
  });
});

export default router;
