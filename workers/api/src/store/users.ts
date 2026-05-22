/**
 * Users store — D1 queries for `users`, `identities`, `api_tokens`.
 */

import type { D1Database } from "@cloudflare/workers-types";

export interface UserRow {
  id:                    number;
  discord_id:            string;
  display_name:          string;
  avatar_url:            string | null;
  email:                 string | null;
  preferred_target_lang: string | null;
  tier_id:               string;
  tier_synced_at:        string | null;
  discord_roles:         string | null;   // JSON array
  created_at:            string;
  last_login_at:         string | null;
}

export interface ApiTokenRow {
  id:          number;
  user_id:     number;
  name:        string;
  token_hash:  string;
  prefix:      string;
  scopes:      string;   // JSON array
  last_used:   string | null;
  created_at:  string;
  revoked_at:  string | null;
}

export async function getUser(
  db: D1Database, userId: number,
): Promise<UserRow | null> {
  return db
    .prepare("SELECT * FROM users WHERE id = ?")
    .bind(userId)
    .first<UserRow>();
}

export async function updateUserPreferences(
  db: D1Database,
  userId: number,
  preferred_target_lang: string | null,
): Promise<void> {
  await db
    .prepare(`UPDATE users SET preferred_target_lang = ? WHERE id = ?`)
    .bind(preferred_target_lang, userId)
    .run();
}

/**
 * Upsert Discord user → users row + identities row.
 * Updates tier_id and discord_roles from the freshly fetched role set.
 */
export async function upsertDiscordUser(
  db: D1Database,
  args: {
    discord_id:    string;
    display_name:  string;
    avatar_url:    string | null;
    email:         string | null;
    tier_id:       string;
    discord_roles: string[];
  },
): Promise<UserRow> {
  // Lookup by identity (authoritative cross-source link).
  const existing = await db
    .prepare(
      `SELECT u.* FROM users u
       JOIN identities i ON i.user_id = u.id
       WHERE i.provider = 'discord' AND i.external_id = ?`,
    )
    .bind(args.discord_id)
    .first<UserRow>();

  const rolesJson = JSON.stringify(args.discord_roles);

  if (existing) {
    const updated = await db
      .prepare(
        `UPDATE users
         SET display_name = ?, avatar_url = ?, email = ?,
             tier_id = ?, tier_synced_at = datetime('now'),
             discord_roles = ?, last_login_at = datetime('now')
         WHERE id = ?
         RETURNING *`,
      )
      .bind(
        args.display_name, args.avatar_url, args.email,
        args.tier_id, rolesJson,
        existing.id,
      )
      .first<UserRow>();
    if (!updated) throw new Error("User update failed");
    return updated;
  }

  // Insert new user
  const created = await db
    .prepare(
      `INSERT INTO users (
         discord_id, display_name, avatar_url, email,
         tier_id, tier_synced_at, discord_roles, last_login_at
       )
       VALUES (?, ?, ?, ?, ?, datetime('now'), ?, datetime('now'))
       RETURNING *`,
    )
    .bind(
      args.discord_id, args.display_name, args.avatar_url, args.email,
      args.tier_id, rolesJson,
    )
    .first<UserRow>();

  if (!created) throw new Error("User insert failed");

  // Insert identity link
  await db
    .prepare(
      `INSERT INTO identities (user_id, provider, external_id, metadata)
       VALUES (?, 'discord', ?, ?)`,
    )
    .bind(created.id, args.discord_id, JSON.stringify({ display_name: args.display_name }))
    .run();

  return created;
}

// ── API token verification ─────────────────────────────────────────
// Mint/list/revoke endpoints not implemented yet; helpers will live
// alongside the route when added (docs/rfc/008-api-tokens.md).
