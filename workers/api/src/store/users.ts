/**
 * Users store — D1 queries for users, identities and api_tokens tables.
 */

import type { D1Database } from "@cloudflare/workers-types";

export interface UserRow {
  id:                    number;
  display_name:          string;
  avatar_url:            string | null;
  email:                 string | null;
  preferred_target_lang: string | null;
  created_at:            string;
  last_login_at:         string | null;
}

export interface ApiTokenRow {
  id:          number;
  user_id:     number;
  name:        string;
  token_hash:  string;
  prefix:      string;
  scopes:      string; // JSON string
  last_used:   string | null;
  created_at:  string;
  revoked_at:  string | null;
}

export async function getUser(
  db: D1Database,
  userId: number,
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
): Promise<UserRow | null> {
  return db
    .prepare(
      `UPDATE users
       SET preferred_target_lang = ?
       WHERE id = ?
       RETURNING *`,
    )
    .bind(preferred_target_lang, userId)
    .first<UserRow>();
}

export async function createApiToken(
  db: D1Database,
  args: {
    user_id:    number;
    name:       string;
    token_hash: string;
    prefix:     string;
    scopes:     string[];
  },
): Promise<ApiTokenRow> {
  const row = await db
    .prepare(
      `INSERT INTO api_tokens (user_id, name, token_hash, prefix, scopes)
       VALUES (?, ?, ?, ?, ?)
       RETURNING *`,
    )
    .bind(
      args.user_id,
      args.name,
      args.token_hash,
      args.prefix,
      JSON.stringify(args.scopes),
    )
    .first<ApiTokenRow>();

  if (!row) throw new Error("API token insert failed");
  return row;
}

export async function listActiveApiTokens(
  db: D1Database,
  userId: number,
): Promise<ApiTokenRow[]> {
  const { results } = await db
    .prepare(
      `SELECT * FROM api_tokens
       WHERE user_id = ? AND revoked_at IS NULL
       ORDER BY created_at DESC`,
    )
    .bind(userId)
    .all<ApiTokenRow>();
  return results;
}

export async function revokeApiToken(
  db: D1Database,
  userId: number,
  tokenId: number,
): Promise<boolean> {
  const result = await db
    .prepare(
      `UPDATE api_tokens
       SET revoked_at = datetime('now')
       WHERE id = ? AND user_id = ? AND revoked_at IS NULL`,
    )
    .bind(tokenId, userId)
    .run();
  return (result.meta.changes ?? 0) > 0;
}
