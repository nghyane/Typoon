/**
 * Cursor-based pagination helpers.
 *
 * Uses base64-encoded JSON `{ id, ts }` as the cursor token.
 * The client passes `?cursor=<token>&limit=N`; the server decodes
 * and appends `WHERE (updated_at, id) < (:cursor_ts, :cursor_id)`.
 */

export interface CursorToken {
  id: number;
  ts: string;
}

/** Encode a cursor token to a URL-safe string. */
export function encodeCursor(token: CursorToken): string {
  const json = JSON.stringify(token);
  return btoa(json);
}

/** Decode a cursor token from a query param. Returns null on bad input. */
export function decodeCursor(raw: string | null | undefined): CursorToken | null {
  if (!raw) return null;
  try {
    const json = atob(raw.trim());
    const parsed = JSON.parse(json);
    if (typeof parsed.id === "number" && typeof parsed.ts === "string") {
      return parsed;
    }
    return null;
  } catch {
    return null;
  }
}

/** Parse cursor + limit from query params with safe defaults. */
export function parseCursorQuery(
  cursor: string | null | undefined,
  limit: string | null | undefined,
  defaultLimit = 20,
  maxLimit = 100,
): { cursor: CursorToken | null; limit: number } {
  return {
    cursor: decodeCursor(cursor),
    limit: Math.min(Math.max(Number(limit) || defaultLimit, 1), maxLimit),
  };
}

/** Build the WHERE clause for cursor pagination.
 *  Returns { sql, params } to append to a query.
 *  Uses (updated_at, id) composite ordering — descending.
 */
export function cursorWhere(
  cursor: CursorToken | null,
  tsColumn = "updated_at",
  idColumn = "id",
): { sql: string; params: (string | number)[] } {
  if (!cursor) return { sql: "", params: [] };
  return {
    sql: ` AND (${tsColumn}, ${idColumn}) < (?, ?)`,
    params: [cursor.ts, cursor.id],
  };
}

/** Encode the last item in a result set as a next_cursor. */
export function nextCursorFromItem(
  item: { id: number; updated_at: string } | null | undefined,
): string | null {
  if (!item) return null;
  return encodeCursor({ id: item.id, ts: item.updated_at });
}
