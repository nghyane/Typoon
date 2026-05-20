/**
 * D1 store helpers.
 *
 * All queries go through these helpers so FK enforcement and
 * error normalization stay consistent across the codebase.
 */

import type { D1Database, D1Result } from "@cloudflare/workers-types";

// D1 requires PRAGMA foreign_keys = ON per-connection.
// Call once at Worker startup or wrap critical mutations.
export async function enforceFk(db: D1Database): Promise<void> {
  await db.prepare("PRAGMA foreign_keys = ON").run();
}

// Run a batch of statements atomically.
export async function batch(
  db: D1Database,
  stmts: D1PreparedStatement[],
): Promise<D1Result[]> {
  return db.batch(stmts);
}

// Assert exactly one row returned; throw 404-style error if missing.
export function requireRow<T>(
  row: T | null,
  label: string,
): T {
  if (!row) throw new NotFoundError(`${label} not found`);
  return row;
}

// Normalised pagination: clamp page size to max 100.
export function paginate(limit?: number, offset?: number) {
  return {
    limit:  Math.min(limit  ?? 50, 100),
    offset: offset ?? 0,
  };
}

// ── Custom errors ────────────────────────────────────────────────────

export class NotFoundError extends Error {
  readonly status = 404;
  constructor(msg: string) { super(msg); this.name = "NotFoundError"; }
}

export class ConflictError extends Error {
  readonly status = 409;
  constructor(msg: string) { super(msg); this.name = "ConflictError"; }
}

export class ForbiddenError extends Error {
  readonly status = 403;
  constructor(msg: string) { super(msg); this.name = "ForbiddenError"; }
}

export class RateLimitError extends Error {
  readonly status     = 429;
  readonly retryAfter: number;
  constructor(msg: string, retryAfterSec: number) {
    super(msg);
    this.name       = "RateLimitError";
    this.retryAfter = retryAfterSec;
  }
}
