/**
 * Typed errors thrown by store/route layers and shaped into JSON responses
 * by the global `onError` handler in `index.ts`. Throw one of these instead
 * of returning ad-hoc `ctx.json({error: ...}, status)` so the error shape
 * stays uniform (`{ error: { code, message } }`).
 */

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
