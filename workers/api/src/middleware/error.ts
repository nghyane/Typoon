/**
 * Global error handler middleware for Hono.
 *
 * Catches known domain errors from `store/db.ts` and maps them to
 * structured `{ error: { code, message, details? } }` responses.
 * Unknown errors become 500 with a generic message (no stack leak).
 */

import type { MiddlewareHandler } from "hono";
import type { Env, ContextVars } from "../types";
import { NotFoundError, ConflictError, ForbiddenError, RateLimitError } from "../store/db";

export function errorHandler(): MiddlewareHandler<{ Bindings: Env; Variables: ContextVars }> {
  return async (ctx, next) => {
    try {
      await next();
    } catch (err: any) {
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
      if (err?.code === "SQLITE_CONSTRAINT") {
        return ctx.json(
          { error: { code: "conflict", message: err.message } },
          409,
        );
      }

      // Hono HTTPException passthrough
      if (err?.status && err?.message) {
        return ctx.json(
          { error: { code: "http_error", message: err.message } },
          err.status,
        );
      }

      // Fallback — 500, no stack trace to client
      console.error("[unhandled]", err);
      return ctx.json(
        { error: { code: "internal_error", message: "Internal server error" } },
        500,
      );
    }
  };
}
