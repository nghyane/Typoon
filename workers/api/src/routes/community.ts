/**
 * Community routes — shared translation feed.
 *
 * GET /api/community/feed?cursor=&limit= → cursor-paginated feed
 */

import { Hono } from "hono";
import type { Env, ContextVars } from "../types";
import { listCommunityFeed } from "../store/community";

type AppEnv = { Bindings: Env; Variables: ContextVars };
const router = new Hono<AppEnv>();

router.get("/feed", async (ctx) => {
  const userId = ctx.get("userId");
  const cursor = ctx.req.query("cursor") ?? undefined;
  const limit = ctx.req.query("limit");

  const result = await listCommunityFeed(ctx.env.DB, {
    cursor,
    limit: limit ? Number(limit) : undefined,
  });

  return ctx.json(result);
});

export default router;
