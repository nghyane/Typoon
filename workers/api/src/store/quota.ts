/**
 * Quota store — D1 queries and rate limiting checks.
 */

import type { D1Database } from "@cloudflare/workers-types";
import { RateLimitError }  from "./db";

const HOUR_SEC = 3600;
const DAY_SEC  = 86400;

export interface RateLimitConfig {
  chapters_per_hour: number;
  chapters_per_day:  number;
}

export const DEFAULT_LIMITS: RateLimitConfig = {
  chapters_per_hour: 10,
  chapters_per_day:  50,
};

export function isAdmin(roles: string[], adminRoleId?: string | null): boolean {
  return !!adminRoleId && roles.includes(adminRoleId);
}

export async function countUserConsumesSince(
  db: D1Database,
  userId: number,
  seconds: number,
): Promise<number> {
  const row = await db
    .prepare(
      `SELECT COUNT(*) AS n FROM chapter_consumes
       WHERE user_id = ?
         AND created_at > datetime('now', '-' || ? || ' seconds')`,
    )
    .bind(userId, seconds)
    .first<{ n: number }>();
  return row ? Number(row.n) : 0;
}

export async function getQuotaSnapshot(
  db: D1Database,
  args: {
    userId:      number;
    roles:       string[];
    adminRoleId?: string | null;
    limits?:     RateLimitConfig;
  },
): Promise<any> {
  const limits = args.limits ?? DEFAULT_LIMITS;
  const admin = isAdmin(args.roles, args.adminRoleId);
  const usedHour = await countUserConsumesSince(db, args.userId, HOUR_SEC);
  const usedDay  = await countUserConsumesSince(db, args.userId, DAY_SEC);

  return {
    is_admin:       admin,
    limit_hour:     limits.chapters_per_hour,
    used_hour:      usedHour,
    remaining_hour: Math.max(0, limits.chapters_per_hour - usedHour),
    limit_day:      limits.chapters_per_day,
    used_day:       usedDay,
    remaining_day:  Math.max(0, limits.chapters_per_day - usedDay),
  };
}

export async function enforceChapterQuota(
  db: D1Database,
  args: {
    userId:      number;
    roles:       string[];
    adminRoleId?: string | null;
    count?:      number;
    limits?:     RateLimitConfig;
  },
): Promise<void> {
  const count = args.count ?? 1;
  if (isAdmin(args.roles, args.adminRoleId)) return;
  if (count <= 0) return;

  const limits = args.limits ?? DEFAULT_LIMITS;

  const usedHour = await countUserConsumesSince(db, args.userId, HOUR_SEC);
  if (usedHour + count > limits.chapters_per_hour) {
    throw new RateLimitError(
      `Đã dùng ${usedHour}/${limits.chapters_per_hour} chương trong giờ này. Thử lại sau ít phút.`,
      HOUR_SEC,
    );
  }

  const usedDay = await countUserConsumesSince(db, args.userId, DAY_SEC);
  if (usedDay + count > limits.chapters_per_day) {
    throw new RateLimitError(
      `Đã dùng ${usedDay}/${limits.chapters_per_day} chương hôm nay. Quota reset mỗi 24h.`,
      DAY_SEC,
    );
  }
}
