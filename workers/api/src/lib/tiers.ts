/**
 * Tier configuration — quota and feature limits per user tier.
 *
 * Tiers are derived from Discord roles in the Typoon community guild
 * during /auth/discord/exchange. Map: env.DISCORD_ROLE_TIER_MAP (JSON).
 *
 * Highest priority tier wins when user holds multiple tier roles.
 * No tier role → free.
 *
 * Quota is enforced at job creation (jobs.ts). The tier ID is cached
 * in users.tier_id and embedded in JWT claims for fast lookup.
 */

export interface TierConfig {
  /** Stable identifier — referenced by users.tier_id and JWT claims. */
  id:                    string;
  /** Display name for UI. */
  name:                  string;
  /** Hard cap on chapter translations per calendar month (UTC). */
  monthly_chapters:      number;
  /** Reject jobs whose prepared page count exceeds this. */
  max_pages_per_chapter: number;
  /** Max non-terminal jobs in flight (init/uploading/pending/running). */
  concurrent_jobs:       number;
  /** Hard cap on user_data.byte_size for cross-device sync. */
  sync_quota_bytes:      number;
  /** Whether user can mint API tokens for programmatic access. */
  can_use_api_tokens:    boolean;
  /** Pipeline priority — higher = served first (reserved, not wired yet). */
  priority:              number;
}

export const TIERS = {
  free: {
    id:                    "free",
    name:                  "Miễn phí",
    monthly_chapters:      20,
    max_pages_per_chapter: 200,
    concurrent_jobs:       2,
    sync_quota_bytes:      1_000_000,
    can_use_api_tokens:    false,
    priority:              0,
  },
  supporter: {
    id:                    "supporter",
    name:                  "Supporter",
    monthly_chapters:      100,
    max_pages_per_chapter: 200,
    concurrent_jobs:       3,
    sync_quota_bytes:      5_000_000,
    can_use_api_tokens:    true,
    priority:              1,
  },
  pro: {
    id:                    "pro",
    name:                  "Pro",
    monthly_chapters:      500,
    max_pages_per_chapter: 200,
    concurrent_jobs:       5,
    sync_quota_bytes:      20_000_000,
    can_use_api_tokens:    true,
    priority:              2,
  },
  unlimited: {
    id:                    "unlimited",
    name:                  "Unlimited",
    monthly_chapters:      99_999,
    max_pages_per_chapter: 500,
    concurrent_jobs:       10,
    sync_quota_bytes:      100_000_000,
    can_use_api_tokens:    true,
    priority:              3,
  },
} as const satisfies Record<string, TierConfig>;

export type TierId = keyof typeof TIERS;

export function getTier(id: string | null | undefined): TierConfig {
  if (id && id in TIERS) return TIERS[id as TierId];
  return TIERS.free;
}

/**
 * Resolve user's tier from Discord role IDs.
 *
 * Discord role ID → tier ID mapping comes from `DISCORD_ROLE_TIER_MAP`
 * env var (JSON, e.g. `{"role_id_a":"supporter","role_id_b":"pro"}`).
 * When user holds multiple tier roles, the highest-priority tier wins.
 * No matching role → free tier.
 */
export function resolveTierFromRoles(
  roleIds:     string[],
  roleTierMap: Record<string, string>,
): TierConfig {
  const matched = roleIds
    .map(r => roleTierMap[r])
    .filter((t): t is string => !!t && t in TIERS)
    .map(t => TIERS[t as TierId]);

  if (matched.length === 0) return TIERS.free;
  return matched.sort((a, b) => b.priority - a.priority)[0]!;
}
