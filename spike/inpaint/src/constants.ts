/** Inpaint orchestration constants. */

// AOT-GAN supports a small set of square input sizes. Bubbles get padded
// up to the smallest bucket that fits; >384 is downscaled then upscaled.
export const BUCKETS = [128, 192, 256, 384] as const;

// Reflect-pad N pixels around each bubble bbox before bucketing.
// Gives the model context outside the masked region.
export const PAD_AROUND_BUBBLE = 16;

// Per-page tile fan-out tuning. See `shard.ts` for rationale.
//   TILES_PER_SHARD     = 3      tiles per container-call serial chunk
//   MAX_SHARDS_PER_PAGE = 6      cap on parallel container-calls per page
export const TILES_PER_SHARD     = 3;
export const MAX_SHARDS_PER_PAGE = 6;

// Class-aware mask close radii. Larger r bridges gaps inside SFX letterforms;
// smaller r is enough for tight dialogue text. Multiplied by the block's
// shorter edge (so radius scales with letter size on splash panels).
export const CLOSE_RADIUS_FRAC = {
  sfx:       0.15,
  narration: 0.12,
  dialogue:  0.10,
} as const;
export const CLOSE_RADIUS_MIN = 2;
