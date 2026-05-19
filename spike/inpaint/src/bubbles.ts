/** Flood-fill connected components on the closed mask.
 *
 * Each connected on-pixel becomes one "bubble" — a region the inpaint
 * model will fill. Bbox-only output (we don't need per-pixel masks
 * downstream; the orig mask is consulted again at compose time).
 *
 * Mask is flat W×H uint8 (stride 1, MSK1 wire format), not RGBA. */

import { BUCKETS } from "./constants";

export interface BBox { x0: number; y0: number; x1: number; y1: number; }

/** Iterative flood-fill over a flat W×H uint8 mask (>= 127 = on). */
export function findBubbles(mask: Uint8Array, W: number, H: number): BBox[] {
  const visited = new Uint8Array(W * H);
  const bboxes:  BBox[] = [];
  const stack:   number[] = [];

  for (let py = 0; py < H; py++) {
    for (let px = 0; px < W; px++) {
      const idx = py * W + px;
      if (visited[idx] || mask[idx] < 127) continue;

      let x0 = px, y0 = py, x1 = px, y1 = py;
      stack.push(idx);
      visited[idx] = 1;

      while (stack.length) {
        const cur = stack.pop()!;
        const cx  = cur % W;
        const cy  = (cur - cx) / W;
        if (cx < x0) x0 = cx; if (cx > x1) x1 = cx;
        if (cy < y0) y0 = cy; if (cy > y1) y1 = cy;
        if (cx > 0)     { const n = cur - 1; if (!visited[n] && mask[n] >= 127) { visited[n] = 1; stack.push(n); } }
        if (cx < W - 1) { const n = cur + 1; if (!visited[n] && mask[n] >= 127) { visited[n] = 1; stack.push(n); } }
        if (cy > 0)     { const n = cur - W; if (!visited[n] && mask[n] >= 127) { visited[n] = 1; stack.push(n); } }
        if (cy < H - 1) { const n = cur + W; if (!visited[n] && mask[n] >= 127) { visited[n] = 1; stack.push(n); } }
      }
      bboxes.push({ x0, y0, x1, y1 });
    }
  }
  return bboxes;
}

/** Smallest bucket that fits `side`, else the largest bucket. */
export function pickBucket(side: number): number {
  for (const b of BUCKETS) if (side <= b) return b;
  return BUCKETS[BUCKETS.length - 1];
}
