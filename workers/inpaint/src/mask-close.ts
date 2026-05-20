/** Morphological closing on the per-page erase mask.
 *
 * Inpaint quality depends on each AOT-GAN tile seeing the WHOLE text region
 * at once. Without closing, `findBubbles` emits one bbox per word (gaps
 * between words = blank pixels = separate components), so a 5-word bubble
 * runs 5 separate inferences and the model leaks bubble-border texture into
 * each tile — the horizontal-streak artefact.
 *
 * Strategy: close per-block with class-aware radii (sfx > narration > dialogue),
 * then apply an outsider guard so a big-radius SFX closing never bridges into
 * a neighbouring dialogue bubble.
 *
 * Input and output are flat W×H uint8 (stride 1, matching the MSK1 wire
 * format), so callers can pass the raw decode straight through. */

import type { ScanPageResult, BlockClass } from "../../shared/src/types";
import { CLOSE_RADIUS_FRAC, CLOSE_RADIUS_MIN } from "./constants";

// ── Rectangular dilate/erode (separable, O(W·H·r) each pass) ─────────────────
//
// Rectangular kernel is fine for closing — small over-dilation in corners is
// invisible after the matching erode pass. Two single-axis sweeps are far
// cheaper than a true 2D N×N kernel.

function dilateRect(src: Uint8Array, W: number, H: number, r: number): Uint8Array {
  if (r <= 0) return src.slice();
  const h1 = new Uint8Array(W * H);
  // horizontal pass: out[x] = OR(src[x-r..x+r])
  for (let y = 0; y < H; y++) {
    const row = y * W;
    for (let x = 0; x < W; x++) {
      const lo = Math.max(0, x - r), hi = Math.min(W - 1, x + r);
      let on = 0;
      for (let xx = lo; xx <= hi; xx++) if (src[row + xx]) { on = 1; break; }
      h1[row + x] = on;
    }
  }
  const out = new Uint8Array(W * H);
  // vertical pass
  for (let x = 0; x < W; x++) {
    for (let y = 0; y < H; y++) {
      const lo = Math.max(0, y - r), hi = Math.min(H - 1, y + r);
      let on = 0;
      for (let yy = lo; yy <= hi; yy++) if (h1[yy * W + x]) { on = 1; break; }
      out[y * W + x] = on;
    }
  }
  return out;
}

function erodeRect(src: Uint8Array, W: number, H: number, r: number): Uint8Array {
  if (r <= 0) return src.slice();
  const h1 = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) {
    const row = y * W;
    for (let x = 0; x < W; x++) {
      const lo = Math.max(0, x - r), hi = Math.min(W - 1, x + r);
      let on = 1;
      for (let xx = lo; xx <= hi; xx++) if (!src[row + xx]) { on = 0; break; }
      h1[row + x] = on;
    }
  }
  const out = new Uint8Array(W * H);
  for (let x = 0; x < W; x++) {
    for (let y = 0; y < H; y++) {
      const lo = Math.max(0, y - r), hi = Math.min(H - 1, y + r);
      let on = 1;
      for (let yy = lo; yy <= hi; yy++) if (!h1[yy * W + x]) { on = 0; break; }
      out[y * W + x] = on;
    }
  }
  return out;
}

/** Per-block class scaled by short edge. */
function closeRadiusForClass(cls: BlockClass, bb: readonly [number, number, number, number]): number {
  const short = Math.min(bb[2] - bb[0], bb[3] - bb[1]);
  const frac  = CLOSE_RADIUS_FRAC[cls as keyof typeof CLOSE_RADIUS_FRAC] ?? CLOSE_RADIUS_FRAC.dialogue;
  return Math.max(CLOSE_RADIUS_MIN, Math.round(short * frac));
}

/** Close per-group with outsider guard. Mutates nothing; returns fresh
 *  flat W×H uint8 mask (stride 1, 0 or 255). */
export function closeMaskPerBlock(
  mask: Uint8Array, W: number, H: number,
  groups: ScanPageResult["groups"],
): Uint8Array {
  // Input mask is W×H uint8 (stride 1) from decodeMaskBin.
  const bin = new Uint8Array(W * H);
  for (let i = 0; i < W * H; i++) bin[i] = mask[i] >= 127 ? 1 : 0;

  const result = bin.slice();

  for (const b of groups) {
    const [bx1, by1, bx2, by2] = b.bbox;
    const r   = closeRadiusForClass(b.class, b.bbox);
    const px0 = Math.max(0, bx1 - r), py0 = Math.max(0, by1 - r);
    const px1 = Math.min(W, bx2 + r), py1 = Math.min(H, by2 + r);
    const pw  = px1 - px0, ph = py1 - py0;
    if (pw <= 0 || ph <= 0) continue;

    // Block-local patch
    const patch = new Uint8Array(pw * ph);
    for (let y = 0; y < ph; y++)
      for (let x = 0; x < pw; x++)
        patch[y * pw + x] = bin[(py0 + y) * W + (px0 + x)];

    const dilated = dilateRect(patch, pw, ph, r);
    const closed  = erodeRect(dilated, pw, ph, r);

    // Outsider centres of all OTHER groups
    const others = groups
      .filter(o => o.idx !== b.idx)
      .map(o => ({ cx: (o.bbox[0] + o.bbox[2]) / 2, cy: (o.bbox[1] + o.bbox[3]) / 2 }));

    // Write closed patch back; skip newly-set pixels if an outsider centre
    // falls inside the closed bbox (would bridge into neighbour bubble).
    for (let y = 0; y < ph; y++) {
      for (let x = 0; x < pw; x++) {
        if (!closed[y * pw + x]) continue;
        const gx = px0 + x, gy = py0 + y;
        if (!bin[gy * W + gx]) {
          const bridged = others.some(c =>
            c.cx >= px0 && c.cx < px1 && c.cy >= py0 && c.cy < py1,
          );
          if (bridged) continue;
        }
        result[gy * W + gx] = 1;
      }
    }
  }

  // Emit flat W×H uint8 (0 or 255) — same stride as the input.
  const out = new Uint8Array(W * H);
  for (let i = 0; i < W * H; i++) out[i] = result[i] ? 255 : 0;
  return out;
}
