/** Tile builder + compositor.
 *
 * One tile per bubble:
 *   - Pad the bubble bbox by PAD_AROUND_BUBBLE
 *   - Round up to the smallest AOT bucket size
 *   - REFLECT_101 outside the padded region (avoids "false content" hallucination
 *     when the bubble touches a page edge)
 *   - Pack RGB(W·H·3) ++ mask(W·H) into a single Uint8Array body */

import { PAD_AROUND_BUBBLE } from "./constants";
import { pickBucket, type BBox } from "./bubbles";

export interface Tile {
  /** Padded bbox in source coords (used by composeTile to know where to paste back). */
  bbox: BBox;
  /** Padded tile width (rounded to bucket). */
  W: number;
  /** Padded tile height (rounded to bucket). */
  H: number;
  /** Packed body: RGB(W·H·3) ++ mask(W·H). The shape the container expects. */
  body: Uint8Array;
}

/** Build one tile crop with REFLECT_101 padding to the chosen bucket. */
export function buildTile(
  img: Uint8Array, mask: Uint8Array,
  srcW: number, srcH: number, bb: BBox,
): Tile {
  const x0 = Math.max(0, bb.x0 - PAD_AROUND_BUBBLE);
  const y0 = Math.max(0, bb.y0 - PAD_AROUND_BUBBLE);
  const x1 = Math.min(srcW - 1, bb.x1 + PAD_AROUND_BUBBLE);
  const y1 = Math.min(srcH - 1, bb.y1 + PAD_AROUND_BUBBLE);
  const w0 = x1 - x0 + 1, h0 = y1 - y0 + 1;
  const tileW = pickBucket(w0), tileH = pickBucket(h0);

  const rgbLen = tileW * tileH * 3;
  const body   = new Uint8Array(rgbLen + tileW * tileH);
  const rgb    = body.subarray(0, rgbLen);
  const msk    = body.subarray(rgbLen);

  // REFLECT_101: for index t > orig_size, mirror back without repeating the edge pixel.
  // `img` is RGBA (stride 4), `mask` is W×H uint8 (stride 1) — different strides.
  for (let ty = 0; ty < tileH; ty++) {
    let sy = ty < h0 ? ty : 2 * (h0 - 1) - ty;
    if (sy < 0) sy = 0; if (sy >= h0) sy = h0 - 1;
    const srcY = y0 + sy;
    for (let tx = 0; tx < tileW; tx++) {
      let sx = tx < w0 ? tx : 2 * (w0 - 1) - tx;
      if (sx < 0) sx = 0; if (sx >= w0) sx = w0 - 1;
      const srcX = x0 + sx;
      const rgbaI = (srcY * srcW + srcX) * 4;   // image stride 4 (RGBA)
      const maskI = srcY * srcW + srcX;          // mask stride 1
      const di    = ty * tileW + tx;
      rgb[di * 3    ] = img[rgbaI    ];
      rgb[di * 3 + 1] = img[rgbaI + 1];
      rgb[di * 3 + 2] = img[rgbaI + 2];
      msk[di]         = mask[maskI];
    }
  }
  return { bbox: { x0, y0, x1, y1 }, W: tileW, H: tileH, body };
}

/** Paste tile RGB back into composite, only where the ORIGINAL mask is set.
 *  Mutates `composite` in place. `composite` is RGBA stride 4, `origMask`
 *  is W×H uint8 stride 1. */
export function composeTile(
  composite: Uint8Array, srcW: number,
  origMask: Uint8Array,
  tile: Tile, tileRgb: Uint8Array,
): void {
  const { bbox, W: tileW, H: tileH } = tile;
  const w0 = bbox.x1 - bbox.x0 + 1, h0 = bbox.y1 - bbox.y0 + 1;
  for (let ty = 0; ty < h0 && ty < tileH; ty++) {
    const dstY = bbox.y0 + ty;
    for (let tx = 0; tx < w0 && tx < tileW; tx++) {
      const dstX  = bbox.x0 + tx;
      const dstI  = (dstY * srcW + dstX) * 4;   // composite RGBA stride
      const maskI = dstY * srcW + dstX;          // origMask stride 1
      if (origMask[maskI] < 127) continue;
      const ti = (ty * tileW + tx) * 3;
      composite[dstI    ] = tileRgb[ti    ];
      composite[dstI + 1] = tileRgb[ti + 1];
      composite[dstI + 2] = tileRgb[ti + 2];
    }
  }
}
