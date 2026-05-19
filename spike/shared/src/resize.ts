/**
 * OpenCV INTER_AREA equivalent — weighted area average with sub-pixel
 * overlap. Used by prepare (color-ratio sample, modal-width fit) and
 * scan (tile resize to Lens ≤1200 px).
 *
 * Matches cv2.resize(_, INTER_AREA) to within ±1 LSB; the residual
 * comes from float vs Python's int rounding in OpenCV's saturate_cast.
 */

export function resizeRGBA(
  src: Uint8ClampedArray, sw: number, sh: number, dw: number, dh: number,
): Uint8ClampedArray {
  if (dw === sw && dh === sh) return src;
  const out = new Uint8ClampedArray(dw * dh * 4);
  const xRatio = sw / dw, yRatio = sh / dh;

  // Precompute x-axis source ranges & weights once per scanline.
  const xStarts = new Array<number>(dw);
  const xEnds   = new Array<number>(dw);
  const xWeights: Float64Array[] = new Array(dw);
  for (let x = 0; x < dw; x++) {
    const fs = x * xRatio, fe = (x + 1) * xRatio;
    const i0 = Math.floor(fs), i1 = Math.min(sw, Math.ceil(fe));
    const w = new Float64Array(i1 - i0);
    for (let xx = i0; xx < i1; xx++) {
      w[xx - i0] = Math.max(0, Math.min(fe, xx + 1) - Math.max(fs, xx));
    }
    xStarts[x] = i0; xEnds[x] = i1; xWeights[x] = w;
  }

  for (let y = 0; y < dh; y++) {
    const fs = y * yRatio, fe = (y + 1) * yRatio;
    const j0 = Math.floor(fs), j1 = Math.min(sh, Math.ceil(fe));
    for (let x = 0; x < dw; x++) {
      const i0 = xStarts[x], i1 = xEnds[x], xw = xWeights[x];
      let r = 0, g = 0, b = 0, a = 0, totW = 0;
      for (let yy = j0; yy < j1; yy++) {
        const yWeight = Math.max(0, Math.min(fe, yy + 1) - Math.max(fs, yy));
        if (yWeight === 0) continue;
        const rowBase = yy * sw * 4;
        for (let xx = i0; xx < i1; xx++) {
          const w = yWeight * xw[xx - i0];
          if (w === 0) continue;
          const i = rowBase + xx * 4;
          r += src[i] * w; g += src[i + 1] * w; b += src[i + 2] * w; a += src[i + 3] * w;
          totW += w;
        }
      }
      const o = (y * dw + x) * 4;
      if (totW > 0) {
        out[o]     = Math.round(r / totW);
        out[o + 1] = Math.round(g / totW);
        out[o + 2] = Math.round(b / totW);
        out[o + 3] = Math.round(a / totW);
      }
    }
  }
  return out;
}
