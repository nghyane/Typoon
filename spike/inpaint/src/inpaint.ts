/** Inpaint-page orchestration.
 *
 *   prepared JPEG + raw mask (.bin) + scan msgpack (R2)
 *     → decode → close mask per-block → flood-fill bubbles
 *     → build tiles (REFLECT_101 padded, bucketed) → shard fan-out to container
 *     → compose tiles back into page → encode PNG → write R2
 *
 * Pure function — takes its dependencies (R2, container stub) as args.
 *
 * Mask format: "MSK1" (4B) + u16LE width (2B) + u16LE height (2B) + W×H uint8
 * Zero decode overhead vs PNG — np.frombuffer equivalent in JS. */

import { decodeJpeg } from "../../shared/src/codec/jpeg-dec";
import { decodePng, isPng } from "../../shared/src/codec/png-dec";
import { encodePng } from "../../shared/src/codec/png-enc";
import { getBytes, putBytes, keys as r2keys } from "../../shared/src/r2";
import type { ScanPageResult } from "../../shared/src/types";
import { decodeMsgpack } from "../../shared/src/codec/msgpack";

import { closeMaskPerBlock } from "./mask-close";
import { findBubbles } from "./bubbles";
import { buildTile } from "./tile";
import { shardAndCompose, type InpaintEnv } from "./shard";

export interface InpaintPageArgs {
  chapter_id: string;
  page_index: number;
  image_key:  string;
  mask_key:   string;
  scan_key:   string;
}

export interface InpaintPageResult {
  output_key:  string;
  bubbles:     number;
  tiles_shape: string[];
}

function toClamped(buf: Uint8Array): Uint8ClampedArray {
  return new Uint8ClampedArray(buf.buffer, buf.byteOffset, buf.byteLength);
}

/** Decode raw mask: "MSK1" header + W×H uint8. Returns flat Uint8Array (RGBA-expanded for compatibility). */
function decodeMaskBin(buf: Uint8Array): { mask: Uint8Array; W: number; H: number } {
  const magic = String.fromCharCode(buf[0], buf[1], buf[2], buf[3]);
  if (magic !== "MSK1") throw new Error(`bad mask magic: ${magic}`);
  const W = buf[5] << 8 | buf[4];   // u16LE
  const H = buf[7] << 8 | buf[6];   // u16LE
  const mask = buf.subarray(8);     // W×H uint8, no copy
  if (mask.length !== W * H) throw new Error(`mask size ${mask.length} !== ${W}×${H}`);
  return { mask, W, H };
}

/** Inpaint one page. */
export async function runInpaintPage(
  args: InpaintPageArgs, env: InpaintEnv, ctx: ExecutionContext,
): Promise<InpaintPageResult> {
  // 1. Fetch inputs in parallel — image, raw mask, scan msgpack.
  const [imgBytes, maskBinBytes, scanBytes] = await Promise.all([
    getBytes(env.R2, args.image_key),
    getBytes(env.R2, args.mask_key),
    getBytes(env.R2, args.scan_key),
  ]);

  const scanJson  = decodeMsgpack<ScanPageResult>(scanBytes);

  // Image may be JPEG (prepare output) or PNG (rerun).
  const imgRgba = isPng(imgBytes) ? await decodePng(imgBytes) : await decodeJpeg(imgBytes);
  const img     = new Uint8Array(imgRgba.data.buffer, imgRgba.data.byteOffset, imgRgba.data.byteLength);
  const W = imgRgba.width, H = imgRgba.height;

  // Raw mask: zero-decode, direct subarray view.
  const { mask, W: mW, H: mH } = decodeMaskBin(maskBinBytes);
  if (mW !== W || mH !== H) throw new Error(`shape mismatch: img=${W}x${H} mask=${mW}x${mH}`);

  // 2. Class-aware mask close (bridges intra-bubble gaps without bleeding into neighbours).
  const closedMask = closeMaskPerBlock(mask, W, H, scanJson.groups);

  // 3. Find bubble components.
  const bubbles = findBubbles(closedMask, W, H);
  const outKey  = r2keys.inpaint(args.chapter_id, args.page_index);

  if (bubbles.length === 0) {
    // No text → re-encode original as PNG passthrough.
    const passthrough = await encodePng(toClamped(img), W, H);
    await putBytes(env.R2, outKey, passthrough, "image/png");
    return { output_key: outKey, bubbles: 0, tiles_shape: [] };
  }

  // 4. Build padded tiles, bucketed to AOT input sizes.
  const tiles = bubbles.map(bb => buildTile(img, mask, W, H, bb));

  // 5. Shard fan-out → compose in-place. `img` is reused as composite to
  //    drop peak memory by 1 page-RGBA (~15 MB on a 1600×2400 page).
  await shardAndCompose(env, ctx, tiles, img, W, mask);

  // 6. Encode and write.
  const outPng = await encodePng(toClamped(img), W, H);
  await putBytes(env.R2, outKey, outPng, "image/png");

  return {
    output_key:  outKey,
    bubbles:     bubbles.length,
    tiles_shape: tiles.map(t => `${t.W}x${t.H}`),
  };
}
