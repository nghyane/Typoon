/** JPEG encode (mozjpeg WASM via @jsquash/jpeg). */
import encode, { init } from "@jsquash/jpeg/encode";
import { requireWasm } from "./types";

let ready: Promise<void> | null = null;
function ensureReady() {
  if (!ready) ready = init(requireWasm("__JPEG_ENC_WASM__"));
  return ready;
}

export async function encodeJpeg(
  rgba: Uint8ClampedArray, width: number, height: number, quality = 92,
): Promise<Uint8Array> {
  await ensureReady();
  const out = await encode({ data: rgba, width, height } as any, {
    quality, optimize_coding: true,
  } as any);
  return new Uint8Array(out);
}
