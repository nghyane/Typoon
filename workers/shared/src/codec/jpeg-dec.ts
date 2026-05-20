/** JPEG decode (mozjpeg WASM via @jsquash/jpeg). */
import decode, { init } from "@jsquash/jpeg/decode";
import { requireWasm, type RgbaImage } from "./types";

let ready: Promise<void> | null = null;
function ensureReady() {
  if (!ready) ready = init(requireWasm("__JPEG_DEC_WASM__"));
  return ready;
}

export async function decodeJpeg(buf: ArrayBuffer | Uint8Array): Promise<RgbaImage> {
  await ensureReady();
  const ab = buf instanceof Uint8Array
    ? buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength)
    : buf;
  return (await decode(ab as ArrayBuffer)) as RgbaImage;
}
