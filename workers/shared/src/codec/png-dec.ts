/** PNG decode (Squoosh OxiPNG WASM via @jsquash/png). */
import decode, { init } from "@jsquash/png/decode";
import { requireWasm, type RgbaImage } from "./types";

let ready: Promise<void> | null = null;
function ensureReady() {
  if (!ready) ready = init(requireWasm("__PNG_WASM__")).then(() => {});
  return ready;
}

export async function decodePng(buf: ArrayBuffer | Uint8Array): Promise<RgbaImage> {
  await ensureReady();
  const ab = buf instanceof Uint8Array
    ? buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength)
    : buf;
  return (await decode(ab as ArrayBuffer)) as RgbaImage;
}

/** Magic-byte sniff — works without WASM init. */
export function isPng(buf: ArrayBuffer | Uint8Array): boolean {
  const u8 = buf instanceof Uint8Array
    ? buf
    : new Uint8Array(buf as ArrayBuffer, 0, Math.min(4, (buf as ArrayBuffer).byteLength));
  return u8[0] === 0x89 && u8[1] === 0x50 && u8[2] === 0x4e && u8[3] === 0x47;
}
