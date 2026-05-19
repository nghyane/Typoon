/** PNG encode (Squoosh OxiPNG WASM via @jsquash/png). */
import encode, { init } from "@jsquash/png/encode";
import { requireWasm } from "./types";

let ready: Promise<void> | null = null;
function ensureReady() {
  if (!ready) ready = init(requireWasm("__PNG_WASM__"));
  return ready;
}

export async function encodePng(
  rgba: Uint8ClampedArray, width: number, height: number,
): Promise<Uint8Array> {
  await ensureReady();
  const out = await encode({ data: rgba, width, height } as any);
  return new Uint8Array(out);
}
