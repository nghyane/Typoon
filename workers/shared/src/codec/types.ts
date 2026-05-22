/** Raw RGBA image, the universal pixel buffer shape across codecs. */
export interface RgbaImage {
  data:   Uint8ClampedArray;
  width:  number;
  height: number;
}

/** Look up a CompiledWasm global the build preamble injected. Throws
 *  immediately if missing — that's a build-config bug, not a runtime
 *  one, and we want it surfaced at module load, not at first call. */
export function requireWasm(name: string): WebAssembly.Module {
  const w = (globalThis as any)[name] as WebAssembly.Module | undefined;
  if (!w) {
    throw new Error(
      `[codec] missing global ${name}. Each worker that uses this codec must `
      + `import the .wasm asset and assign it to globalThis (see workers/inpaint/src/index.ts).`,
    );
  }
  return w;
}
