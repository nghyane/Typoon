/**
 * Image-codec layer for Cloudflare Workers.
 *
 * Each codec is its own module. Workers import only what they need;
 * esbuild dead-code-eliminates the rest. The build helper inspects
 * those imports to decide which `.wasm` blobs to copy into `.worker/`
 * and what preamble globals to emit.
 *
 * Contract:
 *
 *   // worker entry
 *   import { decodeJpeg } from "../../shared/src/codec/jpeg";
 *   import { encodePng }  from "../../shared/src/codec/png";
 *
 *   const img = await decodeJpeg(bytes);
 *   const out = await encodePng(rgba, w, h);
 *
 * Each module:
 *   - declares the `__X_WASM__` global it expects
 *   - kicks off the @jsquash `init()` once at module load
 *   - throws fast at import time if the worker's bundler didn't ship
 *     the matching .wasm
 *
 * Magic-byte dispatch is the caller's job. The only worker that needs
 * to handle "either JPEG or PNG" is render-archive; it imports both
 * decoders and switches explicitly.
 */
export type { RgbaImage } from "./types";
