/**
 * Build-time helper for per-codec WASM bundling.
 *
 * Scans the worker's source tree for imports of
 * `shared/src/codec/{jpeg,png}-{dec,enc}` and emits exactly the .wasm
 * blobs + import preamble those modules need. No manual codec lists in
 * each worker's build.mjs — the imports are the contract.
 *
 * Returns `{ needs, copy(), preamble(), verify(), report() }`.
 */
import { cpSync, existsSync, statSync, readdirSync, readFileSync } from "node:fs";
import { resolve, join } from "node:path";

const CODECS = {
  "jpeg-dec": { wasm: "@jsquash/jpeg/codec/dec/mozjpeg_dec.wasm",  file: "mozjpeg_dec.wasm",   global: "__JPEG_DEC_WASM__" },
  "jpeg-enc": { wasm: "@jsquash/jpeg/codec/enc/mozjpeg_enc.wasm",  file: "mozjpeg_enc.wasm",   global: "__JPEG_ENC_WASM__" },
  "png-dec":  { wasm: "@jsquash/png/codec/pkg/squoosh_png_bg.wasm", file: "squoosh_png_bg.wasm", global: "__PNG_WASM__" },
  "png-enc":  { wasm: "@jsquash/png/codec/pkg/squoosh_png_bg.wasm", file: "squoosh_png_bg.wasm", global: "__PNG_WASM__" },
};

/** Walk a directory tree returning every .ts file path. */
function walkTs(root) {
  const out = [];
  for (const d of readdirSync(root, { withFileTypes: true })) {
    const p = join(root, d.name);
    if (d.isDirectory()) out.push(...walkTs(p));
    else if (d.name.endsWith(".ts")) out.push(p);
  }
  return out;
}

/** Detect codec modules imported by `srcDir`. Returns a sorted Set. */
export function detectCodecsUsed(srcDir) {
  const re = /from\s+["'][^"']*shared\/src\/codec\/(jpeg-dec|jpeg-enc|png-dec|png-enc)["']/g;
  const used = new Set();
  for (const f of walkTs(srcDir)) {
    const text = readFileSync(f, "utf8");
    for (const m of text.matchAll(re)) used.add(m[1]);
  }
  return used;
}

/** Bundle codec WASM + emit preamble for a worker.
 *
 *   const codec = bundleCodecs({ srcDir, nmDir, outDir });
 *   const code  = codec.preamble + bundledCode;
 *   codec.verify();
 */
export function bundleCodecs({ srcDir, nmDir, outDir }) {
  const needs = detectCodecsUsed(srcDir);
  if (needs.size === 0) return emptyBundle();

  // Copy WASM (dedup by filename: png-dec + png-enc share the same blob).
  const filesCopied = new Set();
  for (const name of needs) {
    const c = CODECS[name];
    if (filesCopied.has(c.file)) continue;
    const src = resolve(nmDir, c.wasm);
    if (!existsSync(src)) {
      throw new Error(`[codec] missing ${src}. Run \`npm i\` in the worker dir.`);
    }
    cpSync(src, resolve(outDir, c.file));
    filesCopied.add(c.file);
  }

  // Emit import + globalThis assignment, one per unique .wasm file.
  const globalsEmitted = new Set();
  const importLines = [];
  const assignLines = [];
  for (const name of needs) {
    const c = CODECS[name];
    if (globalsEmitted.has(c.global)) continue;
    globalsEmitted.add(c.global);
    importLines.push(`import ${c.global} from "./${c.file}";`);
    assignLines.push(`globalThis.${c.global} = ${c.global};`);
  }
  const preamble = [...importLines, ...assignLines, ""].join("\n");

  return {
    needs: [...needs].sort(),
    files: [...filesCopied].sort(),
    preamble,
    verify() {
      for (const f of filesCopied) {
        const p = resolve(outDir, f);
        if (!existsSync(p) || statSync(p).size === 0) {
          throw new Error(`[codec] verify: missing or empty ${f}`);
        }
      }
    },
    report() {
      return [...filesCopied].sort().map(f =>
        `  ${f.padEnd(28)} ${(statSync(resolve(outDir, f)).size / 1024).toFixed(1)} KiB`
      ).join("\n");
    },
  };
}

function emptyBundle() {
  return {
    needs: [], files: [], preamble: "",
    verify() {}, report() { return "  (no codecs)"; },
  };
}
