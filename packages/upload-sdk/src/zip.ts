// Pack a list of (filename, bytes) pairs into a single ZIP blob.
//
// Store mode (level 0) — JPEG/PNG/WebP are already compressed; running
// deflate on top wastes CPU and adds latency. Engine accepts ZIP via
// /chapters/upload-finalize and natural-sorts filenames into reading
// order, so the only contract this module enforces is the numeric-
// padded filename scheme.
//
// Filenames: `0001.png`, `0002.jpg`, … padded to 4 digits.
//
// Synchronous `zipSync` (not the worker-based `zip` callback API):
// MV3 service workers lack the Worker constructor, and store-mode is
// byte-copying anyway — nothing to parallelise.

import { zipSync, type Zippable } from 'fflate'

const ALLOWED_EXT = ['png', 'jpg', 'jpeg', 'webp', 'gif'] as const
type AllowedExt = (typeof ALLOWED_EXT)[number]

export interface PackPage {
  /** Original filename or URL — used only to recover the file extension. */
  source: string
  /** Page bytes. Already-compressed image format. */
  bytes:  Uint8Array
}

export function packPagesToZip(pages: PackPage[]): Blob {
  if (pages.length === 0) {
    throw new Error('Không có trang nào để upload.')
  }
  const files: Zippable = {}
  pages.forEach((p, i) => {
    const ext  = pickExt(p.source)
    const name = `${String(i + 1).padStart(4, '0')}.${ext}`
    // Tuple form: [data, options]. level: 0 = STORE.
    files[name] = [p.bytes, { level: 0 }]
  })
  const buf = zipSync(files, { level: 0 })
  return new Blob([new Uint8Array(buf)], { type: 'application/zip' })
}

function pickExt(source: string): AllowedExt {
  // Strip query string + fragment first — MangaDex/CDN URLs often end
  // in `…?token=…` which would otherwise eat the real extension.
  const clean = source.split(/[?#]/)[0] ?? source
  const tail  = clean.split('.').pop()?.toLowerCase() ?? ''
  if (tail === 'jpeg') return 'jpg'   // canonicalise
  return (ALLOWED_EXT as readonly string[]).includes(tail)
    ? (tail as AllowedExt)
    : 'jpg'
}
