// Smart-drop builder — converts dropped files into a single zip Blob.
//
// Accepts (in order of preference):
//   - a .zip / .cbz file  → returned as-is
//   - a folder            → traverse via DataTransferItem webkitGetAsEntry
//   - loose image files   → packed into a fresh zip in natural sort order
//
// Output zip uses `fflate` (stored, not deflated) — pipeline's unzip
// stage decodes either way and stored compression is ~free CPU.
//
// Natural sort: handles "page 9", "page 10" correctly (not lexical).

import { zip, type Zippable } from 'fflate'

const IMAGE_RE = /\.(?:jpe?g|png|webp|gif|avif|jxl|bmp)$/i

export interface BuildZipResult {
  blob:        Blob
  /** Page count inside the zip (0 means pass-through, undecoded). */
  page_count:  number
  /** True when the source was already a zip — we didn't repack. */
  passthrough: boolean
  /** Human-readable filename for download/share. */
  filename:    string
}

/** Entry point — call from `onDrop` / file input change handler. */
export async function buildZipFromDrop(
  dataTransfer: DataTransfer | null,
  fallbackFile: File | null,
): Promise<BuildZipResult> {
  // 1. Single archive file → pass through unchanged.
  const single = pickArchive(dataTransfer, fallbackFile)
  if (single) {
    return {
      blob:        single,
      page_count:  0,
      passthrough: true,
      filename:    single.name,
    }
  }

  // 2. Folder (only DataTransfer has entries; bare file input doesn't).
  const folderFiles = await readFolders(dataTransfer)
  if (folderFiles.length > 0) {
    return packImages(folderFiles, suggestFilename(folderFiles[0]?.path))
  }

  // 3. Loose images from DataTransfer or fallback file selection.
  const loose = await collectLooseImages(dataTransfer, fallbackFile)
  if (loose.length > 0) {
    return packImages(
      loose.map(file => ({ path: file.name, file })),
      'chapter.zip',
    )
  }

  throw new Error('Không có file nào phù hợp (zip/cbz, ảnh, hoặc folder).')
}


// ── Internals ───────────────────────────────────────────────────────

interface NamedFile {
  path: string
  file: File
}

function pickArchive(
  dt:       DataTransfer | null,
  fallback: File | null,
): File | null {
  if (dt) {
    for (let i = 0; i < dt.items.length; i++) {
      const it = dt.items[i]
      if (it?.kind !== 'file') continue
      const f = it.getAsFile()
      if (f && isArchive(f)) return f
    }
  }
  if (fallback && isArchive(fallback)) return fallback
  return null
}

function isArchive(f: File): boolean {
  return /\.(zip|cbz)$/i.test(f.name)
      || f.type === 'application/zip'
      || f.type === 'application/x-cbz'
}

async function readFolders(dt: DataTransfer | null): Promise<NamedFile[]> {
  if (!dt) return []
  const out: NamedFile[] = []
  const promises: Promise<void>[] = []
  for (let i = 0; i < dt.items.length; i++) {
    const it = dt.items[i]
    if (!it || it.kind !== 'file') continue
    const entry = (it as DataTransferItem & {
      webkitGetAsEntry?: () => FileSystemEntry | null
    }).webkitGetAsEntry?.()
    if (entry?.isDirectory) {
      promises.push(walkDirectory(entry as FileSystemDirectoryEntry, '', out))
    }
  }
  await Promise.all(promises)
  return out.filter(e => IMAGE_RE.test(e.path))
}

function walkDirectory(
  dir:    FileSystemDirectoryEntry,
  prefix: string,
  out:    NamedFile[],
): Promise<void> {
  return new Promise((resolve, reject) => {
    const reader = dir.createReader()
    const read = () => {
      reader.readEntries(async (entries) => {
        if (entries.length === 0) { resolve(); return }
        const pending: Promise<void>[] = []
        for (const ent of entries) {
          const path = prefix ? `${prefix}/${ent.name}` : ent.name
          if (ent.isFile) {
            pending.push(new Promise<void>((res, rej) => {
              (ent as FileSystemFileEntry).file(
                f => { out.push({ path, file: f }); res() },
                rej,
              )
            }))
          } else if (ent.isDirectory) {
            pending.push(walkDirectory(ent as FileSystemDirectoryEntry, path, out))
          }
        }
        await Promise.all(pending).catch(reject)
        read()    // readEntries chunks; keep reading until empty
      }, reject)
    }
    read()
  })
}

async function collectLooseImages(
  dt:       DataTransfer | null,
  fallback: File | null,
): Promise<File[]> {
  const out: File[] = []
  if (dt) {
    for (let i = 0; i < dt.files.length; i++) {
      const f = dt.files[i]
      if (f && IMAGE_RE.test(f.name)) out.push(f)
    }
  }
  if (out.length === 0 && fallback && IMAGE_RE.test(fallback.name)) {
    out.push(fallback)
  }
  return out
}

async function packImages(
  files:    NamedFile[],
  filename: string,
): Promise<BuildZipResult> {
  if (files.length === 0) throw new Error('Không có ảnh để pack.')

  files.sort((a, b) => naturalCompare(a.path, b.path))

  const entries: Zippable = {}
  for (let i = 0; i < files.length; i++) {
    const { file } = files[i]
    const ext = (file.name.match(/\.[^.]+$/)?.[0] ?? '').toLowerCase()
    const key = `${String(i + 1).padStart(4, '0')}${ext}`
    const buf = new Uint8Array(await file.arrayBuffer())
    // `[bytes, opts]` tuple → stored (no DEFLATE) — images are already
    // compressed; deflate would burn CPU for negligible savings.
    entries[key] = [buf, { level: 0 }]
  }

  return new Promise<BuildZipResult>((resolve, reject) => {
    zip(entries, (err, data) => {
      if (err) { reject(err); return }
      resolve({
        blob:        new Blob([data], { type: 'application/zip' }),
        page_count:  files.length,
        passthrough: false,
        filename:    filename.endsWith('.zip') ? filename : `${filename}.zip`,
      })
    })
  })
}

function suggestFilename(samplePath: string | undefined): string {
  if (!samplePath) return 'chapter.zip'
  const root = samplePath.split('/')[0] ?? 'chapter'
  return `${root}.zip`
}

function naturalCompare(a: string, b: string): number {
  return a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' })
}
