# RFC-004: Bunle Archives for Prepared and Render Artifacts

## Status

Proposed (v3)

## Scope

Storage representation for long-lived chapter image artifacts in SaaS / worker
mode.

This RFC does **not** change the logical pipeline boundary accepted in
RFC-001:

```text
RawSource -> prepare -> PreparedChapter -> page-local pipeline -> RenderedChapter
```

It changes how prepared and rendered page images are persisted after each
stage. It is a storage cutover, not a compatibility migration. Once
implemented, the active runtime must use Bunle archives as the canonical
artifact representation.

## Non-goals

- This RFC does not redefine the Bunle format. Bunle is consumed as a Python
  package (`bunle`, PyO3 binding from `nghyane/bunle@<commit>`). Format
  evolution belongs in the Bunle repo.
- No user accounts, billing, or retention plans.
- R2 is not required immediately; local archive storage is valid.
- No live per-page render preview during processing.
- No final mask editing UX.
- No prepared/render history retention.
- No backward compatibility with the old loose-page artifact layout after
  cutover.

## Problem

Current filesystem layout persists one file per page:

```text
chapter/
  pages/0000.png
  masks/0000.npz
  render/0000.png
  scan.npz
```

Acceptable for local development, poor long-term SaaS layout:

- R2/object storage would contain too many objects per chapter.
- UI preview needs chapter-level before/after access.
- Workers need a clean cache boundary: download one input, process in `/tmp`,
  upload one output.
- Raw source should not be retained after prepare.
- `scan.npz` duplicates geometry that should be queryable in DB.

## Decision

Persist prepared and rendered pages as content-addressed Bunle archives:

```text
p/<project_id>/c/<chapter_id>/prepared-<rev>.bnl
p/<project_id>/c/<chapter_id>/render-<rev>.bnl
```

`<rev>` is `sha256(archive_bytes)[:8]`. Active key is stored in DB. Replacing
an archive is write-new + flip-pointer + delete-old. Old object is deleted
only after the pointer flip succeeds.

Re-importing or re-preparing a chapter writes a new prepared archive and must
reset downstream state derived from old prepared pixels (OCR, geometry,
translations, masks, render).

Rerender writes a new render archive. A failed rerender leaves the current
`render_key` intact; DB state moves to `failed` or `stale`.

Raw source files are temporary import inputs only:

```text
source URL/folder -> tmp/raw -> prepare -> prepared-<rev>.bnl -> clean tmp
```

Loose page files are allowed only as worker temp materialization or local
debug output. They are never canonical storage and never referenced by DB.

## Bunle integration

Typoon imports the `bunle` Python module directly. No subprocess, no JSON
parsing, no CLI version pinning concerns.

```python
import bunle

# pack
bunle.pack_dir(src_dir, out_file)            # raises on v1 limit overflow

# random-access read
with bunle.Reader(archive) as r:
    n = r.page_count
    for i in range(n):
        png_or_webp_bytes = r.page(i)        # zero-copy from mmap
        meta = r.info(i)                     # {width, height, format, ...}

# bulk extract (only when a stage really needs file paths)
bunle.unpack(archive, dst_dir)
```

Typoon owns encoding decisions; Bunle is the container.

### Encoding policy

Typoon encodes pages before `pack_dir`. Bunle stores them byte-identical.

```text
prepared pages -> WebP lossless    (Pillow / cwebp -lossless)
render pages   -> WebP q=95        (configurable per project)
mask pages     -> WebP lossless, grayscale  (deferred)
```

Per-archive constraint (Typoon-side): all pages in a single archive use the
same encoder settings. Keeps validation and UI preview simple.

Prepared pages must be lossless. All downstream geometry, OCR boxes, and
render input are defined against prepared pixels.

Render pages may be lossy (WebP q=95) for product delivery, or lossless when
an export-grade render is requested. Render quality is a chapter-level
setting, not a format change.

PNG is not used in canonical archives. PNG appears only inside `debug-runs/`.

### Archive limits

Bunle v1 enforces:

- `page_count` ≤ 65,535
- per-page encoded size ≤ ~4 GB
- archive total size ≤ ~4 GB

`bunle.pack_dir` raises on overflow. Typoon does not need to re-check.

## Runtime cutover

After this RFC ships, runtime stages load prepared pages from the active
prepared archive plus DB metadata only. The old chapter filesystem layout
is not a runtime input.

Forbidden after cutover:

- loading `PreparedChapter` by scanning persistent `pages/` directories
- loading render output by scanning persistent `render/` directories
- treating `scan.npz` as canonical geometry
- falling back from missing `.bnl` archives to loose page folders
- retaining raw source files after prepare completes
- storing downstream coordinates against anything other than prepared
  archive pages

## Storage keys

Local dev and R2 production share the same logical key shape:

```text
p/<project_id>/c/<chapter_id>/prepared-<rev>.bnl
p/<project_id>/c/<chapter_id>/render-<rev>.bnl
```

Only the keys named in `chapters.prepared_key` / `chapters.render_key` are
canonical. Any other revision present in storage is garbage.

## Database shape

```sql
ALTER TABLE chapters ADD COLUMN prepared_key   TEXT;
ALTER TABLE chapters ADD COLUMN render_key     TEXT;
ALTER TABLE chapters ADD COLUMN render_state   TEXT NOT NULL DEFAULT 'none';
ALTER TABLE chapters ADD COLUMN render_job_id  TEXT;
ALTER TABLE chapters ADD COLUMN page_count     INTEGER NOT NULL DEFAULT 0;
```

`page_count` is denormalized for list views. The archive index is the source
of truth for per-page dimensions; query via `bunle.Reader(...).info(i)` when
needed. No `page_metadata` table in MVP.

No `*_sha256` columns. Integrity is the format's responsibility (Bunle has
structural validation and can add CRC trailer in v1.1 when required); R2
stores ETags natively. Adding application-level checksums is overhead.

Render state machine:

```text
none       # no successful render exists
rendering  # render job is running
rendered   # render archive matches current DB/editor state
stale      # translation, geometry, font, or layout changed since last render
failed     # latest render attempt failed; retry is allowed
```

Concurrency rule: a render worker claims a chapter via CAS:

```sql
UPDATE chapters
   SET render_state = 'rendering', render_job_id = :new_job
 WHERE id = :cid AND render_state <> 'rendering';
```

Only the same `render_job_id` may transition out of `rendering` and update
`render_key`. Two workers cannot both win.

Bubble geometry moves into structured DB rows:

```sql
-- Direction, not final DDL.
ALTER TABLE bubbles ADD COLUMN polygon_json   TEXT;
ALTER TABLE bubbles ADD COLUMN erase_box_json TEXT;
ALTER TABLE bubbles ADD COLUMN fit_box_json   TEXT;
ALTER TABLE bubbles ADD COLUMN text_box_json  TEXT;
```

`scan.npz` is not part of the active runtime after cutover. `.npz` files are
allowed only inside worker temp/debug outputs and must not be required by
later stages.

## Worker flows

### Prepare / import

```python
1. Pull source -> tmp/raw/
2. Run prepare -> tmp/prepared/*.webp (lossless)
3. bunle.pack_dir(tmp/prepared, tmp/prepared.bnl)
4. rev = sha256(tmp/prepared.bnl)[:8]
5. key = f"p/{pid}/c/{cid}/prepared-{rev}.bnl"
6. await store.put_file(key, tmp/prepared.bnl)
7. UPDATE chapters
        SET prepared_key=:key, page_count=:n,
            render_state='none', render_key=NULL
      WHERE id=:cid
8. await store.delete(old_prepared_key)            # if any
9. await store.delete(old_render_key)              # if any
10. rm -rf tmp
```

### Scan

```python
1. await store.get_file(prepared_key, tmp/prepared.bnl)
2. with bunle.Reader(tmp/prepared.bnl) as r:
       for i in range(r.page_count):
           img = decode(r.page(i))                 # in-memory; no extract
           run_ocr_and_geometry(img) -> DB
3. if previous render existed:
       UPDATE chapters SET render_state='stale' WHERE id=:cid
```

No `tmp/prepared/` directory. Random-access read straight from mmapped
archive.

### Render

```python
1. job = uuid4()
   rows = UPDATE chapters
            SET render_state='rendering', render_job_id=:job
          WHERE id=:cid AND render_state <> 'rendering'
   if rows == 0: abort (another worker holds the lock)

2. await store.get_file(prepared_key, tmp/prepared.bnl)
3. with bunle.Reader(tmp/prepared.bnl) as r:
       for i in range(r.page_count):
           src = decode(r.page(i))
           rendered = render_page(src, db_geometry, db_translations)
           encode_webp(rendered, tmp/render/{i:03d}.webp, quality=95)

4. bunle.pack_dir(tmp/render, tmp/render.bnl)
5. rev = sha256(tmp/render.bnl)[:8]
6. key = f"p/{pid}/c/{cid}/render-{rev}.bnl"
7. await store.put_file(key, tmp/render.bnl)

8. UPDATE chapters
        SET render_key=:key, render_state='rendered'
      WHERE id=:cid AND render_job_id=:job
9. await store.delete(old_render_key)              # if any

   On failure (any step):
   UPDATE chapters SET render_state='failed' WHERE id=:cid AND render_job_id=:job
   # render_key untouched
```

## ArtifactStore interface

```python
class ArtifactStore(Protocol):
    async def put_file(self, key: str, path: Path) -> None: ...
    async def get_file(self, key: str, dest: Path) -> None: ...
    async def delete(self, key: str) -> None: ...                # no-op on missing
    async def head(self, key: str) -> ArtifactHead | None: ...
    async def url(self, key: str, expires: int = 3600) -> str: ...
```

Implementations:

```text
LocalArtifactStore  -> ~/.typoon/artifacts/<key>
R2ArtifactStore     -> r2://<bucket>/<key>
```

Local store rules:

- `put_file`: write to `<key>.tmp`, fsync, `os.replace` to `<key>` (atomic).
- `url`: returns a signed URL via the local dev API server, not `file://`.
- `delete`: best-effort; never raises on missing key.

R2 store rules:

- `put_file`: single PUT. Keys are content-addressed; we never overwrite.
- `url`: presigned GET URL.

`Content-Type: application/octet-stream` for now (`application/x-bunle`
optional later).

## UI preview model

Before/after preview opens two archives via the JS Bunle SDK:

```ts
const prepared = await Bunle.open(preparedUrl)
const render   = await Bunle.open(renderUrl)
const before   = await prepared.blob(i)
const after    = await render.blob(i)
```

Bubble overlays are drawn from DB geometry on top of prepared pages.

For MVP, no live per-page render preview. UI shows:

- prepared preview immediately (key already in DB)
- stage/progress from DB/SSE
- final render archive once `render_key` flips

If live render previews are required later, use a separate ephemeral
namespace **outside** the canonical artifact key tree:

```text
ephemeral/p/<pid>/c/<cid>/<job_id>/<page>.webp
```

Short TTL, never named by `render_key`, separate from `ArtifactStore` for
canonical artifacts.

## Masks policy

Mask persistence is deferred.

For MVP/editor-light workflows:

- editable geometry in DB
- erase masks reconstructed from geometry at render time
- pixel masks live in worker temp only

If pro brush editing arrives, persist as `masks-<rev>.bnl` (WebP lossless
grayscale). `.npz` is not used for canonical mask storage. Same
revision/atomic-replace rules as render.

## Local development

Local dev uses the same archive representation:

```text
~/.typoon/artifacts/p/<pid>/c/<cid>/prepared-<rev>.bnl
~/.typoon/artifacts/p/<pid>/c/<cid>/render-<rev>.bnl
```

No second local-only artifact layout, no migration when R2 lands.

## Cutover from current layout

Existing local development artifacts may be discarded for a clean SaaS
launch if no user data must be preserved.

If selected data must be preserved, a one-time offline conversion script
may:

1. Re-encode `pages/*.png` to WebP lossless, `bunle.pack_dir` -> archive.
2. Re-encode `render/*.png` to WebP q=95, `bunle.pack_dir` -> archive.
3. Save keys in `chapters`.
4. Move geometry from `scan.npz` into DB once geometry migration lands.
5. Delete loose page folders after verification.

The conversion script must not be called by production/runtime loaders, and
runtime code must not fall back to the old layout.

## Acceptance criteria

- Raw import data is deleted after the prepared archive is uploaded.
- `prepared-<rev>.bnl` is the only canonical prepared image artifact.
- `render-<rev>.bnl` is the only canonical rendered image artifact.
- Active artifacts are content-addressed; only the keys named in
  `chapters.prepared_key` / `chapters.render_key` are canonical.
- Replacement is atomic from a reader's perspective: write-new, flip-pointer,
  delete-old.
- A failed rerender does not delete or invalidate the current `render_key`.
- Two render workers cannot both update `render_key` for the same chapter.
- Active runtime does not load canonical artifacts from persistent `pages/`
  or `render/` directories.
- No compatibility fallback reads the old loose-page layout.
- Loose page files exist only inside worker temp directories or
  `debug-runs/*`.
- Prepared and render preview are served from their respective archives via
  signed URLs.
- DB `page_count` matches archive page count after upload.
- DB pointers (`prepared_key`, `render_key`) are updated only after upload
  succeeds.
- Scan/render workers consume prepared pages via `bunle.Reader` random
  access; they do not extract the entire archive to disk first.
- A worker can reconstruct all render input from the active prepared archive
  plus DB state.
- Worker temp directories can be deleted after each run without losing
  canonical artifacts.
- Local and R2 artifact stores share the same logical keys.
- Live render progress, if implemented, lives in a separate ephemeral
  namespace and is never named by `render_key`.
