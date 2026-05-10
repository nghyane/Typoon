# RFC-004: Bunle Archives for Prepared and Render Artifacts

## Status

Proposed (v4)

## Scope

Storage layout for chapter image artifacts produced by the queue worker.
The pipeline goal is **import → prepared archive → scan → translate →
render archive on store**. Per-chapter URL streaming, signed URLs, ETags,
content-addressing, and live preview are app-layer concerns and explicitly
out of scope here.

This RFC does not change the logical pipeline boundary accepted in RFC-001:

```text
RawSource -> prepare -> PreparedChapter -> page-local pipeline -> RenderedChapter
```

It is a storage cutover, not a compatibility migration. Once implemented,
the active runtime must use Bunle archives as the canonical artifact
representation.

## Non-goals

- App / UI concerns: signed URLs, ETags, mid-stream consistency, live render
  progress, before/after preview routing. These are added at the app layer
  when needed.
- Content-addressed key strategy. Worker model uses a single deterministic
  key per chapter; rewriting in place is safe because no concurrent reader
  consumes the archive while the writer is active (state machine gates it).
- Bunle format evolution. Bunle is consumed as-is from `nghyane/bunle@<sha>`
  through the Python module.
- User accounts, billing, retention.
- R2 immediately. Local storage is valid until SaaS lands.
- Mask editing UX.
- History retention.
- Backward compatibility with the loose-page layout after cutover.

## Problem

Current filesystem layout persists one file per page:

```text
chapter/
  pages/0000.png
  masks/0000.npz
  render/0000.png
  scan.npz
```

Acceptable locally, poor for a SaaS worker model:

- Object storage would carry too many objects per chapter.
- Workers want a clean cache boundary: download one input, process in `/tmp`,
  upload one output.
- Raw source should not be retained after prepare.
- `scan.npz` duplicates geometry that can live in DB.

## Decision

Persist prepared and render pages as Bunle archives at deterministic keys:

```text
p/<project_id>/c/<chapter_id>/prepared.bnl
p/<project_id>/c/<chapter_id>/render.bnl
```

A single key per (chapter, kind). Prepare overwrites `prepared.bnl`;
render overwrites `render.bnl`. The DB state machine guarantees a reader
never observes an in-flight write:

- `chapters.render_state` is the only signal a stage uses to decide whether
  the render archive is ready.
- A worker claims its slot via CAS on `render_state` + `render_job_id`
  before producing or replacing an archive.

Loose page files remain only inside worker temp directories or
`debug-runs/*`. They are not canonical and never named in DB.

## Bunle integration

Typoon imports the `bunle` Python module directly:

```python
import bunle

bunle.pack_dir(src_dir, out_file)        # raises on v1 limit overflow
bunle.validate(out_file)                 # structural check
with bunle.Reader(archive) as r:
    n = r.page_count
    for i in range(n):
        webp_bytes = r.page(i)           # zero-copy slice from mmap
```

Typoon owns encoding decisions; Bunle is the container.

### Encoding policy

Typoon encodes pages before `pack_dir`. Bunle stores them byte-identical.

```text
prepared pages -> WebP q=92 method=4   (Pillow save)
render pages   -> WebP q=92 method=4   (configurable per project)
```

Per-archive constraint (Typoon-side): all pages in a single archive use the
same encoder settings.

The canonical-pixel contract for prepared pages is **consistency** (every
stage decodes the same bytes for page i), not bit-level fidelity. Lossy
WebP q=92 round-trips through downstream YOLO / OCR / AOT inpaint without
measurable accuracy loss while saving ~10× on encode time and ~4× on
archive size vs lossless. An export-grade lossless render is still
available as a per-export option when bit-fidelity matters; the canonical
runtime archive is lossy.

PNG is not used in canonical archives. PNG appears only inside `debug-runs/`.

### Archive limits

Bunle v1 enforces:

- `page_count` ≤ 65,535
- per-page encoded size ≤ ~4 GB
- archive total size ≤ ~4 GB

`bunle.pack_dir` raises on overflow. Typoon does not re-check.

## Runtime cutover

After this RFC ships, runtime stages load prepared pages from `prepared.bnl`
plus DB metadata only.

Forbidden after cutover:

- loading prepared pages by scanning persistent `pages/` directories
- loading render output by scanning persistent `render/` directories
- treating `scan.npz` as canonical geometry
- retaining raw source files after prepare completes
- storing downstream coordinates against anything other than prepared
  archive pages

## Database shape

```sql
ALTER TABLE chapters ADD COLUMN render_state   TEXT NOT NULL DEFAULT 'none';
ALTER TABLE chapters ADD COLUMN render_job_id  TEXT;
ALTER TABLE chapters ADD COLUMN page_count     INTEGER NOT NULL DEFAULT 0;
```

No `prepared_key` / `render_key` columns: keys are deterministic. No
`*_sha256` columns: integrity is the format's responsibility (Bunle has
structural validation).

`page_count > 0` ⇒ a prepared archive exists for the chapter. Per-page
dimensions come from the archive index via `bunle.Reader.info(i)`.

Render state machine:

```text
none       # no successful render exists
rendering  # render job is running
rendered   # render archive is current
stale      # translation, geometry, font, or layout changed since last render
failed     # latest render attempt failed; retry is allowed
```

CAS render claim:

```sql
UPDATE chapters
   SET render_state = 'rendering', render_job_id = :new_job
 WHERE id = :cid AND render_state <> 'rendering';
```

Only the same `render_job_id` may transition out of `rendering`.

Bubble geometry moves into structured DB rows in a follow-up; `scan.npz`
remains the geometry source for now and lives in worker temp/debug only —
never required by later stages once geometry is in DB.

## ArtifactStore interface

```python
class ArtifactStore(Protocol):
    async def put_file(self, key: str, src: Path) -> None: ...
    async def get_file(self, key: str, dest: Path) -> None: ...
```

Two operations. Worker writes archives; the next stage reads them. Anything
beyond that — listing, deletion, signed URLs, ETags — belongs to the app
layer and is added there when needed.

`LocalArtifactStore` writes to disk under a configurable root.
`R2ArtifactStore` lands when SaaS does.

## Worker flows

### Prepare / import

```python
1. Pull source -> tmp/raw/.
2. Run prepare; emit lossless WebP pages into tmp/prepared/.
3. bunle.pack_dir(tmp/prepared, tmp/prepared.bnl)
4. key = prepared_key(pid, cid)
5. await store.put_file(key, tmp/prepared.bnl)
6. await db.set_prepared_done(cid, page_count=n)   # also resets render_state
7. rm -rf tmp.
```

### Scan

```python
1. await store.get_file(prepared_key(pid, cid), tmp/prepared.bnl)
2. with bunle.Reader(tmp/prepared.bnl) as r:
       for i in range(r.page_count):
           img = decode(r.page(i))
           run_ocr_and_geometry(img) -> DB
3. await db.mark_render_stale(cid)   # only no-op when no prior render
```

### Render

```python
1. job = uuid4().hex
   ok  = await db.claim_render_job(cid, job)
   if not ok: abort (another worker holds the slot).

2. await store.get_file(prepared_key(pid, cid), tmp/prepared.bnl)
3. with bunle.Reader(tmp/prepared.bnl) as r:
       for i in range(r.page_count):
           src = decode(r.page(i))
           rendered = render_page(src, db_geometry, db_translations)
           encode_webp(rendered, tmp/render/{i:03d}.webp, quality=95)

4. bunle.pack_dir(tmp/render, tmp/render.bnl)
5. await store.put_file(render_key(pid, cid), tmp/render.bnl)

6. await db.finish_render_job(cid, job)

   On failure:
   await db.fail_render_job(cid, job)
   # render archive is left at its last successful state if any; the
   # state machine prevents readers from consuming a half-written one.
```

## Local development

Local dev uses the same archive representation:

```text
~/.typoon/artifacts/p/<pid>/c/<cid>/prepared.bnl
~/.typoon/artifacts/p/<pid>/c/<cid>/render.bnl
```

No second local-only artifact layout, no migration when R2 lands.

## Cutover from current layout

Existing local development artifacts may be discarded for a clean SaaS
launch if no user data must be preserved.

If selected data must be preserved, a one-time offline conversion script
may:

1. Re-encode `pages/*.png` to WebP lossless, `bunle.pack_dir` -> archive.
2. Re-encode `render/*.png` to WebP q=95, `bunle.pack_dir` -> archive.
3. `set_prepared_done(cid, n)` after upload.
4. Move geometry from `scan.npz` into DB once geometry migration lands.
5. Delete loose page folders after verification.

The conversion script must not be called by production/runtime loaders, and
runtime code must not fall back to the old layout.

## Acceptance criteria

- Raw import data is deleted after the prepared archive is uploaded.
- `prepared.bnl` is the only canonical prepared image artifact.
- `render.bnl` is the only canonical rendered image artifact.
- Active runtime does not load canonical artifacts from persistent `pages/`
  or `render/` directories.
- No compatibility fallback reads the old loose-page layout.
- Loose page files exist only inside worker temp directories or
  `debug-runs/*`.
- DB `page_count` matches archive page count after upload.
- `render_state` is the only signal stages use to decide render readiness.
- Two render workers cannot both transition out of `rendering` for the same
  chapter — CAS on `render_job_id` enforces single-winner.
- A failed render leaves DB at `failed`; the next claim transitions out of
  `failed`/`stale`/`rendered` cleanly.
- Scan/render workers consume prepared pages via `bunle.Reader` random
  access; they do not extract the entire archive to disk first.
- Local and R2 artifact stores share the same logical keys.
