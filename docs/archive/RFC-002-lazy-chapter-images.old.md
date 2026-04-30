# RFC-002: Lazy Chapter Images After Strip Scan

## Status

Implemented

## Context

Manhwa/manhua sources often expose a continuous vertical strip as many arbitrary
image chunks. A speech bubble can be split at a source page boundary. Scanning
each source page independently makes the detector see partial bubbles and causes
missed/incorrect OCR.

The previous workaround scanned source pages, then re-scanned small boundary
zones. That was fragile: it still depended on partial detections and had complex
merge/dedup logic.

The first correct fix was to stitch all source pages into one strip, scan the
strip once, then re-cut at bubble-safe positions. This fixes detection, but it
introduced a memory lifetime problem: the chapter-sized stitched buffer was kept
through translation and render even though those stages only need one page at a
time.

For long chapters (for example 120 source pages), translation can take minutes.
Keeping a full stitched strip in RAM for that entire time is unnecessary.

## Decision

Keep the correct scan model, but decouple scan memory from page image lifetime:

1. Load source images and stitch into one normalized strip only for scan.
2. Scan the full strip. The detector already tiles long images internally.
3. Compute logical page ranges that avoid cutting through detected bubbles.
4. Convert detected groups into `Page`/`Bubble` domain objects using logical page
   coordinates.
5. Release the stitched strip immediately after preprocess.
6. Return a lazy page provider (`LazyChapterImages`) that stores only:
   - original `ChapterSource`
   - normalized source heights
   - target width
   - logical page ranges
7. During translation/look-at/render, load and crop only the requested logical
   page on demand.

## Architecture

### Eager scan buffer

`ChapterImages` remains the eager owner of a stitched buffer. It is used inside
`Engine.preprocess` as a short-lived scan buffer.

### Lazy runtime provider

`LazyChapterImages` implements the page-provider surface used by downstream
stages:

- `page_count()`
- `page(index)`
- `page_height(index)`
- `page_offset(index)`
- `alive`
- `free()`

`page(index)` reconstructs a logical page by loading only overlapping source
images, normalizing their width exactly like stitching, cropping the relevant
global Y range, and concatenating the pieces when a logical page spans multiple
source images.

## Memory Profile

Before:

- Scan: full chapter strip in RAM
- Translate: full chapter strip still in RAM
- Render: full chapter strip still in RAM

After:

- Scan: full chapter strip in RAM temporarily
- Translate: no resident page pixels except one on-demand look-at image
- Render: one logical page image at a time

## Performance

The scan remains correct because bubbles are detected on the continuous strip.
The detector already uses vertical tiling, so scanning a long strip does not
require one model input for the full height.

Lazy page reconstruction adds small I/O/resize work when translation requests a
page image or render processes a page. This is acceptable because those stages
operate per page and no longer hold chapter-sized buffers idle.

## Risks

- Source images must remain available after preprocess. Current pipeline keeps
  the `ChapterSource` on the job, so this is satisfied.
- Lazy reconstruction must match stitch normalization. `LazyChapterImages` uses
  the target width and normalized heights produced during scan to avoid drift.
- If a source adapter is expensive to reload, repeated look-at calls can reload
  the same page. A small LRU cache can be added later if profiling shows need.

## Verification

- Full test suite: `python -m pytest tests/ -q`

