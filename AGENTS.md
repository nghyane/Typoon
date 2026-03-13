# ComicScan — Agent Instructions

Manga/manhwa translation tool in Rust. Local-first, no cloud dependency except LLM API.

## Architecture

```
Pipeline: detect → OCR → translate → fit → render

src/
├── api/            # Axum HTTP server, request/response models
├── pipeline/       # Orchestration: process_image, detect_and_ocr, chapter
│   ├── common.rs   # translate_and_fit, translate_only, resolve_engine
│   ├── merge.rs    # PP-OCR text line → bubble grouping (union-find)
│   └── chapter.rs  # Multi-page chapter translation (single LLM call)
├── detection/      # comic-text-detector YOLOv5 (ONNX), LocalTextMask
├── ocr/            # manga-ocr (JP) + PP-OCR v5 (EN/KR/CN) — ONNX
├── translation/    # OpenAI-compatible adapter with tool calling
├── fit_engine/     # Binary search font size, page-level normalization
├── overlay/        # Erase original text + render translated text
├── border_detect/  # Auto-detect bubble border thickness from image
├── canvas_agent/   # Vision LLM typesetting (optional, experimental)
├── text_layout/    # Font loading, text measurement, word wrap
├── glossary/       # SQLite FTS5 glossary for term consistency
├── model_hub/      # HuggingFace Hub model resolver + local cache
├── cache/          # redb disk cache for translation results
└── config/         # config.toml loading
```

## Two Detection Pipelines

Source language drives which pipeline runs:

- **Japanese (`ja`)**: `ComicTextDetector` (YOLOv5 + UNet + DBNet) → `manga-ocr`
  - Detects whole bubble polygons
  - Model outputs 3 tensors: `blk` (bbox), `seg` (UNet region), `det` (DBNet shrink map)
  - Text mask built by combining: dilate `det` channel 0 by 3px → intersect with `seg`
  - This gives tight per-character masks bounded by the text region
  - Mask is cropped per-region, resized to original coords, attached as `LocalTextMask`

- **Other languages (`en`, `ko`, `zh`)**: `PP-OCR det` (DBNet) → `PP-OCR rec`
  - Detects individual text lines (rotated quads)
  - Lines merged into bubbles by spatial proximity (`merge.rs`)
  - Text mask built from DB probability map (`prob_data`) per-line
  - Line masks composited into bubble mask via logical OR

Both paths output `(Vec<BubbleInput>, Vec<polygon>, Vec<Option<LocalTextMask>>)`.

## Text Erasure Strategy

`overlay::render()` erases original text before drawing translations:

1. **ML mask path** (preferred): If `BubbleResult.text_mask` is `Some`, use it directly.
   Mask is in page coordinates (`LocalTextMask { x, y, image }`). Paint masked pixels with `median_bg_color`.

2. **Fallback path**: If no ML mask, use classical image processing:
   threshold (lum > 180) → morphological close (dilate+erode, radius 5) → flood fill from center.
   This is less accurate but works when ML mask is unavailable.

Do NOT go back to old approaches that were tried and failed:
- ❌ Polygon shrink (centroid-based) — clips round bubble borders at corners
- ❌ Ellipse fill — only covers ~78% of bbox, misses text at edges
- ❌ Canny edge + contour finding — text strokes split contours, leaves residual text
- ❌ Snapshot + rect fill + ellipse restore — restores original text in corners

## Key Types

- `LocalTextMask { x: u32, y: u32, image: GrayImage }` — binary mask in page coords (255=text, 0=bg). Defined in `detection/mod.rs`. Skipped in API serialization (`#[serde(skip)]`).
- `DrawableArea { bbox, insets }` — canonical inner rectangle for text placement. Computed once from polygon + border detection, shared by fit_engine, overlay, canvas_agent.
- `BubbleResult` — per-bubble output carrying polygon, translated text, font size, drawable_area, and text_mask.
- `TextRegion { polygon, crop, confidence, mask }` — detection output per region.

## Conventions

- All ONNX inference runs inside `tokio::task::spawn_blocking` to avoid blocking the async runtime.
- Models auto-download from HuggingFace Hub (`nghyane/typoon-models`) if not in local `models/` dir.
- `border_detect/mod.rs` is stable — do not modify without good reason.
- Font: SamaritanTall-TB (comic style, Vietnamese diacritics coverage). Embedded at compile time.
- Line height multiplier: 1.22 (balances Vietnamese diacritics with compact typesetting).
- FitEngine does 2-pass: binary search max font per bubble → page-level cap at median × 1.35.

## Testing

```bash
cargo test --lib                        # All unit tests (20 tests)
cargo test --lib -- overlay::tests      # Overlay module only
cargo test --lib -- border_detect       # Border detection
cargo test --lib -- fit_engine          # Font fitting
cargo test --lib -- text_layout         # Text measurement + wrapping
```

Note: `DynamicImage::new_rgb8()` creates all-BLACK images. Use `RgbaImage::from_pixel(w, h, Rgba([255,255,255,255]))` for white test images.

## Future: LaMa Inpainting

Current median-color fill works for white/flat bubbles but fails on textured backgrounds (screentone, gradients). Next step is integrating LaMa FFC-ResNet inpainting model (~50MB ONNX):
- Flat background (stddev < threshold) → keep median fill (fast)
- Complex background → LaMa neural inpainting
- Reference: github.com/mayocream/koharu uses this dual-path approach
