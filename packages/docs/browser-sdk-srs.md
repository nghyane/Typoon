# SRS — `@typoon/browser-sdk`

## Goal

Build a local-only browser SDK that can receive an image element and produce a realtime translated overlay without calling Typoon cloud services.

Initial public target:

```ts
const typoon = await Typoon.create({ sourceLang: "ja", targetLang: "vi" })
const result = await typoon.translateImage(imgElement)
result.attachOverlay(pageHost)
```

Until bundled local models exist, dev mode injects a `PageAnalyzer` and
`TranslateRuntime` explicitly.

## Non-goals

- No cloud fallback.
- No Lens API in the SDK runtime.
- No integration into the production reader until the dev reader proves the page contract.
- No server job/archive path reuse for realtime mode.

Unsupported capability must fail with a typed `TypoonError` instead of silently routing to a server.

## Product acceptance

The browser SDK is accepted by final visual quality, not by reproducing the
server pipeline's intermediate data exactly.

Clean result means:

1. translated text lands in the correct visual bubble/caption;
2. text is readable, balanced, and fitted with a manga-appropriate font;
3. no line visibly escapes the drawable area;
4. source text removal/inpaint does not leave obvious ghosts when final render is enabled;
5. visible page renders first, adjacent pages warm in background;
6. no source pixels are uploaded by the SDK.

The existing Python/Rust pipeline is a reference oracle for debugging and
regression comparison. It is not a required internal architecture for the
browser SDK.

## Package layout

```text
packages/browser-sdk/
  src/
    domain/       stable contracts and geometry
    image/        browser image input -> RGBA pixels
    analyze/      page analyzer runtime boundary + dev artifact replay
    translate/    local translation runtime boundary
    inpaint/      future WebGPU/AOT runtime boundary
    render/       realtime DOM overlay and future canvas render
    models/       future manifest/cache/runtime helpers
  dev/            standalone debug reader
  scripts/        real analysis artifact generation
```

## Browser pipeline contract

The SDK should keep a browser-native page contract:

```text
HTMLImageElement | Blob | ImageBitmap
  -> ImagePixels
  -> PageAnalysis[]          # text regions, candidate bubbles, OCR text, confidence
  -> RenderPlan[]            # drawable regions + source/target text + fit hints
  -> realtime DOM overlay    # fast path
  -> optional clean canvas    # inpaint + font-render path
```

The active SDK path uses browser-native `PageAnalysis` and `RenderPlan` data.
Old server-shaped scan/group contracts are debug references only and are not part
of the public browser SDK API.

## Reference logic to reuse selectively

Reuse these ideas because they encode hard-won visual heuristics, but do not
treat field-by-field equality as the goal:

- `typoon/vision/groupers/_classify.py`
  - `sfx | dialogue | narration`
  - rotation `> 5°` is SFX
  - short wide text is SFX
  - long text is narration
- `typoon/vision/groupers/_spatial_join.py`
  - Comic-DETR anchor precedence: `text_free > bubble > text_bubble`
  - text block assignment by centre-in-anchor
  - `text_bubble` hint for dialogue container
  - reading order from writing direction
  - typesetting hints from word/line boxes
  - page body ratio calibrated padding
- `workers/translate/src/window.ts`
  - SFX bypasses dialogue windows
  - dialogue/narration preserve reading order context
- `python/typoon_inpaint/scan.py`
  - future mask planning: burst OBB, precise raster, regen fallback

The highest priority browser ports are:

1. font choice and fit logic from `crates/render/src/fit.rs` / `layout.rs`;
2. source text geometry enough to place translations correctly;
3. region grouping good enough for visual bubbles, even if group IDs differ;
4. inpaint masks that erase what will be repainted, not necessarily the same masks as server scan.

## Browser-first architecture

```text
TypoonBrowser
  ImageSource             HTMLImageElement/Blob/ImageBitmap -> ImageBitmap/RGBA
  PageScheduler           visible page priority, neighbour prewarm, cancellation
  ModelStore              OPFS/Cache Storage lazy model packs
  VisionRuntime           WebGPU/ONNX models for detection/OCR
  PageAnalyzer            browser-native region/text analysis
  TranslationRuntime      local or browser-callable translator; dev uses Google Translate
  FitRuntime              font metrics + wrapping + page-level body cap
  OverlayRenderer         fast DOM overlay for reader realtime
  CleanRenderer           inpaint + canvas/WASM render for final clean output
```

Reader integration should be pull-based per visible page. Do not run a full
chapter pipeline before showing anything.

## Model runtime plan

Local-only production runtime needs model packs with lazy download/cache:

```text
comic_detr WebGPU/ONNX   -> bubble/text anchors
OCR WebGPU/ONNX          -> text + word/line geometry
inpaint AOT/ONNX/WebGPU  -> final clean render path
translator local runtime -> no cloud fallback
```

The OCR runtime is the critical replacement for Lens. It must emit word boxes, line boxes, rotation, and text direction. Plain text plus a bbox is not enough for high-quality grouping, masks, or typesetting.

OCR outputs from tiled or per-anchor passes must be deduped by geometry before
group text is assembled. Do not remove duplicate-looking text by string content:
intentional repeated dialogue is valid manga content; overlap of bbox/line boxes
is the signal that two OCR records describe the same glyphs.

For dev mode only, Google Translate's browser endpoint is acceptable to validate
render/fit/overlay. It is not the final local-only translation runtime.

## Dev mode

Generate the real analysis artifact first:

```bash
bun run --cwd packages/browser-sdk generate:analysis
```

Run the standalone debug reader:

```bash
bun run --cwd packages/browser-sdk dev
```

It loads `dev/public/sample-page.jpg`, a hosted MangaDex page copied into the package for stable local debugging. The dev reader consumes `dev/public/artifacts/sample-analysis.json`, generated by the current project scan path (`comic_detr ONNX + LensBlocksDetector + LensNativeGrouper`) and converted to browser `RenderPlan[]`, then calls Google Translate directly in the browser. No mock OCR/group/translation data is used by dev mode.

Visual overlays/screenshots are the final gate.

## Implementation sequence

1. **Dev visual harness**
   - real MangaDex sample image;
   - real scan artifact from current pipeline;
   - Google Translate dev runtime;
   - screenshot/visual review path.
2. **Fit + font first**
   - load/render with the production manga font: `crates/render/assets/SamaritanTall-TB.ttf`;
   - do not introduce a separate browser-only font family;
   - port or call the Rust/WASM fit logic;
   - report per-bubble font size, lines, overflow.
3. **Browser page analyzer**
   - replace artifact replay with WebGPU model output;
   - focus on correct visual regions, not identical Python group IDs.
4. **Clean render path**
   - add inpaint mask/runtime;
   - output canvas/blob in addition to DOM overlay.
5. **Reader integration**
   - mount per `PageRenderer` visible page;
   - cache per page by `sourceKey + pageIndex + modelPackVersion`;
   - prewarm adjacent pages in idle time.

## Reader integration plan

Production reader integration should happen later at `web/src/features/reader/PageRenderer.tsx`:

1. keep `ReaderSource.getUrl(index)` unchanged;
2. render the image inside a `position: relative` host;
3. pass the actual `HTMLImageElement` to SDK after decode;
4. attach overlay keyed by `sourceKey + pageIndex`;
5. cache per-page scan/translation results separately from reader image blob URLs.

This keeps realtime overlay independent from the existing server job/archive flow.
