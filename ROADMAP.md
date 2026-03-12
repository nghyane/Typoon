# ComicScan — Roadmap

Manga/manhwa translator in Rust. Ưu tiên chạy local cho cá nhân, tránh overhead.

Pipeline: **detect → OCR → translate → fit → render**

---

## 1. Single Image — Flow chạy đúng ✅

- [x] Unified pipeline (`source_lang` drives detector + OCR selection)
- [x] Translation via OpenAI-compatible API (tool calling)
- [x] FitEngine — binary search font size, page-level normalization
- [x] Overlay renderer — erase + draw
- [x] Border detection — auto inset per bubble
- [x] Cache (redb)
- [x] Canvas agent (optional, vision LLM typesetting)
- [x] API: `POST /translate-image`, `GET /health`

## 2. Single Image — Flow chạy tốt 🔲

- [ ] Test end-to-end với ảnh thật (manga JP, manhwa KR, manhua CN)
- [ ] Tune OCR confidence threshold — giảm noise, giữ text ngắn
- [ ] Tune watermark filter — tránh false positive
- [ ] Verify font rendering quality (diacritics VN, size readability)
- [ ] Fix edge cases: bubble quá nhỏ, text quá dài, polygon lệch
- [ ] Example script chạy 1 ảnh từ CLI (không cần HTTP server)

## 3. CLI Tool — Dịch file/folder nhanh 🔲

- [ ] `comicscan translate image.png -o output.png --target vi`
- [ ] `comicscan translate ./chapter/ -o ./output/ --target vi`
- [ ] Đọc config từ `config.toml`, override bằng CLI flags
- [ ] Progress bar (indicatif) cho folder mode
- [ ] Skip ảnh đã có output (simple file-based cache)

## 4. Chapter Context — Nhất quán bản dịch 🔲

- [ ] Gom tất cả pages trong folder → detect+OCR all → 1 LLM call
- [ ] Bubble IDs prefixed by page (`p0_b0`, `p1_b3`)
- [ ] Prompt grouped by page → LLM thấy reading order cả chapter
- [ ] Cache per page (cached pages excluded from translate call)
- [ ] CLI: `comicscan translate-chapter ./chapter/ -o ./output/ --target vi`

## 5. Glossary — Cố định thuật ngữ 🔲

- [ ] File `glossary.toml` per project: tên nhân vật, thuật ngữ
- [ ] Inject glossary vào system prompt
- [ ] CLI flag: `--glossary ./glossary.toml`

## 6. Nếu cần sau này

- Parallel detect+OCR (rayon)
- Web UI review + manual edit
- Multi-format export (PDF, CBZ)
- HTTP API cho browser extension
- Project/chapter DB persistence

---

## Architecture

```
src/
├── api/            # Axum handlers, models, router
├── pipeline/       # process_image, detect_and_ocr
│   ├── common.rs   # translate_and_fit, resolve_engine
│   └── merge.rs    # PP-OCR line → bubble grouping
├── detection/      # comic-text-detector (ONNX)
├── ocr/            # manga-ocr + PP-OCR (ONNX)
├── translation/    # OpenAI-compatible adapter (tool calling)
├── fit_engine/     # Binary search font size, page normalization
├── overlay/        # Render translated text on image
├── border_detect/  # Auto-detect bubble border thickness
├── canvas_agent/   # Vision LLM typesetting (optional)
├── text_layout/    # Font, measure, wrap, bbox
├── cache/          # redb disk cache
└── config/         # config.toml loading
```
