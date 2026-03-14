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

## 2. Chapter Context — Nhất quán bản dịch ✅

- [x] Gom tất cả pages trong folder → detect+OCR all → 1 LLM call
- [x] Bubble IDs prefixed by page (`p0_b0`, `p1_b3`)
- [x] Prompt grouped by page → LLM thấy reading order cả chapter
- [x] Context store (SQLite + FTS5) — cross-chapter memory
- [x] Context agent (sub-agent with tool calling) — search translations + notes
- [x] Proactive notes injection (relationship, character) vào prompt
- [x] CLI: `comicscan translate` — series/chapter mode

## 3. Glossary — Cố định thuật ngữ ✅

- [x] SQLite FTS5 glossary DB
- [x] Inject glossary matches vào system prompt
- [x] Import từ `glossary.toml`

## 4. Performance ✅

- [x] CoreML EP cho tất cả ONNX sessions (macOS)
- [x] Parallel render (std::thread::scope) cho median-fill pages
- [x] LaMa inpainting sequential under Mutex
- [x] Inter-chapter pipeline parallelism (detect N+1 while translating N)
- [x] FTS5 thay SemanticEmbedder — bỏ tokenizers crate, không còn CoreML context leak

## 5. Nếu cần sau này

- LaMa inpainting thực tế test (scaffold xong, chưa test production)
- Web UI review + manual edit
- Multi-format export (PDF, CBZ)
- HTTP API cho browser extension
