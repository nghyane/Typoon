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
- [x] LaMa optimized model: BN fusion + FP16 — 3s(CPU)→280ms(CoreML)→3ms/tile (1000x), 71dB PSNR
- [x] Lock-free concurrent inference (ort fork, thread-safe Session::run)
- [x] Rayon par_iter page rendering

## 5. Nếu cần sau này

- Web UI review + manual edit
- Multi-format export (PDF, CBZ)
- HTTP API cho browser extension

---

## Blog: LaMa FP16 Black Output — Root Cause & Fix (93x speedup)

> Ghi chú cho blog post sau này.

### Problem
LaMa ONNX FP16 model (Carve/LaMa-ONNX) thi thoảng ra ảnh đen.
Reported trên GitHub (advimman/lama#315) nhưng chưa ai tìm root cause.

### Root Cause
LaMa FFC architecture: `Conv_local + Conv_global → Add → BatchNormalization`.
BN có `running_var` giá trị rất lớn (lên tới 10000+ ở FP32).
Khi naive convert FP32→FP16:
- FP16 max = 65504 nhưng precision chỉ ~3 chữ số thập phân
- `running_var = 10000` trong FP16 → `(x - mean) / sqrt(10000)` mất precision
- Nhiều resblock có `running_var` min=max=10000 (tất cả channels bão hòa)
- Kết quả: BN output sai → cascade qua 18 resblocks → output toàn 0 (ảnh đen)

### Tại sao standard BN folding không apply được
- BN folding chuẩn yêu cầu BN **ngay sau Conv**: `y = BN(Conv(x))` → fuse vào weight
- LaMa FFC đặt BN sau **Add**: `y = BN(Conv_l2l(x) + Conv_g2l(x))`
- Các tool tự động (ORT graph optimizer, onnxsim) không nhận diện pattern này
- Nên 75 BN nodes vẫn nguyên, mang theo running_var overflow

### Fix
1. **Replace BN → Mul+Add** (precompute scale/bias ở FP64):
   - `scale = gamma / sqrt(var + eps)` — giá trị nhỏ, safe cho FP16
   - `bias = beta - mean * scale` — cũng nhỏ
   - Max |weight| sau fuse: 31.2 (vs 10000 trước đó)
2. **onnxsim** constant folding — giảm 125 ops
3. **onnxconverter-common FP16** — proper type propagation

### Kết quả
- **Speed**: 3s (CPU) → 280ms (CoreML FP32) → 3ms/tile (CoreML FP16 fused) = **1000x**
- **Quality**: PSNR 71dB, max pixel diff 2.8/255 (imperceptible)
- **Size**: 196MB → 98MB
- **Concurrent**: 1ms/tile với 4 threads (rayon par_iter, lock-free session)
- 3s→280ms: chuyển từ CPU sang CoreML GPU (nhưng BN vẫn gây overhead)
- 280ms→3ms: BN fusion loại bỏ running_var → CoreML GPU chạy full speed
  (BN với extreme running_var có thể khiến CoreML fallback CPU hoặc
  gây excessive dispatch overhead do numerical instability checks)

### Script
`scripts/optimize_lama.py` — chạy: `python scripts/optimize_lama.py in.onnx out.onnx`

### Key Insight
Naive FP16 conversion fails khi model có BN với large running statistics.
BN folding (dù không fold vào Conv) loại bỏ hoàn toàn running_var/mean,
biến BN thành Mul+Add với giá trị nhỏ → FP16 safe → CoreML GPU full speed.
