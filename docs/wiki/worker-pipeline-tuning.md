# Worker pipeline — trạng thái và điều chỉnh thuật toán

Tài liệu này ghi lại trạng thái hiện tại của spike workers, các tham số
đang điều chỉnh, và cách chạy lại để verify sau khi thay đổi.

---

## Kiến trúc tóm tắt

```
prepare → scan(i) ──┬── brief (vision) ──→ noise.json / glossary / ...
                    │                              │
                    ├── inpaint(i) ◄───────────────┤
                    │                              │
                    └──────────────────────────────┴──→ translate(window)
                                                              │
                    inpaint(i) ──────────────────────────────┤
                                                              ▼
                                                         typeset(i)
                                                              │
                                                              ▼
                                                       render-archive
```

Tất cả workers deploy lên account `hoangvananhnghia99@gmail.com`,
bucket R2 `typoon-work`, region HKG.

---

## Workers và file chính

| Worker | Source | Tham số quan trọng |
|---|---|---|
| `prepare-worker` | `spike/prepare-worker/src/index.ts` | `COLOR_RATIO_THRESHOLD=0.15`, `TALL_ASPECT_RATIO=2.0` |
| `scan-worker` | `spike/scan-worker/src/index.ts` | `STROKE_FRACTION=0.12`, `DEDUP_IOU=0.5`, `SUBSTRING_IOU=0.05` |
| `brief-worker` | `spike/brief-worker/src/index.ts` | `PAGES_PER_STORYBOARD=9`, `STORYBOARD_JPEG_Q=75` |
| `translate-worker` | `spike/translate-worker/src/index.ts` | `WINDOW_CHAR_BUDGET=3000`, `CONTEXT_SIZE=20` |
| `ort-inpaint-orchestrator` | `spike/orchestrator/src/index.ts` | `PAD_AROUND_BUBBLE=16`, `STROKE_FRACTION=0.12`, closing radius = `p25_short/3` |
| `typeset-worker` | `spike/typeset-worker/src/index.ts` | `STROKE_FRACTION=0.12` (dedupe), `orientedPolygon` rotation |
| `render-archive-worker` | `spike/render-archive-worker/src/index.ts` | `JPEG_QUALITY=92` |
| `chapter-workflow-v2` | `spike/chapter-workflow-v2/src/index.ts` | DAG orchestration |

---

## Tham số đang điều chỉnh

### 1. Mask dilation — `scan-worker`

```ts
// spike/scan-worker/src/index.ts
const STROKE_FRACTION = 0.12;

function maskDilatePx(bb): number {
  const short = Math.min(bb[2]-bb[0], bb[3]-bb[1]);
  return Math.ceil(short * STROKE_FRACTION);
}
```

**Tăng** nếu text gốc còn lờ mờ sau inpaint (halo chưa bị xoá).
**Giảm** nếu mask tràn ra ngoài bubble (xoá quá nhiều).

Python mirror: `typoon/vision/groupers/ppocr_yolo_union_find.py`
```python
_ERASE_DILATE_FRACTION_NORMAL = 0.10  # dialogue
_ERASE_DILATE_FRACTION_GLOW   = 0.16  # SFX thick stroke
```

### 2. Mask morphological closing — `orchestrator`

```ts
// spike/orchestrator/src/index.ts
const rawCcs = findBubbles(mask.data, W, H);
const shortEdges = rawCcs.map(cc => Math.min(cc.x1-cc.x0+1, cc.y1-cc.y0+1)).sort();
const p25 = shortEdges[Math.floor(shortEdges.length * 0.25)] ?? 20;
const closeRadius = Math.max(2, Math.round(p25 / 3));
```

Closing bridges word-level mask components thành bubble-level trước
flood-fill. Nếu không có closing, AOT inpaint mỗi từ riêng → leak
texture giữa các dòng (horizontal streak artifact).

**Tăng divisor** (vd `/4`) nếu closing vẫn merge neighbouring bubbles.
**Giảm divisor** (vd `/2`) nếu intra-bubble words vẫn tách rời.

Debug: chạy `docs/wiki/scripts/debug_mask.py` (xem phần Debug bên dưới).

### 3. Prepare strategy — `prepare-worker`

```ts
// spike/prepare-worker/src/index.ts
const isColor = cr >= COLOR_RATIO_THRESHOLD;  // 0.15
const isTall  = chapterIsTall(images);         // H/W >= 2.0 ở mọi sample
strategy = (isColor && isTall) ? "stitch" : "one_to_one";
```

`one_to_one` là default safe. `stitch` chỉ khi CẢ hai signal đồng ý.
Nếu chapter bị gộp sai → thêm `"strategy": "one_to_one"` vào `/start` body.

### 4. Brief noise_pages — `brief-worker`

```ts
// spike/brief-worker/src/index.ts
const noisePages = args.strategy === "stitch" ? [] : fullNoisePages(bubbles, m.noise);
```

Dưới `stitch` mode, `noise_pages` bị disable vì 1 prepared page = nhiều
raw pages → page-level noise mất ý nghĩa. Chỉ `noise_keys` per-bubble
còn hiệu lực.

### 5. Typeset rotation + dedupe — `typeset-worker`

```ts
// spike/typeset-worker/src/index.ts
function orientedPolygon(bbox, rotationDeg, isVertical)
// Word-overlap dedupe: threshold 0.6
const overlap = shared / Math.min(words.size, a.words.size);
if (overlap >= 0.6) { dupIdx = i; break; }
```

Nếu 2 bubble vẫn overlap → giảm threshold (vd 0.5).
Nếu bubble bị merge nhầm → tăng threshold (vd 0.7).

---

## Quy trình test sau khi thay đổi

### Build + deploy 1 worker

```bash
cd spike/<worker-name>
node build.mjs
CLOUDFLARE_ACCOUNT_ID=818e551312970df676abe1a0e61819c7 npx wrangler deploy --no-bundle
```

### Chạy 1 chapter test

```bash
HOST="https://chapter-workflow-v2.hoangvananhnghia99.workers.dev"
CID="test-$(date +%H%M%S)"

# Upload
curl -sX PUT --data-binary "@/tmp/test-chapter.zip" \
  -H "Content-Type: application/zip" \
  "$HOST/upload?key=raw/$CID/source.zip"

# Start
ID=$(curl -sX POST \
  -d "{\"chapter_id\":\"$CID\",\"source_lang\":\"en\",\"target_lang\":\"vi\",\"zip_key\":\"raw/$CID/source.zip\"}" \
  "$HOST/start" | python3 -c "import json,sys;print(json.load(sys.stdin)['id'])")

# Poll
while true; do
  st=$(curl -s "$HOST/status?id=$ID" | python3 -c "import json,sys;print(json.load(sys.stdin).get('status'))")
  echo "$st"
  [ "$st" = "complete" ] || [ "$st" = "errored" ] && break
  sleep 10
done
```

### Download + xem PDF

```bash
CLOUDFLARE_ACCOUNT_ID=818e551312970df676abe1a0e61819c7 \
  npx wrangler r2 object get "typoon-work/render/$CID.bnl" \
  --file /tmp/chapter.bnl --remote

python3 - <<'EOF'
import struct
from PIL import Image
data = open('/tmp/chapter.bnl', 'rb').read()
_, _, count = struct.unpack('<IHH', data[:8])
imgs = []
for i in range(count):
    base = 8 + i*16
    off, sz = struct.unpack('<II', data[base:base+8])
    body = data[off:off+sz]
    open(f'/tmp/p{i:02d}.jpg', 'wb').write(body)
    imgs.append(Image.open(f'/tmp/p{i:02d}.jpg').convert('RGB'))
imgs[0].save('/tmp/chapter.pdf', save_all=True, append_images=imgs[1:], resolution=150)
print(f'{count} pages → /tmp/chapter.pdf')
EOF
open /tmp/chapter.pdf
```

---

## Debug mask

Visualize mask overlay (đỏ = mask, vàng = block bbox, xanh = word bbox):

```bash
CID=<chapter_id>
for i in 0 1 2 3 4; do
  CLOUDFLARE_ACCOUNT_ID=818e551312970df676abe1a0e61819c7 \
    npx wrangler r2 object get "typoon-work/prepared/$CID/000$i.jpg" \
    --file /tmp/prep_$i.jpg --remote
  CLOUDFLARE_ACCOUNT_ID=818e551312970df676abe1a0e61819c7 \
    npx wrangler r2 object get "typoon-work/scan/$CID/000$i.json" \
    --file /tmp/scan_$i.json --remote
  CLOUDFLARE_ACCOUNT_ID=818e551312970df676abe1a0e61819c7 \
    npx wrangler r2 object get "typoon-work/scan/$CID/000$i.mask.png" \
    --file /tmp/mask_$i.png --remote
done
```

```python
import json, numpy as np
from PIL import Image, ImageDraw

for i in range(5):
    prep = Image.open(f'/tmp/prep_{i}.jpg').convert('RGBA')
    mask = np.array(Image.open(f'/tmp/mask_{i}.png').convert('L'))
    scan = json.load(open(f'/tmp/scan_{i}.json'))
    overlay = prep.copy()
    d = ImageDraw.Draw(overlay, 'RGBA')
    # Red mask
    red = np.zeros((*mask.shape, 4), dtype=np.uint8)
    red[mask > 127] = [255, 0, 0, 100]
    overlay = Image.alpha_composite(overlay, Image.fromarray(red, 'RGBA'))
    d = ImageDraw.Draw(overlay, 'RGBA')
    for b in scan['blocks']:
        d.rectangle(b['bbox'], outline=(255,200,0,255), width=2)
        for w in b.get('words', []):
            d.rectangle(w['bbox'], outline=(0,255,80,255), width=1)
    overlay.convert('RGB').save(f'/tmp/overlay_{i}.jpg', quality=85)
```

**Đọc overlay:**
- Đỏ phủ hết text gốc (kể cả halo trắng) → mask đủ
- Đỏ tràn ra ngoài bubble → `STROKE_FRACTION` quá cao
- Đỏ không bridge các dòng trong bubble → closing radius quá nhỏ
- Đỏ merge 2 bubble liền nhau → closing radius quá lớn

---

## Vấn đề đã biết / đang theo dõi

| Vấn đề | Nguyên nhân | Trạng thái |
|---|---|---|
| Text gốc còn lờ mờ sau inpaint | Mask không cover halo, hoặc AOT model yếu | Đang điều chỉnh `STROKE_FRACTION` + closing |
| Font quá to / quá nhỏ | Hint từ scan không chính xác | Hint từ `b.lines.length` + bbox, không hardcap |
| Bubble overlap (2 text đè nhau) | Lens tile-merge detect 2 lần | Word-overlap dedupe ≥0.6 trong typeset |
| Vertical bubble sai vị trí | `normToPx` không rotate AABB | Đã fix: rotate 4 corners trước min/max |
| Cover page bị dịch | brief `noise_pages` skip toàn page | Đã fix: stitch mode disable noise_pages |
| Chapter manga bị stitch | `isColor=true` → stitch sai | Đã fix: AND(isColor, isTall) |

---

## Secrets cần set sau khi deploy worker mới

```bash
cd spike/<worker>
echo -n "sk-..." | npx wrangler secret put PACKY_API_KEY
```

Workers cần key: `brief-worker`, `translate-worker`.
