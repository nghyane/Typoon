"""media-container — image pipeline service.

Two endpoints:

POST /prepare?chapter_id=X[&strategy=auto]
  Body : ZIP bytes (raw chapter images)
  Returns: PreparedChapterMeta JSON
  Writes:  prepared/{chapter_id}/{i:04d}.jpg + meta.json → R2 via FUSE

POST /pack?chapter_id=X
  Body : JSON { "typeset_keys": [...] }
  Returns: { "bnl_key", "size_bytes", "pages" }
  Reads:   typeset_keys from R2 via FUSE
  Writes:  render/{chapter_id}.bnl → R2 via FUSE

R2 access via tigrisfs FUSE mount at /mnt/r2.
"""

import io, os, time, zipfile, json, logging, struct, statistics
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("media")

R2_MOUNT = Path(os.environ.get("R2_MOUNT", "/mnt/r2"))

# ── Constants ──────────────────────────────────────────────────────────────────
COLOR_RATIO_THRESHOLD = 0.15
TALL_ASPECT_RATIO     = 2.0
SAT_THRESHOLD         = 30
MAX_PAGE_HEIGHT       = 4096
MIN_PAGE_HEIGHT       = 2048
SENSITIVITY           = 97
WINDOW                = 10
X_MARGINS             = 10
JPEG_QUALITY          = 92

# BNL spec constants
BNL_MAGIC       = b"MCZ\x01"
BNL_HEADER_SIZE = 8
BNL_ENTRY_SIZE  = 16
FORMAT_JPEG     = 1

app = FastAPI()


@app.get("/health")
def health():
    return {"ok": True, "service": "media-container", "r2_mount": str(R2_MOUNT)}


# ── Filesystem helpers ────────────────────────────────────────────────────────

def r2_read(key: str) -> bytes:
    return (R2_MOUNT / key).read_bytes()


def r2_write(key: str, data: bytes) -> None:
    path = R2_MOUNT / key
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


# ── Image helpers ──────────────────────────────────────────────────────────────

def encode_jpeg(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=False)
    return buf.getvalue()


def color_ratio(img: Image.Image) -> float:
    small = img.resize((256, 256), Image.BILINEAR) if min(img.size) > 256 else img
    arr   = np.array(small.convert("HSV"))
    return float((arr[:, :, 1] > SAT_THRESHOLD).mean())


def chapter_color_ratio(images: list[Image.Image]) -> float:
    n    = len(images)
    idxs = sorted({n // 4, n // 2, 3 * n // 4})
    return statistics.mean(color_ratio(images[i]) for i in idxs)


def chapter_is_tall(images: list[Image.Image]) -> bool:
    n = len(images)
    for i in sorted({n // 2, 2 * n // 3, 3 * n // 4, max(0, n - 2)}):
        if i >= n: continue
        if images[i].width <= 0 or images[i].height / images[i].width < TALL_ASPECT_RATIO:
            return False
    return True


def modal_width(images: list[Image.Image]) -> int:
    from collections import Counter
    return Counter(im.width for im in images).most_common(1)[0][0]


# ── Stitch helpers ─────────────────────────────────────────────────────────────

def confirmed_rows(strip: np.ndarray, W: int, H: int) -> np.ndarray:
    x0, x1 = X_MARGINS, W - X_MARGINS
    thr     = int(255 * (1 - SENSITIVITY / 100))
    cols    = strip[:, x0:x1, :]
    gray    = (cols @ [0.299, 0.587, 0.114])
    diff    = np.abs(np.diff(gray.astype(int), axis=1))
    valid   = ((diff.max(axis=1) <= thr) &
               (gray.max(axis=1) - gray.min(axis=1) <= thr)).astype(np.uint8)
    conf    = np.zeros(H, np.uint8)
    if H >= WINDOW:
        acc = int(valid[:WINDOW].sum())
        if acc == WINDOW: conf[0] = 1
        for y in range(1, H - WINDOW + 1):
            acc += int(valid[y + WINDOW - 1]) - int(valid[y - 1])
            if acc == WINDOW: conf[y] = 1
    return conf


def nearest_confirmed(conf: np.ndarray, lo: int, hi: int, target: int) -> int | None:
    lo = max(lo, 0); hi = min(hi, len(conf))
    idxs = np.where(conf[lo:hi])[0]
    if len(idxs) == 0: return None
    return int(lo + idxs[np.abs(idxs - (target - lo)).argmin()])


# ── Prepare strategies ─────────────────────────────────────────────────────────

def prepare_one_to_one(images: list[Image.Image], chapter_id: str) -> tuple[list[dict], list[list[int]]]:
    pages, groups = [], []
    for i, im in enumerate(images):
        t0  = time.perf_counter()
        jpg = encode_jpeg(im)
        key = f"prepared/{chapter_id}/{i:04d}.jpg"
        r2_write(key, jpg)
        log.info("prepared %02d %dx%d %dKB %.0fms",
                 i, im.width, im.height, len(jpg) // 1024,
                 (time.perf_counter() - t0) * 1000)
        pages.append({"index": i, "width": im.width, "height": im.height})
        groups.append([i])
        im.close()
    return pages, groups


def prepare_stitch(images: list[Image.Image], chapter_id: str) -> tuple[list[dict], list[list[int]]]:
    W    = modal_width(images)
    imgs = [im.resize((W, round(im.height * W / im.width)), Image.BILINEAR)
            if im.width != W else im for im in images]

    bounds: list[tuple[int, int]] = []
    row = 0
    rows_list: list[np.ndarray] = []
    for im in imgs:
        arr = np.array(im.convert("RGB"))
        rows_list.append(arr)
        bounds.append((row, row + im.height))
        row += im.height
        im.close()

    strip   = np.concatenate(rows_list, axis=0); rows_list.clear()
    H_total = strip.shape[0]
    conf    = confirmed_rows(strip, W, H_total)

    pages, groups = [], []
    prev = 0; out_idx = 0; target = MAX_PAGE_HEIGHT

    def emit(y0: int, y1: int):
        nonlocal out_idx
        jpg = encode_jpeg(Image.fromarray(strip[y0:y1], "RGB"))
        key = f"prepared/{chapter_id}/{out_idx:04d}.jpg"
        r2_write(key, jpg)
        grp = [i for i, (rs, re) in enumerate(bounds) if rs < y1 and re > y0]
        pages.append({"index": out_idx, "width": W, "height": y1 - y0})
        groups.append(grp)
        log.info("stitch %02d rows %d-%d %dKB", out_idx, y0, y1, len(jpg) // 1024)
        out_idx += 1

    while target < H_total:
        lo  = prev + MIN_PAGE_HEIGHT
        hi  = min(prev + int(MAX_PAGE_HEIGHT * 1.5), H_total)
        cut = (nearest_confirmed(conf, lo, hi, prev + MAX_PAGE_HEIGHT)
               or nearest_confirmed(conf, lo, H_total, prev + MAX_PAGE_HEIGHT)
               or prev + MAX_PAGE_HEIGHT)
        emit(prev, cut)
        prev = cut; target = prev + MAX_PAGE_HEIGHT
    if prev < H_total:
        emit(prev, H_total)

    return pages, groups


# ── /prepare ──────────────────────────────────────────────────────────────────

@app.post("/prepare")
async def prepare(req: Request):
    chapter_id = req.query_params.get("chapter_id")
    strategy   = req.query_params.get("strategy", "auto")
    if not chapter_id: raise HTTPException(400, "chapter_id required")

    t0        = time.perf_counter()
    zip_bytes = await req.body()
    log.info("prepare chapter=%s zip=%dKB strategy=%s",
             chapter_id, len(zip_bytes) // 1024, strategy)

    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except Exception as e:
        raise HTTPException(400, f"invalid ZIP: {e}")

    names = sorted(
        n for n in zf.namelist()
        if not n.endswith("/") and not n.startswith("__MACOSX/")
        and not Path(n).name.startswith(".")
        and Path(n).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    )
    if not names: raise HTTPException(400, "ZIP has no image entries")

    raw = {n: zf.read(n) for n in names}; zf.close(); zip_bytes = None
    n   = len(names)

    def open_img(name: str) -> Image.Image:
        img = Image.open(io.BytesIO(raw[name])); img.load(); return img

    samp     = [open_img(names[i]) for i in sorted({n // 4, n // 2, 3 * n // 4})]
    cr       = chapter_color_ratio(samp)
    is_color = cr >= COLOR_RATIO_THRESHOLD
    if strategy == "auto":
        strategy = "stitch" if (is_color and chapter_is_tall(samp)) else "one_to_one"
    for s in samp: s.close()

    log.info("color_ratio=%.3f is_color=%s strategy=%s n=%d", cr, is_color, strategy, n)

    images = [open_img(name) for name in names]; raw.clear()

    if strategy == "stitch":
        pages, groups = prepare_stitch(images, chapter_id)
    else:
        pages, groups = prepare_one_to_one(images, chapter_id)

    meta = {
        "chapter_id": chapter_id, "strategy": strategy,
        "is_color": is_color, "color_ratio": round(cr, 4),
        "pages": pages, "groups": groups, "raw_count": n,
    }
    r2_write(f"prepared/{chapter_id}/meta.json", json.dumps(meta).encode())

    log.info("prepare done %d pages %.1fs", len(pages), time.perf_counter() - t0)
    return JSONResponse(meta)


# ── /pack ──────────────────────────────────────────────────────────────────────

def build_bnl(pages: list[tuple[bytes, int, int]]) -> bytes:
    n          = len(pages)
    offset     = 0
    entries    = []
    for jpeg, w, h in pages:
        entries.append((offset, len(jpeg), w, h))
        offset += len(jpeg)

    buf = bytearray()
    buf += BNL_MAGIC
    buf += struct.pack("<BB", 1, 0)
    buf += struct.pack("<H", n)
    for off, sz, w, h in entries:
        buf += struct.pack("<IIHHB3x", off, sz, w, h, FORMAT_JPEG)
    for jpeg, _, _ in pages:
        buf += jpeg
    return bytes(buf)


@app.post("/storyboard")
async def storyboard(req: Request):
    """
    Body: JSON { chapter_id, pages: [{index, width, height}, ...] }
    Reads prepared JPEGs from R2 via FUSE. No scan data needed.
    Writes storyboard/{chapter_id}/{n:02d}.jpg
    Returns: { storyboard_keys }
    """
    from storyboard import build_storyboards

    body       = await req.json()
    chapter_id = body.get("chapter_id")
    pages      = body.get("pages", [])   # [{index, width, height}]
    if not chapter_id: raise HTTPException(400, "chapter_id required")
    if not pages:      raise HTTPException(400, "pages required")

    t0 = time.perf_counter()
    page_order = sorted(p["index"] for p in pages)

    pages_rgb: dict[int, np.ndarray] = {}
    for pi in page_order:
        raw = r2_read(f"prepared/{chapter_id}/{pi:04d}.jpg")
        img = Image.open(io.BytesIO(raw)); img.load()
        pages_rgb[pi] = np.array(img.convert("RGB"))

    sb_chunks = build_storyboards(pages_rgb, page_order)
    pages_rgb.clear()

    storyboard_keys: list[str] = []
    for i, (chunk_range, jpeg_bytes) in enumerate(sb_chunks):
        sb_key = f"storyboard/{chapter_id}/{i:02d}.jpg"
        r2_write(sb_key, jpeg_bytes)
        storyboard_keys.append(sb_key)
        log.info("storyboard %02d pages %d-%d %dKB",
                 i, chunk_range.start, chunk_range.stop - 1, len(jpeg_bytes) // 1024)

    log.info("storyboard done chapter=%s %.1fs", chapter_id, time.perf_counter() - t0)
    return JSONResponse({"storyboard_keys": storyboard_keys})


@app.post("/pack")
async def pack(req: Request):
    body         = await req.json()
    chapter_id   = body.get("chapter_id")
    typeset_keys = body.get("typeset_keys", [])

    if not chapter_id:   raise HTTPException(400, "chapter_id required")
    if not typeset_keys: raise HTTPException(400, "typeset_keys required")

    t0 = time.perf_counter()
    log.info("pack chapter=%s pages=%d", chapter_id, len(typeset_keys))

    pages_data: list[tuple[bytes, int, int]] = []
    for i, key in enumerate(typeset_keys):
        raw = r2_read(key)
        img = Image.open(io.BytesIO(raw)); img.load()
        jpg = encode_jpeg(img)
        w, h = img.width, img.height
        img.close()
        pages_data.append((jpg, w, h))
        log.info("packed %02d %dx%d %dKB", i, w, h, len(jpg) // 1024)

    bnl     = build_bnl(pages_data)
    bnl_key = f"render/{chapter_id}.bnl"
    r2_write(bnl_key, bnl)

    log.info("pack done %d pages %.1fs %dKB",
             len(pages_data), time.perf_counter() - t0, len(bnl) // 1024)
    return JSONResponse({
        "bnl_key":    bnl_key,
        "size_bytes": len(bnl),
        "pages":      len(pages_data),
    })
