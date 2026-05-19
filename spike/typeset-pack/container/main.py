"""typeset-pack container — render + pack in one call.

POST /typeset-pack
  Body: JSON {
    chapter_id: str,
    pages: [{ page_index, inpaint_key, scan_key, page_width }],
    translate_key: str,
  }
  Returns: { archive_key, size_bytes, pages }

Uses PyO3 typoon_render (same Rust core as the Worker WASM, Python binding).
No Node.js, no subprocess. Direct Python → Rust → numpy → Pillow → BNL.
"""

from __future__ import annotations

import io, json, os, struct, time, asyncio, logging
from pathlib import Path

import numpy as np
from PIL import Image
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import msgpack

# PyO3 render binding — built from crates/render with feature=python
import typoon_render

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("typeset-pack")

R2_MOUNT  = Path(os.environ.get("R2_MOUNT", "/mnt/r2"))
JPEG_Q    = 92
BNL_MAGIC = b"MCZ\x01"
FORMAT_JPEG = 1

app = FastAPI()


@app.get("/health")
def health():
    return {"ok": True, "service": "typeset-pack"}


# ── Filesystem helpers ────────────────────────────────────────────────────────

def r2_read(key: str) -> bytes:
    return (R2_MOUNT / key).read_bytes()

def r2_write(key: str, data: bytes) -> None:
    p = R2_MOUNT / key
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


# ── BNL builder ───────────────────────────────────────────────────────────────

def build_bnl(pages: list[tuple[bytes, int, int]]) -> bytes:
    n, offset, entries = len(pages), 0, []
    for jpeg, w, h in pages:
        entries.append((offset, len(jpeg), w, h)); offset += len(jpeg)
    buf = bytearray(BNL_MAGIC + struct.pack("<BBH", 1, 0, n))
    for off, sz, w, h in entries:
        buf += struct.pack("<IIHHB3x", off, sz, w, h, FORMAT_JPEG)
    for jpeg, _, _ in pages:
        buf += jpeg
    return bytes(buf)


# ── Per-page render ───────────────────────────────────────────────────────────

def _render_page(
    page_index:   int,
    inpaint_key:  str,
    scan_key:     str,
    page_width:   int,
    trans_by_page: dict[int, dict[int, dict]],
) -> tuple[bytes, int, int]:
    """Read, render, JPEG-encode. Returns (jpeg_bytes, width, height)."""
    inpaint_raw = r2_read(inpaint_key)
    scan        = msgpack.unpackb(r2_read(scan_key), raw=False)

    img  = Image.open(io.BytesIO(inpaint_raw)); img.load()
    rgba = np.array(img.convert("RGBA"), dtype=np.uint8)
    W, H = img.width, img.height

    translations = trans_by_page.get(page_index, {})
    polygons, texts, hints = [], [], []
    for g in scan.get("groups", []):
        op = translations.get(g["idx"])
        if not op or op.get("kind") == "skip" or not (op.get("text") or "").strip():
            continue
        polygons.append([[float(x), float(y)] for x, y in g["polygon"]])
        texts.append(op["text"])
        hints.append(None)  # typesetting hints not stored in msgpack yet

    if not texts:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=JPEG_Q)
        return buf.getvalue(), W, H

    result = typoon_render.render(rgba, polygons, texts, page_width, hints)
    out_rgba = np.asarray(result.image)           # H×W×4 uint8
    out_rgb  = Image.fromarray(out_rgba[:, :, :3], "RGB")
    buf = io.BytesIO()
    out_rgb.save(buf, format="JPEG", quality=JPEG_Q)
    return buf.getvalue(), out_rgb.width, out_rgb.height


# ── Main endpoint ─────────────────────────────────────────────────────────────

@app.post("/typeset-pack")
async def typeset_pack(req: Request):
    body          = await req.json()
    chapter_id    = body.get("chapter_id")
    pages         = body.get("pages", [])
    translate_key = body.get("translate_key")
    if not chapter_id:    raise HTTPException(400, "chapter_id required")
    if not pages:         raise HTTPException(400, "pages required")
    if not translate_key: raise HTTPException(400, "translate_key required")

    log.info("typeset-pack chapter=%s pages=%d", chapter_id, len(pages))
    t0 = time.perf_counter()

    translate = json.loads(r2_read(translate_key))
    trans_by_page: dict[int, dict[int, dict]] = {}
    for op in translate.get("translations", []):
        trans_by_page.setdefault(op["page_index"], {})[op["block_idx"]] = {
            "text": op["text"], "kind": op["kind"],
        }

    results = await asyncio.gather(*[
        asyncio.to_thread(
            _render_page,
            p["page_index"], p["inpaint_key"], p["scan_key"],
            p.get("page_width", 1250), trans_by_page,
        )
        for p in pages
    ])

    bnl     = build_bnl(list(results))
    bnl_key = f"render/{chapter_id}.bnl"
    await asyncio.to_thread(r2_write, bnl_key, bnl)

    total_ms = round((time.perf_counter() - t0) * 1000)
    log.info("done chapter=%s %d pages %dms %dKB",
             chapter_id, len(pages), total_ms, len(bnl) // 1024)

    return JSONResponse({
        "archive_key": bnl_key,
        "size_bytes":  len(bnl),
        "pages":       len(pages),
        "timings_ms":  {"total": total_ms},
    })
