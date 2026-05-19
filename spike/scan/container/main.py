"""Scan service — Cloudflare Container.

Uses typoon vision pipeline directly (same code as Mac):
  detect (comic_detr ONNX) + Lens OCR → spatial_join → BubbleGroup
  → write scan/{chapter}/{i:04d}.msgpack + mask/{chapter}/{i:04d}.bin
  → build storyboard/{chapter}/{n:02d}.jpg

POST /scan?chapter_id=X
  Body: JSON { pages: [{page_index, prepared_key, is_color}], lang_hint?, total_pages? }
  Returns: { scan_keys, mask_keys, storyboard_keys, timings_ms }

R2 via tigrisfs FUSE mount at /mnt/r2.
"""

from __future__ import annotations

import io
import json
import os
import struct
import time
import asyncio
import logging
from pathlib import Path

import numpy as np
import msgpack
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

from storyboard import build_storyboards

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("scan")

R2_MOUNT    = Path(os.environ.get("R2_MOUNT", "/mnt/r2"))
MODEL_PATH  = Path(os.environ.get("MODEL_PATH", "/app/models/comic-detr-v4s-int8.onnx"))

# Direct S3 client for writes — tigrisfs FUSE is async/eventually-consistent;
# the workflow's next step (brief) immediately reads what scan just wrote,
# so we need synchronous PutObject for output keys.
_S3_BUCKET = os.environ.get("R2_BUCKET_NAME", "")
_S3_ENDPOINT = f"https://{os.environ.get('R2_ACCOUNT_ID', '')}.r2.cloudflarestorage.com"
_s3_client = None

def _get_s3():
    global _s3_client
    if _s3_client is None:
        import boto3
        _s3_client = boto3.client(
            "s3",
            endpoint_url=_S3_ENDPOINT,
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            region_name="auto",
        )
    return _s3_client



app = FastAPI()


@app.exception_handler(Exception)
async def _unhandled_exception(request, exc):
    """Surface stack traces in the JSON response (debug visibility).

    Wrangler tail doesn't capture container stdout, so without this the
    pipeline only sees a bare "500 Internal Server Error".
    """
    import traceback
    log.exception("unhandled %s on %s", type(exc).__name__, request.url.path)
    return JSONResponse(
        {
            "error": type(exc).__name__,
            "detail": str(exc),
            "trace": traceback.format_exc(),
        },
        status_code=500,
    )


# ── Filesystem helpers ────────────────────────────────────────────────────────

def r2_read(key: str) -> bytes:
    return (R2_MOUNT / key).read_bytes()

def r2_write(key: str, data: bytes) -> None:
    """Synchronous PutObject. Returns only after R2 acks the upload, so the
    next workflow step (brief) can read what we just wrote.
    """
    _get_s3().put_object(Bucket=_S3_BUCKET, Key=key, Body=data)


# ── Lazy runtime init ─────────────────────────────────────────────────────────

_runtime = None


class FastLensAPI:
    """Minimal Lens client — reuses one httpx.AsyncClient + protobuf builders
    from chrome_lens_py. Drop-in replacement for ``chrome_lens_py.LensAPI``
    with only the surface ``LensBlocksDetector`` needs (``process_image``
    with ``output_format="detailed"``).

    Why this exists: upstream ``LensAPI.send_request`` opens a fresh
    ``httpx.AsyncClient`` per call, paying TCP+TLS+HTTP/2 handshake for
    every OCR request (~200-500ms × ~280 calls/chapter = tens of seconds
    wasted). Sharing one keep-alive client drops that to ~zero.
    """

    def __init__(self, *, max_concurrent: int = 100, timeout: int = 60,
                 endpoint: str | None = None) -> None:
        import httpx
        from chrome_lens_py.constants import (
            DEFAULT_HEADERS, LENS_CRUPLOAD_ENDPOINT,
        )
        self._endpoint = endpoint or LENS_CRUPLOAD_ENDPOINT
        self._headers  = DEFAULT_HEADERS.copy()
        self._timeout  = timeout
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._client = httpx.AsyncClient(
            http2=True, timeout=timeout,
            limits=httpx.Limits(max_keepalive_connections=max_concurrent,
                                max_connections=max_concurrent * 2),
        )
        # Per-call session state mirrors LensRequestHandler — Lens server
        # expects monotonically-increasing sequence ids within a session.
        self._session_uuid: int | None = None
        self._seq_id = 0
        self._img_seq_id = 0

    async def aclose(self) -> None:
        await self._client.aclose()

    async def process_image(
        self, image, *,
        ocr_language: str = "",
        output_format: str = "detailed",
        new_session: bool = True,
        **_ignored,
    ) -> dict:
        from chrome_lens_py.core.protobuf_builder import create_ocr_translate_request
        from chrome_lens_py.utils.lens_betterproto import LensOverlayServerResponse

        # Encode JPEG in a worker thread (upstream uses PNG + RGBA on event loop).
        img_bytes, w, h = await asyncio.to_thread(_encode_for_lens, image)

        async with self._semaphore:
            if new_session:
                self._session_uuid = None
                self._seq_id = 0
                self._img_seq_id = 0
            self._seq_id += 1
            if new_session:
                self._img_seq_id += 1

            # Protobuf build is CPU work — push to thread too.
            payload, uuid_used = await asyncio.to_thread(
                create_ocr_translate_request,
                image_bytes=img_bytes, width=w, height=h,
                ocr_language=ocr_language,
                session_uuid=self._session_uuid,
                sequence_id=self._seq_id,
                image_sequence_id=self._img_seq_id,
            )
            if self._session_uuid is None:
                self._session_uuid = uuid_used

            resp = await self._client.post(
                self._endpoint, content=payload, headers=self._headers,
            )
            resp.raise_for_status()
            body = await resp.aread()
            # Protobuf parse is also CPU work.
            proto = await asyncio.to_thread(
                LensOverlayServerResponse.FromString, body,
            )

        return _format_response(proto, output_format)


def _encode_for_lens(image) -> tuple[bytes, int, int]:
    """Encode an image (np.ndarray or PIL.Image) to JPEG bytes for Lens.

    Runs in a worker thread so the event loop stays unblocked. Lens
    resizes inputs >1500px to ~1000px server-side; we pre-resize to
    avoid sending bytes that will be thrown away.
    """
    from PIL import Image as _PILImage
    if isinstance(image, np.ndarray):
        pil = _PILImage.fromarray(image)
    elif hasattr(image, "save"):     # PIL image
        pil = image
    else:
        raise TypeError(f"unsupported image type: {type(image)}")
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    MAX_DIM = 1500
    if pil.width > MAX_DIM or pil.height > MAX_DIM:
        pil.thumbnail((MAX_DIM, MAX_DIM), _PILImage.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=85, optimize=False)
    return buf.getvalue(), pil.width, pil.height


def _format_response(proto, output_format: str) -> dict:
    """Shape the protobuf reply into the dict ``LensBlocksDetector`` expects."""
    from math import pi as _pi

    if (not proto.HasField("objects_response")
        or not proto.objects_response.HasField("text")
        or not proto.objects_response.text.HasField("text_layout")):
        return {"detailed_blocks": [], "raw_response_objects": proto.objects_response}

    text_layout = proto.objects_response.text.text_layout

    def _word_geom(box):
        return {
            "center_x": box.center_x, "center_y": box.center_y,
            "width": getattr(box, "width", 0.0),
            "height": getattr(box, "height", 0.0),
            "angle_deg": getattr(box, "rotation_z", 0.0) * (180 / _pi),
            "coordinate_type": "NORMALIZED" if getattr(box, "coordinate_type", 0) == 1 else "IMAGE",
        }

    def _line(line):
        text = "".join(w.plain_text + w.text_separator for w in line.words).strip()
        g = line.geometry.bounding_box
        return {
            "text": text,
            "geometry": {
                "center_x": g.center_x, "center_y": g.center_y,
                "width": g.width, "height": g.height,
                "angle_deg": g.rotation_z * (180 / _pi) if g.rotation_z else 0.0,
            },
            "words": [
                {
                    "word":      w.plain_text,
                    "separator": w.text_separator,
                    "geometry":  (_word_geom(w.geometry.bounding_box)
                                  if w.HasField("geometry")
                                  and w.geometry.HasField("bounding_box") else None),
                }
                for w in line.words
            ],
        }

    def _paragraph(p):
        text = "\n".join(
            "".join(w.plain_text + w.text_separator for w in line.words).strip()
            for line in p.lines
        )
        g = p.geometry.bounding_box
        return {
            "text": text,
            "geometry": {
                "center_x": g.center_x, "center_y": g.center_y,
                "width": g.width, "height": g.height,
                "angle_deg": g.rotation_z * (180 / _pi) if g.rotation_z else 0.0,
            },
            "lines": [_line(line) for line in p.lines],
        }

    detailed = [_paragraph(p) for p in text_layout.paragraphs]
    return {
        "detailed_blocks": detailed,
        "raw_response_objects": proto.objects_response,
    }


def _get_runtime():
    global _runtime
    if _runtime is not None:
        return _runtime

    from typoon.vision._backends.comic_detr import load_session
    from typoon.vision.detectors.lens.detector import LensBlocksDetector
    from typoon.vision.groupers.lens_native import LensNativeGrouper
    from typoon.vision.pipeline import VisionPipelineSpec
    from typoon.vision.runtime import VisionRuntime

    t0 = time.perf_counter()
    comic = load_session(str(MODEL_PATH))
    fast_api = FastLensAPI(
        max_concurrent=100,
        endpoint=os.environ.get("LENS_ENDPOINT") or None,
    )
    detector = LensBlocksDetector(
        comic_detr=comic,
        endpoint=os.environ.get("LENS_ENDPOINT"),
        max_concurrent=100,
        api=fast_api,
    )
    grouper  = LensNativeGrouper()

    class _Runtime:
        def __init__(self):
            self.detector = detector
            self.grouper  = grouper

    _runtime = _Runtime()
    log.info("Vision runtime ready in %.1fs", time.perf_counter() - t0)
    return _runtime


@app.get("/health")
def health():
    return {
        "ok": True, "service": "scan-container", "r2_mount": str(R2_MOUNT),
        "lens_endpoint": os.environ.get("LENS_ENDPOINT", "(unset)"),
    }


@app.get("/pingproxy")
async def pingproxy(target: str = "", n: int = 10):
    """Measure raw RTT to the configured Lens endpoint from inside the
    container. ``?n=`` controls parallel batch size; default 10.
    """
    import httpx
    if target == "google":
        endpoint = "https://lensfrontend-pa.googleapis.com/v1/crupload"
    elif target.startswith("http"):
        endpoint = target
    else:
        endpoint = os.environ.get("LENS_ENDPOINT") or "https://lensfrontend-pa.googleapis.com/v1/crupload"

    async with httpx.AsyncClient(http2=True, timeout=15,
        limits=httpx.Limits(max_keepalive_connections=200, max_connections=400),
    ) as client:
        # Warm
        try:
            await client.post(endpoint, content=b"", timeout=10)
        except Exception:
            pass

        # n parallel
        t0 = time.perf_counter()
        await asyncio.gather(*[client.post(endpoint, content=b"", timeout=10) for _ in range(n)],
                             return_exceptions=True)
        par_ms = round((time.perf_counter() - t0) * 1000)

        # Sequential 10
        seq_ms = []
        for _ in range(10):
            t = time.perf_counter()
            try:
                await client.post(endpoint, content=b"", timeout=10)
            except Exception:
                pass
            seq_ms.append(round((time.perf_counter() - t) * 1000))

    return {
        "endpoint": endpoint,
        "n_parallel": n,
        f"par{n}_total_ms": par_ms,
        "seq_avg_ms": round(sum(seq_ms) / len(seq_ms)),
        "seq_ms": seq_ms,
    }


@app.get("/warm")
def warm():
    """Force lazy runtime init and return exception text on failure (debug)."""
    try:
        _get_runtime()
        return {"ok": True, "runtime": "ready"}
    except Exception as e:
        import traceback
        return JSONResponse(
            {"ok": False, "error": str(e), "trace": traceback.format_exc()},
            status_code=500,
        )


# ── Mask builder ──────────────────────────────────────────────────────────────

_MASK_MAGIC = b"MSK1"

def _build_mask(groups, W: int, H: int) -> bytes:
    """Build raw mask from BubbleGroup erase_masks.

    erase_masks are the spatial_join output: per-line/per-word
    binary rasters that tightly cover actual text glyphs with
    shape-aware padding already baked in. We just OR them into the
    page mask.

    Fallback to polygon AABB only when a group has no erase_masks
    (decoration-only / no-line groups). A final 5x5 dilate smooths
    anti-aliased glyph edges that Lens word bboxes miss.

    DETR bubble_regions are not used here: they cover whole speech
    balloons (including blank interior), so painting them as masks
    bleeds into the balloon background.
    """
    import cv2
    page_mask = np.zeros((H, W), dtype=np.uint8)

    for g in groups:
        used_erase = False
        for em in getattr(g, "erase_masks", ()) or ():
            ex, ey = int(em.x), int(em.y)
            tile = em.image
            th, tw = tile.shape[:2]
            x0 = max(0, ex); y0 = max(0, ey)
            x1 = min(W, ex + tw); y1 = min(H, ey + th)
            if x1 <= x0 or y1 <= y0:
                continue
            sx0 = x0 - ex; sy0 = y0 - ey
            sx1 = sx0 + (x1 - x0); sy1 = sy0 + (y1 - y0)
            region = page_mask[y0:y1, x0:x1]
            np.maximum(region, tile[sy0:sy1, sx0:sx1], out=region)
            used_erase = True
        if not used_erase:
            poly = np.array(g.polygon, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(page_mask, [poly], 255)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    page_mask = cv2.dilate(page_mask, kernel, iterations=1)

    header = _MASK_MAGIC + struct.pack("<HH", W, H)
    return header + page_mask.tobytes()


# ── Key assignment ────────────────────────────────────────────────────────────

import hashlib

_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"

def _make_key(chapter_id: str, page_index: int, idx: int, salt: int = 0) -> str:
    payload = json.dumps(
        {"chapter_id": chapter_id, "idx": idx, "page": page_index, "salt": salt},
        sort_keys=True, ensure_ascii=False,
    ).encode("utf-8")
    n = int.from_bytes(hashlib.blake2s(payload, digest_size=5).digest(), "big")
    chars = []
    for _ in range(7):
        n, r = divmod(n, len(_ALPHABET))
        chars.append(_ALPHABET[r])
    return "".join(chars)

def _assign_keys(chapter_id: str, page_index: int, count: int, used: set) -> list[str]:
    keys = []
    for idx in range(count):
        salt = 0
        while True:
            k = _make_key(chapter_id, page_index, idx, salt)
            if k not in used:
                break
            salt += 1
        used.add(k)
        keys.append(k)
    return keys


# ── Per-page scan ─────────────────────────────────────────────────────────────

async def _scan_page(
    runtime,
    chapter_id: str,
    page_index: int,
    prepared_key: str,
    lang_hint: str,
    used_keys: set,
) -> dict:
    tp0 = time.perf_counter()

    # Read + decode prepared JPEG
    raw = await asyncio.to_thread(r2_read, prepared_key)
    img = Image.open(io.BytesIO(raw)); img.load()
    image = np.array(img.convert("RGB"))
    H, W = image.shape[:2]

    # detect (comic_detr + Lens tile pass) — parallel inside detector
    detection = await runtime.detector.detect(image, lang_hint or None)

    # group — spatial_join(blocks, bubble_regions)
    groups = await runtime.grouper.group(image, detection, lang_hint or None)

    # Assign stable keys
    bubble_keys = _assign_keys(chapter_id, page_index, len(groups), used_keys)

    # Build scan msgpack
    from typoon.stages.noise import strip_noise_tokens
    scan_data = {
        "page_index":        page_index,
        "page_size":         [W, H],
        "detected_language": detection.detected_lang,
        "reading_order_rule":"ltr_topdown",
        "groups": [
            {
                "idx":           i,
                "key":           bubble_keys[i],
                "page_index":    page_index,
                "source_text":   strip_noise_tokens(g.text),
                "confidence":    float(g.confidence),
                "polygon":       [[float(x), float(y)] for x, y in g.polygon],
                "bbox":          list(map(int, [
                    min(p[0] for p in g.polygon),
                    min(p[1] for p in g.polygon),
                    max(p[0] for p in g.polygon),
                    max(p[1] for p in g.polygon),
                ])),
                "shape_kind":    g.shape_kind,
                "rotation_deg":  float(g.rotation_deg),
                "text_direction":"vertical" if getattr(g, "is_vertical", False) else "horizontal",
                "typesetting":   {
                    "font_size_px":       getattr(g.typesetting, "font_size_px", 0),
                    "line_count":         getattr(g.typesetting, "line_count", 0),
                    "avg_chars_per_line": getattr(g.typesetting, "avg_chars_per_line", 0.0),
                } if g.typesetting else None,
                "erase_polygons":[],
            }
            for i, g in enumerate(groups)
        ],
        "rejected":       [],
        "page_body_ratio": 0,
        "tile_count":     0,
        "timing_ms":      {"total": round((time.perf_counter() - tp0) * 1000)},
    }

    scan_key = f"scan/{chapter_id}/{page_index:04d}.msgpack"
    mask_key = f"mask/{chapter_id}/{page_index:04d}.bin"
    mask_bin = await asyncio.to_thread(_build_mask, groups, W, H)

    await asyncio.gather(
        asyncio.to_thread(r2_write, scan_key, msgpack.packb(scan_data, use_bin_type=True)),
        asyncio.to_thread(r2_write, mask_key, mask_bin),
    )

    log.info("page %02d: %d groups %.0fms", page_index, len(groups),
             (time.perf_counter() - tp0) * 1000)
    return {
        "page_index":  page_index,
        "scan_key":    scan_key,
        "mask_key":    mask_key,
        "bubble_keys": bubble_keys,
        "image":       image,   # kept in memory for storyboard
        "groups":      groups,
    }


# ── Main endpoint ─────────────────────────────────────────────────────────────

@app.post("/scan")
async def scan(req: Request):
    import traceback
    try:
        return await _scan_impl(req)
    except Exception as e:
        log.exception("scan failed")
        return JSONResponse(
            {"error": type(e).__name__, "detail": str(e), "trace": traceback.format_exc()},
            status_code=500,
        )


async def _scan_impl(req: Request):
    chapter_id  = req.query_params.get("chapter_id")
    if not chapter_id:
        raise HTTPException(400, "chapter_id required")

    body        = await req.json()
    pages       = body.get("pages", [])
    lang_hint   = body.get("lang_hint", "")
    if not pages:
        raise HTTPException(400, "pages required")

    log.info("scan chapter=%s pages=%d lang=%s", chapter_id, len(pages), lang_hint)
    t0      = time.perf_counter()
    runtime = _get_runtime()
    used_keys: set = set()

    # Scan all pages in parallel
    results = await asyncio.gather(*[
        _scan_page(runtime, chapter_id, p["page_index"], p["prepared_key"],
                   lang_hint, used_keys)
        for p in pages
    ])
    results = sorted(results, key=lambda r: r["page_index"])
    t_scan = time.perf_counter() - t0
    page_order = [r["page_index"] for r in results]

    t_sb0 = time.perf_counter()
    # Build {page_index → [{"key", "bbox"}]} for label overlay.
    bubbles_by_page: dict[int, list[dict]] = {}
    for r in results:
        page_bubbles = []
        for i, g in enumerate(r["groups"]):
            poly = g.polygon
            if not poly:
                continue
            xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
            page_bubbles.append({
                "key":  r["bubble_keys"][i],
                "bbox": [min(xs), min(ys), max(xs), max(ys)],
            })
        bubbles_by_page[r["page_index"]] = page_bubbles

    sb_chunks = await asyncio.to_thread(
        build_storyboards,
        {r["page_index"]: r["image"][:, :, :3] for r in results},  # RGB
        page_order,
        bubbles_by_page,
    )
    storyboard_keys = []
    for i, (chunk_range, jpeg_bytes) in enumerate(sb_chunks):
        sb_key = f"storyboard/{chapter_id}/{i:02d}.jpg"
        await asyncio.to_thread(r2_write, sb_key, jpeg_bytes)
        storyboard_keys.append(sb_key)
    t_sb = time.perf_counter() - t_sb0

    # Free images
    for r in results:
        del r["image"]

    total_ms = round((time.perf_counter() - t0) * 1000)
    log.info("scan done chapter=%s scan=%.1fs storyboard=%.1fs total=%.1fs",
             chapter_id, t_scan, t_sb, time.perf_counter() - t0)

    return JSONResponse({
        "scan_keys":       [r["scan_key"]  for r in results],
        "mask_keys":       [r["mask_key"]  for r in results],
        "storyboard_keys": storyboard_keys,
        "timings_ms":      {"scan": round(t_scan*1000), "storyboard": round(t_sb*1000), "total": total_ms},
    })
