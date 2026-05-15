"""Bing reverse-image-search OCR detector.

Bing Visual Search returns OCR with line-level bounding boxes (normalized
0-1 coordinates) and recognized text per line. Unlike Lens, Bing does NOT
group lines into bubbles — it returns one flat region containing all lines.
Pair this detector with `ppocr_yolo_union_find` to get bubble grouping.

Two-step protocol:
  1. POST image to /images/search (multipart form, sbiupload mode)
     → 302 redirect with insightsToken in query string
  2. POST to /images/api/custom/knowledge with {invokedSkills: ["OCR"]}
     → JSON with tags[].actions[].data.regions[].lines[]

Bing rejects requests from non-Chrome TLS fingerprints. We use curl_cffi
with impersonate="chrome" to mimic a real browser handshake.

Image limits enforced before upload (mirrors owocr/Bing constraints):
  - max edge   : 4000px
  - max bytes  : ~767 KB encoded (PNG → JPEG fallback with quality ramp)
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

from ..contracts import DetectionResult, TextBlock


__all__ = ["BingBlocksDetector", "BingUnavailableError"]

logger = logging.getLogger(__name__)


# ─── Constants ────────────────────────────────────────────────────────────


_UPLOAD_URL  = "https://www.bing.com/images/search?view=detailv2&iss=sbiupload"
_API_URL     = "https://www.bing.com/images/api/custom/knowledge"
_REFERER_FMT = "https://www.bing.com/images/search?view=detailV2&insightstoken={token}"
_ORIGIN      = "https://www.bing.com"
_IMPERSONATE = "chrome"

# Bing image preprocessing limits (from owocr's Bing class)
_MIN_PIXEL = 50
_MAX_PIXEL = 4000
_MAX_BYTES = 767_772

_REQUEST_TIMEOUT_S = 30.0


# ─── Errors ───────────────────────────────────────────────────────────────


class BingUnavailableError(RuntimeError):
    """curl_cffi missing or upstream unreachable / shape changed."""


# ─── Image preprocessing ──────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _PreparedImage:
    payload_b64:  str
    encoded_size: int
    width:        int
    height:       int


def _prepare_for_bing(image: np.ndarray) -> _PreparedImage:
    """Resize + encode RGB array to fit Bing's upload limits.

    Strategy mirrors owocr.limit_image_size:
      - downscale longest edge to <= 4000
      - try PNG; if too big, fall back to JPEG with quality ramp 80→60
    """
    pil = Image.fromarray(image)
    if max(pil.size) > _MAX_PIXEL:
        scale = _MAX_PIXEL / max(pil.size)
        pil = pil.resize(
            (int(pil.width * scale), int(pil.height * scale)),
            Image.Resampling.LANCZOS,
        )

    payload = _encode_png(pil)
    if len(payload) <= _MAX_BYTES:
        return _PreparedImage(
            payload_b64=base64.b64encode(payload).decode("ascii"),
            encoded_size=len(payload),
            width=pil.width,
            height=pil.height,
        )

    # PNG too big — try a moderate downscale (owocr.limit_image_size).
    scale = 0.60 if max(pil.size) > 2000 else 0.75
    smaller = pil.resize(
        (int(pil.width * scale), int(pil.height * scale)),
        Image.Resampling.LANCZOS,
    )
    smaller_png = _encode_png(smaller)
    if len(smaller_png) <= _MAX_BYTES:
        return _PreparedImage(
            payload_b64=base64.b64encode(smaller_png).decode("ascii"),
            encoded_size=len(smaller_png),
            width=smaller.width,
            height=smaller.height,
        )

    # Still oversized — JPEG quality ramp, retry once on the smaller copy.
    for candidate in (pil, smaller):
        jpeg = _encode_jpeg_iterative(candidate, _MAX_BYTES)
        if len(jpeg) <= _MAX_BYTES:
            return _PreparedImage(
                payload_b64=base64.b64encode(jpeg).decode("ascii"),
                encoded_size=len(jpeg),
                width=candidate.width,
                height=candidate.height,
            )

    # Last-resort: return the smallest JPEG attempt; caller raises.
    return _PreparedImage(
        payload_b64=base64.b64encode(jpeg).decode("ascii"),
        encoded_size=len(jpeg),
        width=candidate.width,
        height=candidate.height,
    )


def _encode_png(pil: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil.convert("RGB").save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def _encode_jpeg_iterative(pil: Image.Image, max_bytes: int) -> bytes:
    rgb = pil.convert("RGB")
    for quality in (80, 75, 70, 65, 60):
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=quality, optimize=True)
        if buf.tell() <= max_bytes:
            return buf.getvalue()
    # Fallback: return last attempt even if oversized — Bing will reject,
    # giving a clearer error than a silent crash.
    return buf.getvalue()


# ─── Response parsing ────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _Quad:
    x: float
    y: float
    w: float
    h: float


def _quad_from_bing_bbox(quad: dict) -> _Quad | None:
    """Bing 4-corner normalized polygon → axis-aligned (x, y, w, h) in [0, 1]."""
    try:
        xs = (
            quad["topLeft"]["x"], quad["topRight"]["x"],
            quad["bottomLeft"]["x"], quad["bottomRight"]["x"],
        )
        ys = (
            quad["topLeft"]["y"], quad["topRight"]["y"],
            quad["bottomLeft"]["y"], quad["bottomRight"]["y"],
        )
    except (KeyError, TypeError):
        return None
    x1, y1 = min(xs), min(ys)
    x2, y2 = max(xs), max(ys)
    return _Quad(x=x1, y=y1, w=x2 - x1, h=y2 - y1)


def _extract_lines(payload: dict) -> list[tuple[str, _Quad]]:
    """Pull out (text, normalized bbox) pairs from Bing's response shape.

    Path: tags[displayName=##TextRecognition]
            .actions[_type=ImageKnowledge/TextRecognitionAction]
              .data.regions[].lines[]
    """
    out: list[tuple[str, _Quad]] = []
    for tag in payload.get("tags") or []:
        if tag.get("displayName") != "##TextRecognition":
            continue
        for action in tag.get("actions") or []:
            if action.get("_type") != "ImageKnowledge/TextRecognitionAction":
                continue
            for region in (action.get("data") or {}).get("regions") or []:
                for line in region.get("lines") or []:
                    text = (line.get("text") or "").strip()
                    bbox = line.get("boundingBox")
                    if not text or not bbox:
                        continue
                    quad = _quad_from_bing_bbox(bbox)
                    if quad is not None:
                        out.append((text, quad))
    return out


def _line_to_block(
    text: str, quad: _Quad, page_w: int, page_h: int,
) -> TextBlock:
    x1 = max(0, int(round(quad.x * page_w)))
    y1 = max(0, int(round(quad.y * page_h)))
    x2 = min(page_w, int(round((quad.x + quad.w) * page_w)))
    y2 = min(page_h, int(round((quad.y + quad.h) * page_h)))
    return TextBlock(
        bbox=(x1, y1, x2, y2),
        polygon=None,
        confidence=1.0,  # Bing doesn't surface per-line confidence
        text=text,
        detector="bing_blocks",
    )


# ─── Detector ─────────────────────────────────────────────────────────────


class BingBlocksDetector:
    """Bing reverse-image-search OCR.

    Produces line-level TextBlocks with recognised text. Pair with
    `ppocr_yolo_union_find` so the YOLO bubble-scope groups lines into
    bubbles.

    Talks directly to bing.com. Discord Activity URL Mappings cannot
    proxy Bing because the upload step relies on a 302 with an
    `insightsToken` query string; CF Worker `fetch()` auto-follows the
    redirect and the token is lost. APAC users pay direct latency.
    """

    name = "bing_blocks"

    def __init__(self) -> None:
        try:
            import curl_cffi  # noqa: F401
        except ImportError as e:
            raise BingUnavailableError(
                "curl_cffi not installed; install with `pip install curl_cffi`"
            ) from e
        self._session = None  # lazy

    async def detect(self, image: np.ndarray, lang: str | None) -> DetectionResult:
        h, w = image.shape[:2]
        # Network + curl-cffi is sync. Run in thread to avoid blocking loop.
        payload, prepared = await asyncio.to_thread(self._call_bing, image)
        lines = _extract_lines(payload)
        blocks = tuple(
            _line_to_block(text, quad, w, h)
            for text, quad in lines
        )
        logger.info(
            "bing_blocks: %d lines (encoded %d KB → %dx%d)",
            len(blocks), prepared.encoded_size // 1024,
            prepared.width, prepared.height,
        )
        return DetectionResult(
            blocks=blocks,
            text_already_recognized=True,
            page_size=(w, h),
        )

    # ── Internal sync work ────────────────────────────────────────────────

    def _get_session(self):
        if self._session is None:
            import curl_cffi
            self._session = curl_cffi.Session()
        return self._session

    def _call_bing(self, image: np.ndarray) -> tuple[dict, _PreparedImage]:
        import curl_cffi
        session = self._get_session()
        prepared = _prepare_for_bing(image)
        if prepared.encoded_size > _MAX_BYTES:
            raise BingUnavailableError(
                f"image too large after preprocessing "
                f"({prepared.encoded_size} > {_MAX_BYTES} bytes)"
            )

        token = self._upload(session, prepared)
        return self._fetch_ocr(session, token), prepared

    def _upload(self, session, prepared: _PreparedImage) -> str:
        import curl_cffi
        mp = curl_cffi.CurlMime()
        mp.addpart(name="imgurl", data="")
        mp.addpart(name="cbir", data="sbi")
        mp.addpart(name="imageBin", data=prepared.payload_b64)

        try:
            res = session.post(
                _UPLOAD_URL,
                headers={"origin": _ORIGIN},
                multipart=mp,
                allow_redirects=False,
                impersonate=_IMPERSONATE,
                timeout=_REQUEST_TIMEOUT_S,
            )
        except Exception as e:  # curl_cffi raises various subclasses
            raise BingUnavailableError(f"upload failed: {e}") from e

        if res.status_code != 302:
            raise BingUnavailableError(
                f"upload expected 302, got {res.status_code}"
            )
        location = res.headers.get("Location") or ""
        token = (
            parse_qs(urlparse(location).query)
            .get("insightsToken", [None])[0]
        )
        if not token:
            raise BingUnavailableError(
                f"no insightsToken in redirect: {location[:200]!r}"
            )
        return token

    def _fetch_ocr(self, session, token: str) -> dict:
        import curl_cffi
        mp = curl_cffi.CurlMime()
        mp.addpart(
            name="knowledgeRequest",
            content_type="application/json",
            data=json.dumps({
                "imageInfo": {
                    "imageInsightsToken": token,
                    "source": "Url",
                },
                "knowledgeRequest": {
                    "invokedSkills": ["OCR"],
                    "index": 1,
                },
            }),
        )
        try:
            res = session.post(
                _API_URL,
                headers={
                    "origin": _ORIGIN,
                    "referer": _REFERER_FMT.format(token=token),
                },
                multipart=mp,
                impersonate=_IMPERSONATE,
                timeout=_REQUEST_TIMEOUT_S,
            )
        except Exception as e:
            raise BingUnavailableError(f"OCR call failed: {e}") from e

        if res.status_code != 200:
            raise BingUnavailableError(
                f"OCR expected 200, got {res.status_code}"
            )
        try:
            return res.json()
        except Exception as e:
            raise BingUnavailableError(f"OCR response not JSON: {e}") from e
