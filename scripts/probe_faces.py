"""Probe idea 3 — augment storyboard with face-crop close-ups.

Pipeline:
  1. Scan chapter (real OCR + polygons).
  2. Detect anime faces per page using deepghs/anime_face_detection
     (yolov8n, 5.9MB, ~25ms/page on CPU).
  3. For each face, find the nearest bubble polygon — that bubble is
     LIKELY spoken by that face (closest tail-end proxy).
  4. Crop each face → tile into a "face gallery" image.
  5. Single LLM call: storyboard + face gallery + bubble list →
     speaker assignment + character extraction + style + noise.

Compare against probe_combined.py (storyboard only) — same 9 pages.

Output: debug-runs/storyboard_proto/probe_faces.md
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import cast

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from speaker_probe_3x3 import WebpPreparedReader, build_grid, OUT
from probe_combined import SYSTEM, build_user, parse_response

from typoon.adapters.vision_runtime import VisionRuntime
from typoon.config import load_config
from typoon.llm.ir import ContentPart, Message
from typoon.providers import make_vision_provider
from typoon.stages.keys import assign_keys
from typoon.stages.scan import scan_chapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
CHAP = ROOT / "cache" / "probe_chapter"
SLICE = list(sorted(CHAP.glob("*.png")))[5:14]
FACE_MODEL = ROOT / "cache" / "models" / "anime_face_v1.4_n.pt"


def _font(size: int) -> ImageFont.ImageFont:
    for cand in [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        if Path(cand).exists():
            try:
                return cast(ImageFont.ImageFont, ImageFont.truetype(cand, size))
            except OSError:
                pass
    return ImageFont.load_default()


def detect_faces(model: YOLO, image_path: Path, conf: float = 0.30) -> list[dict]:
    """Run face detector, return [{box: (x1,y1,x2,y2), conf}]."""
    res = model(str(image_path), verbose=False, conf=conf, imgsz=640)
    out = []
    if res[0].boxes is None:
        return out
    for b in res[0].boxes:
        x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
        out.append({"box": (x1, y1, x2, y2), "conf": float(b.conf)})
    return out


def crop_face(image: np.ndarray, box: tuple[float, float, float, float], pad: float = 0.25) -> np.ndarray:
    """Crop with padding around face box; clamp to image bounds."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    fw, fh = x2 - x1, y2 - y1
    px, py = fw * pad, fh * pad
    cx1 = max(0, int(x1 - px))
    cy1 = max(0, int(y1 - py))
    cx2 = min(w, int(x2 + px))
    cy2 = min(h, int(y2 + py))
    return image[cy1:cy2, cx1:cx2]


def build_face_gallery(
    face_crops: list[tuple[str, np.ndarray]],
    cols: int = 4,
    tile: int = 256,
    pad: int = 6,
) -> Image.Image:
    """Tile face crops into a grid with face-id labels."""
    rows = math.ceil(len(face_crops) / cols)
    canvas_w = cols * (tile + pad) + pad
    canvas_h = rows * (tile + pad) + pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), (15, 15, 15))

    font = _font(20)
    draw = ImageDraw.Draw(canvas)

    for i, (label, crop) in enumerate(face_crops):
        r, c = i // cols, i % cols
        x = pad + c * (tile + pad)
        y = pad + r * (tile + pad)

        ch, cw = crop.shape[:2]
        scale = min(tile / max(cw, 1), tile / max(ch, 1))
        nw, nh = max(1, int(cw * scale)), max(1, int(ch * scale))
        pil = Image.fromarray(crop).resize((nw, nh), Image.LANCZOS)
        # Center within tile.
        ox = x + (tile - nw) // 2
        oy = y + (tile - nh) // 2
        canvas.paste(pil, (ox, oy))

        # Label band at top-left of tile.
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle((x + 1, y + 1, x + tw + 8, y + th + 6), fill=(255, 80, 80))
        draw.text((x + 4, y + 2), label, fill=(255, 255, 255), font=font)

    return canvas


def nearest_bubble_for_face(face_center: tuple[float, float], bubbles: list[dict]) -> str | None:
    """Find bubble whose polygon center is closest to the face center."""
    if not bubbles:
        return None
    fx, fy = face_center
    best_key, best_d = None, float("inf")
    for b in bubbles:
        poly = b["polygon"]
        cx = sum(p[0] for p in poly) / len(poly)
        cy = sum(p[1] for p in poly) / len(poly)
        d = (cx - fx) ** 2 + (cy - fy) ** 2
        if d < best_d:
            best_d = d
            best_key = b["key"]
    return best_key


# ── Augmented prompt — give model both storyboard + face gallery ─────


SYSTEM_AUGMENTED = SYSTEM + """

Additional input: a SECOND image is a "face gallery" — face crops detected
across the chapter, labelled F1, F2, F3… Use this gallery to confirm character
identity. If a face appears multiple times across pages, it's the same person —
prefer using a stable name for that face throughout.

In the CHARACTERS section, you may reference face IDs in the role field
(e.g. role="appears as F1, F3"). This helps the downstream pipeline verify
character consistency across chapters.
"""


def build_user_with_faces(
    bubbles: list[dict],
    face_hints: list[dict],
    target_lang: str,
) -> str:
    lines = [
        f"Target language: {target_lang}",
        "",
        "Bubble list:",
    ]
    for b in bubbles:
        text = b["text"].replace("\n", " ")[:80] or "(empty)"
        lines.append(f"@@ {b['key']} page={b['page']} kind={b['shape_kind']} text={text!r}")
    if face_hints:
        lines.append("")
        lines.append("Face crops (gallery image follows storyboard):")
        for h in face_hints:
            nearest = h.get("nearest_bubble", "")
            n_part = f" nearest_bubble={nearest}" if nearest else ""
            lines.append(f"@@ {h['id']} page={h['page']}{n_part}")
    return "\n".join(lines)


# ── Driver ──────────────────────────────────────────────────────────


async def main() -> None:
    config, paths = load_config()
    runtime = VisionRuntime.from_config(config, paths, source_lang="en")[0]

    reader = WebpPreparedReader(SLICE)
    prepared = reader.chapter("chainsaw")
    log.info("scanning…")
    t0 = time.monotonic()
    out = scan_chapter(prepared, reader, runtime, source_lang="en")
    scan_t = time.monotonic() - t0
    log.info("scan: %.1fs, %d bubbles", scan_t, len(out.chapter.all_bubbles))

    keyed = assign_keys(out.chapter.all_bubbles, chapter_id=1)
    per_page: list[list[dict]] = [[] for _ in range(reader.page_count)]
    flat: list[dict] = []
    for bk in keyed:
        b = bk.bubble
        e = {"key": bk.key, "page": b.page_index, "text": b.source_text,
             "shape_kind": b.shape_kind, "polygon": b.box.polygon}
        per_page[b.page_index].append(e)
        flat.append(e)

    # Storyboard (same as combined probe).
    sb = build_grid(SLICE, per_page, cols=3, rows=3,
                    cell_w=700, cell_h=950, max_edge=2048, label_size=26)
    sb_path = OUT / "probe_faces_storyboard.jpg"
    sb.save(sb_path, quality=88, optimize=True)

    # Face detection.
    log.info("loading face detector…")
    face_model = YOLO(str(FACE_MODEL))
    _ = face_model(str(SLICE[0]), verbose=False, conf=0.3)  # warmup

    log.info("detecting faces…")
    t0 = time.monotonic()
    face_records: list[dict] = []  # {id, page, box, crop, nearest_bubble}
    face_id = 1
    for pi, page_path in enumerate(SLICE):
        img = reader.read_rgb(pi)
        detections = detect_faces(face_model, page_path, conf=0.35)
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            crop = crop_face(img, (x1, y1, x2, y2))
            face_center = ((x1 + x2) / 2, (y1 + y2) / 2)
            nearest = nearest_bubble_for_face(face_center, per_page[pi])
            face_records.append({
                "id": f"F{face_id}",
                "page": pi,
                "box": (x1, y1, x2, y2),
                "conf": det["conf"],
                "crop": crop,
                "nearest_bubble": nearest,
            })
            face_id += 1
    face_t = time.monotonic() - t0
    log.info("faces: %.1fs, %d detections", face_t, len(face_records))

    if not face_records:
        log.warning("no faces detected; aborting probe")
        return

    gallery = build_face_gallery(
        [(r["id"], r["crop"]) for r in face_records],
        cols=4, tile=256, pad=6,
    )
    gallery_path = OUT / "probe_faces_gallery.jpg"
    gallery.save(gallery_path, quality=88, optimize=True)
    log.info("gallery: %s, %d KB",
             gallery.size, gallery_path.stat().st_size // 1024)

    # Vision call with BOTH images.
    provider = make_vision_provider(config)
    sb_uri = "data:image/jpeg;base64," + base64.b64encode(sb_path.read_bytes()).decode()
    gallery_uri = "data:image/jpeg;base64," + base64.b64encode(gallery_path.read_bytes()).decode()

    user = build_user_with_faces(flat, face_records, target_lang="vi")
    msgs = [Message.system(SYSTEM_AUGMENTED), Message.user_parts([
        ContentPart.of_text(user),
        ContentPart.of_image(sb_uri),
        ContentPart.of_image(gallery_uri),
    ])]

    log.info("calling vision (2 images, %d bubbles, %d faces)…",
             len(flat), len(face_records))
    t0 = time.monotonic()
    resp = await provider.call(msgs, [])
    elapsed = time.monotonic() - t0
    text = resp.text or ""
    log.info("vision: %.1fs, %d chars", elapsed, len(text))

    parsed = parse_response(text)
    speakers = parsed["speakers"]
    unknowns = sum(1 for v in speakers.values() if v.lower() == "unknown")
    named = {v for v in speakers.values() if v.lower() not in ("unknown", "sfx", "narrator")}

    log.info("characters: %d, speakers: %d/%d (%d unknown, %d distinct), "
             "noise: %d, address: %d",
             len(parsed["characters"]),
             len(speakers), len(flat), unknowns, len(named),
             len(parsed["noise"]), len(parsed["address"]))

    # Score vs combined-probe baseline.
    from probe_deterministic import COMBINED_GT
    correct = 0
    wrong = 0
    for b in flat:
        gt = COMBINED_GT.get(b["key"])
        pred = speakers.get(b["key"], "unknown")
        if gt is None:
            continue
        if gt == pred:
            correct += 1
        elif gt == "unknown" and pred in ("unknown",):
            correct += 1
        elif gt == "sfx" and pred in ("sfx",):
            correct += 1
        else:
            wrong += 1
    log.info("agreement with combined baseline: %d/%d (%.0f%%)",
             correct, len(flat), correct / len(flat) * 100)

    report = [
        "# Probe — storyboard + face gallery (idea 3)",
        "",
        f"- model: {config.vision_agent.model}",
        f"- pages: {reader.page_count}, bubbles: {len(flat)}, faces: {len(face_records)}",
        f"- storyboard: {sb.size[0]}x{sb.size[1]}, {sb_path.stat().st_size // 1024} KB",
        f"- gallery: {gallery.size[0]}x{gallery.size[1]}, {gallery_path.stat().st_size // 1024} KB",
        f"- scan: {scan_t:.1f}s, face detect: {face_t:.1f}s, vision: {elapsed:.1f}s",
        f"- response: {len(text)} chars",
        f"- characters: {len(parsed['characters'])}, "
        f"speakers named: {len(speakers) - unknowns}/{len(flat)} "
        f"({len(named)} distinct), noise: {len(parsed['noise'])}",
        f"- agreement vs combined baseline: {correct}/{len(flat)} "
        f"({correct/len(flat)*100:.0f}%)",
        "",
        "## Characters",
        "",
    ]
    for c in parsed["characters"]:
        report.append(f"- **{c['name']}** (gender: {c['gender']}, role: {c['role'] or '—'})")
    report.append("")
    report.append("## Speakers")
    report.append("")
    report.append("| key | page | text | baseline | with-faces |")
    report.append("|---|---|---|---|---|")
    for b in flat:
        sp = speakers.get(b["key"], "—")
        gt = COMBINED_GT.get(b["key"], "—")
        mark = ""
        if gt != "—":
            mark = " ✓" if sp == gt else " ✗"
        tx = b["text"].replace("|", "\\|").replace("\n", " ")[:50]
        report.append(f"| `{b['key']}` | {b['page']} | {tx!r} | {gt} | {sp}{mark} |")
    report.append("")
    report.append("## Face detections")
    report.append("")
    report.append("| id | page | conf | nearest bubble |")
    report.append("|---|---|---|---|")
    for r in face_records:
        report.append(f"| {r['id']} | {r['page']} | {r['conf']:.2f} | "
                      f"`{r['nearest_bubble'] or '—'}` |")
    report.append("")
    report.append("## Raw")
    report.append("```")
    report.append(text)
    report.append("```")

    out_path = OUT / "probe_faces.md"
    out_path.write_text("\n".join(report), "utf-8")
    log.info("wrote %s", out_path)

    print("\n=== SUMMARY ===")
    print(f"vision: {elapsed:.1f}s (2 images)")
    print(f"characters: {[c['name'] for c in parsed['characters']]}")
    print(f"speakers: {len(speakers) - unknowns}/{len(flat)} named, "
          f"{unknowns} unknown")
    print(f"agreement vs baseline (combined probe): "
          f"{correct}/{len(flat)} ({correct/len(flat)*100:.0f}%)")


if __name__ == "__main__":
    asyncio.run(main())
