"""POC — full pipeline: Lens detect + comic_detr + spatial join → erase → render.

Runs end-to-end on a single page:
  1. LensBlocksDetector (with comic_detr) → DetectionResult
  2. LensNativeGrouper → tuple[BubbleGroup]
  3. AOTGANEraser → wipe text from canvas
  4. typoon_render.render → fit + draw Vietnamese text into bubble outline

Validates the full chain after the spatial-join rewrite. Vietnamese
text is a hand-written translation per bubble — no LLM call.

Output to debug-runs/<probe>/full_pipeline/:
  01_source.png         original input
  02_groups.png         comic_detr + grouper overlay
  03_erase_mask.png     union of erase masks (what AOT sees)
  04_inpainted.png      after AOT-GAN
  05_rendered.png       final with VI text
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from typoon.models import ModelHub  # noqa: E402
from typoon.vision._backends.comic_detr import load_session  # noqa: E402
from typoon.vision.contracts import BubbleGroup  # noqa: E402
from typoon.vision.detectors.lens_blocks import LensBlocksDetector  # noqa: E402
from typoon.vision.erasers import AOTGANEraser  # noqa: E402
from typoon.vision.groupers.lens_native import LensNativeGrouper  # noqa: E402


# ─── Hand-typed VI translations per fixture ────────────────────────────────
# Keyed by a unique substring of the source text so the order doesn't matter.

_VI: dict[str, dict[str, str]] = {
    "lens_bubble_probe": {
        "奇怪": "Lạ thật, từ bao giờ ánh nắng trở nên mờ nhạt, mà cậu ấy lại càng rạng rỡ hơn?",
        "即使": "Dù có thất bại trong việc nuốt chửng Yêu Hoàng, cậu ấy vẫn không từ bỏ mình, mặt trời vẫn chiếu rọi mình.",
        "难道他才是": "Lẽ nào cậu ấy mới là mặt trời thật sự?",
        "他说": "Cậu ấy nói sẽ đưa mình đi xem khắp các dòng thời gian của những giới khác.",
        "我承诺": "Mình hứa sẽ đồng hành cùng cậu ấy tìm ra—",
        "一切都太晚了": "Tất cả đều đã muộn rồi…",
        "盒子里是一串": "Trong hộp là một chuỗi ngọc trai, coi như tín vật đính ước.",
        "你怎么": "Sao cậu lại…",
        "你:你怎么在这": "Sao… sao cậu lại ở đây?",
        "我:我刚才在练剑": "Ta vừa luyện kiếm xong.",
        "前辈": "Tiền bối.",
        "特来送与前辈": "Đặc biệt mang đến tặng tiền bối.",
        "这是我与海族": "Đây là dạ minh châu mà ta đánh cuộc thắng được từ thái tử Hải tộc.",
        "你当我是": "Cậu coi tôi là loại đàn bà không biết liêm sỉ sao?",
        "待我说服": "Chờ ta thuyết phục Tình muội.",
        "就娶你": "Sẽ cưới em.",
        "难道是因为": "Lẽ nào vì không thích chuỗi ngọc trai này?",
        "滚": "Cút!",
        "太晚了": "Muộn rồi…",
    },
    "lens_bubble_probe2": {
        "奇怪": "Lạ thật, từ bao giờ ánh nắng trở nên mờ nhạt, mà cậu ấy lại càng rạng rỡ hơn?",
        "即使": "Dù có thất bại trong việc nuốt chửng Yêu Hoàng, cậu ấy vẫn không từ bỏ mình, mặt trời vẫn chiếu rọi mình.",
        "难道他才是": "Lẽ nào cậu ấy mới là mặt trời thật sự? Lẽ nào… cậu ấy mới là ý nghĩa mình tìm kiếm?",
        "他说": "Cậu ấy nói sẽ đưa mình đi xem khắp các dòng thời gian của những giới khác.",
        "我承诺": "Mình hứa sẽ đồng hành cùng cậu ấy tìm ra—",
    },
    "lens_bubble_probe3": {
        "你不觉得最近": "Cậu không thấy gần đây Khải-chan có gì đó lạ lạ à?",
        "该不会在偷偷": "Lẽ nào đang lén lút hẹn hò?",
        "周一综合症": "Bệnh thứ hai à?",
        "不仅如此": "Không chỉ vậy đâu, gần đây thầy cũng thấy lạ lắm.",
        "不会吧": "Lẽ nào…? Khải-chan vốn thông minh, nghiêm túc… lại có thể…",
        "んむっ、": "Hmph…",
        "んむっ": "Hmph.",
        "んぐっ": "Ưgh.",
        "ぎゅっ": "Ôm chặt…",
        "コッ": "Cộc.",
        "凯伊酱…?": "Khải-chan…?",
        "じゅく": "Chín…",
        "ザワ": "Xào xạc.",
    },
}


def _resolve_vi(src_text: str, fixture: str) -> str:
    table = _VI.get(fixture, {})
    for key, vi in table.items():
        if key in src_text:
            return vi
    # No VI mapping — fall back to a placeholder so the renderer still
    # runs and we can see the fit area on the debug overlay. Length
    # roughly mirrors the source so fit behaviour is representative.
    src = (src_text or "").strip().replace("\n", " ")
    if not src:
        return ""
    return f"[VI] {src[:60]}"


# ─── Helpers ───────────────────────────────────────────────────────────────


def _to_rgba(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    return np.dstack([img, np.full((h, w), 255, dtype=np.uint8)])


def _draw_groups(img: np.ndarray, groups: tuple[BubbleGroup, ...]) -> np.ndarray:
    out = img.copy()
    for i, g in enumerate(groups):
        x1, y1, x2, y2 = g.bbox
        color = (255, 60, 60) if g.text_direction == "vertical" else (60, 200, 60)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        label = f"#{i} {g.text_direction[:4]} {g.shape_kind[:4]}"
        cv2.putText(out, label, (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return out


def _build_erase_visual(h: int, w: int, groups: tuple[BubbleGroup, ...]) -> np.ndarray:
    union = np.zeros((h, w), dtype=np.uint8)
    for g in groups:
        for em in g.erase_masks:
            mh, mw = em.image.shape[:2]
            x1, y1 = max(0, em.x), max(0, em.y)
            x2, y2 = min(w, em.x + mw), min(h, em.y + mh)
            sx1, sy1 = x1 - em.x, y1 - em.y
            sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)
            if x2 > x1 and y2 > y1:
                union[y1:y2, x1:x2] = np.maximum(
                    union[y1:y2, x1:x2], em.image[sy1:sy2, sx1:sx2],
                )
    return cv2.cvtColor(union, cv2.COLOR_GRAY2RGB)


def _save_fit_debug(
    path: Path,
    base_rgb: np.ndarray,
    triples: list,
    bubble_infos: list,
    groups: tuple[BubbleGroup, ...],
    hints: list,
    *,
    lens_blocks,
    comic_regions,
) -> None:
    """Overlay box sources on the ORIGINAL page so we can SEE the gap.

    Drawn on the untouched source (not the rendered output) so VI glyphs
    don't obscure the geometry we're trying to debug.

    Layers:
      - BLUE   Comic-DETR `bubble`
      - PURPLE Comic-DETR `text_bubble`
      - OLIVE  Comic-DETR `text_free`
      - GREEN  BubbleGroup.bbox  (fit area the renderer used)
      - YELLOW renderer drawable rect
      - RED    overflow flag
    """
    canvas = base_rgb.copy()
    if canvas.shape[-1] == 4:
        canvas = canvas[..., :3].copy()
    H, W = canvas.shape[:2]

    # Layer 1 — Comic-DETR regions (outlines only, no fills)
    cls_color = {
        "bubble":      (60,  60, 255),
        "text_bubble": (180, 120, 255),
        "text_free":   (200, 200,  60),
    }
    for cls, bbox, conf in comic_regions:
        x1, y1, x2, y2 = bbox
        color = cls_color.get(cls, (128, 128, 128))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1)

    # Layer 2 — BubbleGroup fit area + renderer rect + overflow flag
    for (g, _vi), info, hint in zip(triples, bubble_infos, hints):
        x1, y1, x2, y2 = g.bbox
        rx, ry, rw, rh = info.rect
        ovf = info.overflow
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.rectangle(
            canvas,
            (int(rx), int(ry)), (int(rx + rw), int(ry + rh)),
            (255, 220, 0), 1,
        )
        if ovf:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 60, 60), 3)

        hint_fs = hint.font_size_px if hint is not None else 0
        label = f"fs={info.font_size_px}px h={hint_fs}{' OVF' if ovf else ''}"
        ty = max(12, y1 - 4)
        cv2.putText(canvas, label, (x1, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, label, (x1, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Legend (top-left)
    legend = [
        ("Comic-DETR bubble",      (60,  60, 255)),
        ("Comic-DETR text_bubble", (180, 120, 255)),
        ("Comic-DETR text_free",   (200, 200,  60)),
        ("BubbleGroup (fit area)", (  0, 220,   0)),
        ("Renderer drawable rect", (255, 220,   0)),
        ("Overflow flag",          (255,  60,  60)),
    ]
    pad = 6
    line_h = 18
    box_w = 260
    box_h = line_h * len(legend) + pad * 2
    overlay = canvas[:box_h, :box_w].copy()
    cv2.rectangle(canvas, (0, 0), (box_w, box_h), (255, 255, 255), -1)
    canvas[:box_h, :box_w] = cv2.addWeighted(overlay, 0.30, canvas[:box_h, :box_w], 0.70, 0)
    for i, (name, color) in enumerate(legend):
        yy = pad + i * line_h + 12
        cv2.rectangle(canvas, (pad, yy - 9), (pad + 14, yy + 1), color, -1)
        cv2.putText(
            canvas, name, (pad + 20, yy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA,
        )

    Image.fromarray(canvas).save(path)





def _build_bubble_mask(
    h: int, w: int, groups: tuple[BubbleGroup, ...],
) -> np.ndarray:
    """Page-sized uint8 labelled mask: unique non-zero id per group.

    The renderer's BubbleIndex keys on the label, so each bubble must
    have a distinct ID. Painting all bubbles with the same value made
    expand_area treat the page as one giant container, sprawling text
    everywhere — see ``_rasterize_bubble_mask`` for the production
    implementation.
    """
    ordered = sorted(
        groups,
        key=lambda g: (g.bbox[2] - g.bbox[0]) * (g.bbox[3] - g.bbox[1]),
        reverse=True,
    )[:254]
    mask = np.zeros((h, w), dtype=np.uint8)
    for label_id, g in enumerate(ordered, start=1):
        x1, y1, x2, y2 = g.bbox
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = label_id
    return mask


# ─── Driver ────────────────────────────────────────────────────────────────


async def _run_fixture(name: str) -> None:
    import typoon_render
    src = ROOT / "debug-runs" / name / "source.png"
    out_dir = ROOT / "debug-runs" / name / "full_pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {name} ===")

    img = np.asarray(Image.open(src).convert("RGB"))
    Image.fromarray(img).save(out_dir / "01_source.png")
    H, W = img.shape[:2]

    # Detector + grouper (production wiring)
    hub = ModelHub(ROOT / "models")
    comic = load_session(hub.resolve_comic_detr())
    detector = LensBlocksDetector(comic_detr=comic)
    grouper = LensNativeGrouper()

    print("  detecting…")
    detection = await detector.detect(img, lang=None)
    groups = await grouper.group(img, detection, lang=None)
    print(f"  groups: {len(groups)} (regions={len(detection.bubble_regions)})")

    Image.fromarray(_draw_groups(img, groups)).save(out_dir / "02_groups.png")

    # Erase mask visualisation
    erase_vis = _build_erase_visual(H, W, groups)
    Image.fromarray(erase_vis).save(out_dir / "03_erase_mask.png")

    # AOT erase
    canvas = _to_rgba(img.copy())
    eraser = AOTGANEraser(models_dir=hub.dir)
    all_erase = tuple(m for g in groups for m in g.erase_masks)
    print(f"  erasing ({len(all_erase)} masks)…")
    await eraser.erase(canvas, all_erase)
    Image.fromarray(canvas[..., :3]).save(out_dir / "04_inpainted.png")

    # Resolve VI translations + build render inputs
    triples = []
    for g in groups:
        vi = _resolve_vi(g.text, name)
        if not vi:
            continue
        triples.append((g, vi))
    print(f"  rendering: {len(triples)} / {len(groups)} groups have VI text")

    polygons = [list(map(list, g.polygon)) for g, _ in triples]
    texts = [vi for _, vi in triples]
    hints = []
    for g, _ in triples:
        ts = g.typesetting
        if ts is None:
            hints.append(None)
        else:
            hints.append(typoon_render.typoon_render.TypesettingHint(
                font_size_px=ts.font_size_px,
                line_count=ts.line_count,
                avg_chars_per_line=ts.avg_chars_per_line,
                text_direction=g.text_direction,
            ))

    bubble_mask = _build_bubble_mask(H, W, groups)
    # The Rust renderer's BubbleIndex.expand_area assumes non-overlapping
    # labelled regions (designed for instance-seg masks). Our axis-aligned
    # bubble bboxes from comic_detr overlap (nested + neighbouring), so a
    # text polygon inside bubble X may have MORE pixels of overlapping
    # bubble Y inside its bounding box → BubbleIndex picks Y → expand_area
    # returns Y's extent → text sprawls. Polygon already is the parent
    # bubble outline (set in _spatial_join) so we pass None and let the
    # renderer use from_polygon_insets directly on the polygon.
    _ = bubble_mask  # kept for visual diff in earlier debug runs

    result = await asyncio.to_thread(
        typoon_render.typoon_render.render,
        canvas, polygons, texts, W, hints, None,
    )
    final = result.image
    Image.fromarray(final[..., :3]).save(out_dir / "05_rendered.png")
    overflow = sum(1 for b in result.bubbles if b.overflow)
    print(f"  rendered: overflow on {overflow}/{len(result.bubbles)} bubbles")

    # Per-bubble fit debug overlay — drawn on top of the rendered page so
    # you can see at a glance whether text is escaping the box, and which
    # box the fitter actually used. Avoids blind debugging when something
    # looks "tràn".
    _save_fit_debug(
        out_dir / "06_fit_debug.png", img, triples, result.bubbles, groups, hints,
        lens_blocks=detection.blocks,
        comic_regions=detection.bubble_regions,
    )

    # Per-bubble fit summary — full trace for visual+log debugging.
    crops_dir = out_dir / "groups"
    crops_dir.mkdir(exist_ok=True)
    H_img, W_img = img.shape[:2]
    print(f"  page={W_img}x{H_img}")
    for i, ((g, vi), b) in enumerate(zip(triples, result.bubbles)):
        gidx = groups.index(g)
        bw = g.bbox[2] - g.bbox[0]
        bh = g.bbox[3] - g.bbox[1]
        # Detect fit bbox spilling off page (renderer crops silently)
        spill_l = max(0, -g.bbox[0])
        spill_t = max(0, -g.bbox[1])
        spill_r = max(0, g.bbox[2] - W_img)
        spill_b = max(0, g.bbox[3] - H_img)
        spill = sum([spill_l, spill_t, spill_r, spill_b])
        # Find member Lens blocks (centers inside group bbox)
        members = [
            blk for blk in detection.blocks
            if g.bbox[0] <= (blk.bbox[0] + blk.bbox[2]) / 2 <= g.bbox[2]
            and g.bbox[1] <= (blk.bbox[1] + blk.bbox[3]) / 2 <= g.bbox[3]
        ]
        # Lens raw union
        if members:
            lx1 = min(m.bbox[0] for m in members)
            ly1 = min(m.bbox[1] for m in members)
            lx2 = max(m.bbox[2] for m in members)
            ly2 = max(m.bbox[3] for m in members)
            lens_dim = f"{lx2-lx1}x{ly2-ly1}"
        else:
            lx1 = ly1 = lx2 = ly2 = 0
            lens_dim = "-"
        # Comic-DETR regions overlapping the group bbox
        overlaps = []
        for cls, rb, conf in detection.bubble_regions:
            inter_w = max(0, min(g.bbox[2], rb[2]) - max(g.bbox[0], rb[0]))
            inter_h = max(0, min(g.bbox[3], rb[3]) - max(g.bbox[1], rb[1]))
            if inter_w * inter_h > 0:
                overlaps.append((cls, rb, conf, inter_w * inter_h))
        overlaps.sort(key=lambda x: -x[3])
        # Path inference: compare g.bbox vs Lens union ratio
        w_ratio = bw / max(1, lx2 - lx1)
        h_ratio = bh / max(1, ly2 - ly1)
        path = "?"
        if w_ratio < 1.05 and h_ratio < 1.05:
            path = "tight≈lens"
        elif w_ratio > 2.0 or h_ratio > 2.0:
            path = "free-pad"
        else:
            path = "promoted"

        print(f"\n  [#{gidx:2d}] path={path:>10}  fs={b.font_size_px:3d}px  "
              f"fit={bw}x{bh}@({g.bbox[0]},{g.bbox[1]})  "
              f"lens={lens_dim}@({lx1},{ly1})  ovf={b.overflow}"
              f"{'  SPILL=L'+str(spill_l)+'/T'+str(spill_t)+'/R'+str(spill_r)+'/B'+str(spill_b) if spill else ''}")
        print(f"        VI={vi[:60]!r}")
        print(f"        members={len(members)}")
        for j, m in enumerate(members):
            mt = (m.text or "").strip().replace("\n", " ")[:40]
            print(f"          m{j}: bbox={m.bbox} dir={m.text_direction} {mt!r}")
        print(f"        comic-overlap (top3):")
        for cls, rb, conf, inter in overlaps[:3]:
            print(f"          {cls:11s} {rb} conf={conf:.2f} inter={inter}")

        # Per-group crop with overlays for visual review
        pad = 20
        cx1 = max(0, g.bbox[0] - pad)
        cy1 = max(0, g.bbox[1] - pad)
        cx2 = min(W_img, g.bbox[2] + pad)
        cy2 = min(H_img, g.bbox[3] + pad)
        crop = img[cy1:cy2, cx1:cx2].copy()
        # Draw Comic-DETR overlap regions
        for cls, rb, conf, _ in overlaps:
            color = {"bubble": (60,60,255), "text_bubble": (180,120,255),
                     "text_free": (200,200,60)}.get(cls, (128,128,128))
            cv2.rectangle(crop,
                          (rb[0]-cx1, rb[1]-cy1), (rb[2]-cx1, rb[3]-cy1),
                          color, 1)
        # Lens members
        for m in members:
            mb = m.bbox
            cv2.rectangle(crop, (mb[0]-cx1, mb[1]-cy1),
                          (mb[2]-cx1, mb[3]-cy1), (0, 200, 200), 1)
        # Final fit bbox
        cv2.rectangle(crop, (g.bbox[0]-cx1, g.bbox[1]-cy1),
                      (g.bbox[2]-cx1, g.bbox[3]-cy1), (0, 220, 0), 2)
        Image.fromarray(crop).save(crops_dir / f"{gidx:02d}_{path}.png")

    print(f"  artefacts → {out_dir}")


async def main() -> None:
    fixtures = sys.argv[1:] or ["lens_bubble_probe", "lens_bubble_probe2", "lens_bubble_probe3"]
    for name in fixtures:
        await _run_fixture(name)


if __name__ == "__main__":
    asyncio.run(main())
