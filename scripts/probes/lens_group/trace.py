"""Per-bubble diagnostic trace — print why blocks landed where they did.

Goes through each Comic-DETR bubble and reports:

  * occupancy (how many Lens blocks centre-overlap the bubble)
  * the texts those blocks carry (truncated)
  * empty bubbles flagged for recovery
  * Lens-rejected blocks that fell inside or near the bubble (filter
    reason from the detector pass)
"""

from __future__ import annotations

from typing import Iterable

from typoon.vision.contracts import DetectionResult, TextBlock


def _centre(b: tuple[int, int, int, int]) -> tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _inside(outer: tuple[int, int, int, int], pt: tuple[float, float]) -> bool:
    return outer[0] <= pt[0] <= outer[2] and outer[1] <= pt[1] <= outer[3]


def _members(
    bubble: tuple[int, int, int, int], blocks: Iterable[TextBlock],
) -> list[tuple[int, TextBlock]]:
    return [(i, b) for i, b in enumerate(blocks) if _inside(bubble, _centre(b.bbox))]


def _rejected_inside(
    bubble: tuple[int, int, int, int],
    rejected: Iterable[tuple[TextBlock, str]],
) -> list[tuple[TextBlock, str]]:
    return [(b, r) for (b, r) in rejected if _inside(bubble, _centre(b.bbox))]


def print_trace(detection: DetectionResult) -> None:
    blocks = list(detection.blocks)
    rejected = list(detection.rejected)
    bubbles = [r for r in detection.bubble_regions if r[0] == "bubble"]

    print("─── per-bubble trace ──────────────────────────────────────────")
    print(f"page={detection.page_size}  bubbles={len(bubbles)}  "
          f"lens_kept={len(blocks)}  lens_rejected={len(rejected)}")
    if rejected:
        print(f"  rejected reasons: {_reason_counts(rejected)}")
    print()

    empty: list[tuple[str, tuple[int, int, int, int], float]] = []
    for i, (_cls, bbox, conf) in enumerate(bubbles):
        members = _members(bbox, blocks)
        rej_in  = _rejected_inside(bbox, rejected)
        bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        marker = "EMPTY" if not members else f"{len(members)} block(s)"
        print(f"  bubble#{i:2d} {bw:3d}x{bh:3d} conf={conf:.2f}  bbox={bbox}  → {marker}")
        for j, b in members:
            text = (b.text or "").replace("\n", " ")
            print(f"      #{j:2d} {b.text_direction[:1].upper()} "
                  f"rot={b.rotation_deg:+.0f} txt={text[:40]!r}")
        for b, reason in rej_in:
            print(f"      REJ {reason:14s} bbox={b.bbox} txt={(b.text or '')[:40]!r}")
        if not members:
            empty.append((_cls, bbox, conf))
    print()
    if empty:
        print(f"empty bubbles ({len(empty)}) — recovery candidates:")
        for cls, bbox, conf in empty:
            print(f"  {cls} bbox={bbox} conf={conf:.2f}")
    else:
        print("no empty bubbles.")
    print("───────────────────────────────────────────────────────────────")


def _reason_counts(rejected) -> dict[str, int]:
    out: dict[str, int] = {}
    for _b, r in rejected:
        out[r] = out.get(r, 0) + 1
    return out
