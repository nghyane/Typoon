"""Audit container-polygon overlap from a probe ``raw.json``.

Reads ``debug-runs/<run-id>/raw.json`` (produced by ``scripts.probes.lens_group``),
computes per-pair polygon intersection for every BubbleGroup, classifies the
result, and writes:

    <run-dir>/overlap_audit.png   visual: containers + red intersections
    <run-dir>/overlap_report.json machine-readable verdict

Polygons emitted by ``_container_polygon`` are always convex (AABB / OBB /
inscribed ellipse), so Sutherland-Hodgman clipping is exact — no shapely
dependency.

Run:

    python -m scripts.probes.lens_group.audit_overlap debug-runs/happymh
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# ─── Polygon math (convex only) ──────────────────────────────────────────

Point = tuple[float, float]
Poly = list[Point]


def _polygon_area(poly: Poly) -> float:
    n = len(poly)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5


def _ensure_ccw(poly: Poly) -> Poly:
    """Sutherland-Hodgman expects CCW clip polygon (positive signed area)."""
    n = len(poly)
    s = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return poly if s > 0 else list(reversed(poly))


def _clip_polygon(subject: Poly, clip: Poly) -> Poly:
    """Sutherland-Hodgman: clip subject polygon by convex clip polygon."""
    out = list(subject)
    clip = _ensure_ccw(clip)
    cn = len(clip)
    for i in range(cn):
        if not out:
            return []
        a = clip[i]
        b = clip[(i + 1) % cn]
        ex, ey = b[0] - a[0], b[1] - a[1]   # edge vector

        def inside(p: Point) -> float:
            # left of directed edge a→b → positive (CCW = inside)
            return ex * (p[1] - a[1]) - ey * (p[0] - a[0])

        def intersect(p1: Point, p2: Point) -> Point:
            # Line p1→p2 vs line a→b
            x1, y1 = p1
            x2, y2 = p2
            x3, y3 = a
            x4, y4 = b
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-12:
                return p1
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
            return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

        new: Poly = []
        n = len(out)
        for j in range(n):
            p = out[j]
            q = out[(j + 1) % n]
            sp = inside(p)
            sq = inside(q)
            if sp >= 0:
                new.append(p)
                if sq < 0:
                    new.append(intersect(p, q))
            elif sq >= 0:
                new.append(intersect(p, q))
        out = new
    return out


def _aabb(poly: Poly) -> tuple[float, float, float, float]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def _aabb_disjoint(a: Poly, b: Poly) -> bool:
    ax1, ay1, ax2, ay2 = _aabb(a)
    bx1, by1, bx2, by2 = _aabb(b)
    return ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1


# ─── Audit ────────────────────────────────────────────────────────────────

# Verdict thresholds on max(intersection / min(area_i, area_j)).
WARN_RATIO = 0.01
FAIL_RATIO = 0.05


@dataclass(slots=True)
class _OverlapPair:
    i: int
    j: int
    area_px: float
    ratio_min: float
    shape_kind: tuple[str, str]


def _audit_groups(groups: list[dict]) -> tuple[list[_OverlapPair], float]:
    polys: list[Poly] = [[(float(x), float(y)) for x, y in g["polygon"]] for g in groups]
    areas = [_polygon_area(p) for p in polys]

    pairs: list[_OverlapPair] = []
    max_ratio = 0.0
    n = len(polys)
    for i in range(n):
        for j in range(i + 1, n):
            if _aabb_disjoint(polys[i], polys[j]):
                continue
            inter = _clip_polygon(polys[i], polys[j])
            if not inter:
                continue
            area = _polygon_area(inter)
            if area < 1.0:
                continue
            min_area = max(1.0, min(areas[i], areas[j]))
            ratio = area / min_area
            if ratio > max_ratio:
                max_ratio = ratio
            pairs.append(_OverlapPair(
                i=i, j=j, area_px=round(area, 2), ratio_min=round(ratio, 4),
                shape_kind=(groups[i]["shape_kind"], groups[j]["shape_kind"]),
            ))
    return pairs, max_ratio


def _verdict(max_ratio: float) -> str:
    if max_ratio >= FAIL_RATIO:
        return "fail"
    if max_ratio >= WARN_RATIO:
        return "warn"
    return "ok"


# ─── Visualisation ────────────────────────────────────────────────────────

_COLOR_CONTAINER = (60, 200, 80)      # green
_COLOR_OVERLAP   = (40, 40, 230)      # red (BGR)


def _render_audit(
    source: np.ndarray, groups: list[dict], pairs: list[_OverlapPair],
) -> np.ndarray:
    """Source + translucent container overlays; intersections in solid red."""
    base = source.copy()
    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    else:
        base = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)

    # Container fills (alpha 0.20)
    overlay = base.copy()
    for g in groups:
        pts = np.array([[int(x), int(y)] for x, y in g["polygon"]], dtype=np.int32)
        cv2.fillPoly(overlay, [pts], _COLOR_CONTAINER)
    cv2.addWeighted(overlay, 0.20, base, 0.80, 0, dst=base)

    # Container outlines
    for i, g in enumerate(groups):
        pts = np.array([[int(x), int(y)] for x, y in g["polygon"]], dtype=np.int32)
        cv2.polylines(base, [pts], True, _COLOR_CONTAINER, 1, cv2.LINE_AA)
        x1, y1, x2, y2 = g["bbox"]
        cv2.putText(
            base, f"#{i}", (int(x1) + 2, max(12, int(y1) - 2)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, _COLOR_CONTAINER, 1, cv2.LINE_AA,
        )

    # Intersection fills (solid red, alpha 0.6)
    if pairs:
        red = base.copy()
        polys: list[Poly] = [
            [(float(x), float(y)) for x, y in g["polygon"]] for g in groups
        ]
        for p in pairs:
            inter = _clip_polygon(polys[p.i], polys[p.j])
            if not inter:
                continue
            pts = np.array([[int(x), int(y)] for x, y in inter], dtype=np.int32)
            cv2.fillPoly(red, [pts], _COLOR_OVERLAP)
        cv2.addWeighted(red, 0.60, base, 0.40, 0, dst=base)

        for p in pairs:
            inter = _clip_polygon(polys[p.i], polys[p.j])
            if not inter:
                continue
            cx = int(sum(pt[0] for pt in inter) / len(inter))
            cy = int(sum(pt[1] for pt in inter) / len(inter))
            label = f"#{p.i}∩#{p.j} {int(p.area_px)}px ({p.ratio_min*100:.1f}%)"
            cv2.putText(
                base, label, (cx + 4, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, _COLOR_OVERLAP, 1, cv2.LINE_AA,
            )

    return base


# ─── CLI ──────────────────────────────────────────────────────────────────

def _audit_run(run_dir: Path) -> dict:
    raw_path    = run_dir / "raw.json"
    source_path = run_dir / "source.png"
    if not raw_path.exists():
        raise SystemExit(f"missing {raw_path}")
    if not source_path.exists():
        raise SystemExit(f"missing {source_path}")

    raw = json.loads(raw_path.read_text())
    groups = raw["groups"]
    pairs, max_ratio = _audit_groups(groups)
    verdict = _verdict(max_ratio)

    report = {
        "run":         run_dir.name,
        "page_size":   raw.get("page_size"),
        "groups":      len(groups),
        "overlap_pairs": len(pairs),
        "max_ratio":   round(max_ratio, 4),
        "verdict":     verdict,
        "thresholds":  {"warn": WARN_RATIO, "fail": FAIL_RATIO},
        "overlaps": [
            {
                "i": p.i, "j": p.j,
                "area_px":   p.area_px,
                "ratio_min": p.ratio_min,
                "shape_kind": list(p.shape_kind),
            }
            for p in sorted(pairs, key=lambda x: -x.ratio_min)
        ],
    }

    (run_dir / "overlap_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
    )

    source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if source is None:
        raise SystemExit(f"cannot read {source_path}")
    source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    audit = _render_audit(source, groups, pairs)
    cv2.imwrite(str(run_dir / "overlap_audit.png"), audit)

    return report


def main() -> None:
    ap = argparse.ArgumentParser(prog="audit_overlap")
    ap.add_argument(
        "run_dirs", type=Path, nargs="+",
        help="One or more probe run directories (containing raw.json + source.png).",
    )
    args = ap.parse_args()

    fail = False
    for d in args.run_dirs:
        report = _audit_run(d)
        marker = {"ok": " ", "warn": "!", "fail": "X"}[report["verdict"]]
        print(
            f"[{marker}] {report['run']}: "
            f"groups={report['groups']} "
            f"overlaps={report['overlap_pairs']} "
            f"max_ratio={report['max_ratio']*100:.2f}% "
            f"→ {report['verdict']}"
        )
        for o in report["overlaps"][:5]:
            print(
                f"      #{o['i']}∩#{o['j']}  "
                f"area={o['area_px']:>7.1f}px  "
                f"ratio={o['ratio_min']*100:>5.2f}%  "
                f"kinds={o['shape_kind']}"
            )
        if report["verdict"] == "fail":
            fail = True

    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
