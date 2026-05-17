"""Spatial join — group Lens TextBlocks into BubbleGroups.

Design:

* Comic-DETR regions decide grouping — they tell us which Lens blocks
  belong to the same balloon / caption when Lens over-splits.
  Overlapping regions are first reduced per cluster (single-link by
  IoU > 0.7) with class-aware precedence:

    text_free  > bubble  > text_bubble

  ``text_free`` wins whenever it appears in the cluster (3-class
  overlap is the canonical caption pattern: bubble + text_bubble +
  text_free → caption, not a balloon). ``text_bubble``-only clusters
  are dropped — Lens word geometry is tighter.

* Container box per group comes from Lens: union of member ``word``
  bboxes expanded by ``0.5 × median glyph short side``. Using the
  short side keeps padding proportional to glyph extent for both
  horizontal and tategaki text. The render stage receives this
  container as the polygon directly — no border scanning, no
  inscribed-ellipse heuristics.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from ..contracts import (
    BubbleGroup,
    TextBlock,
    TextMask,
    TypesettingHint,
)
from ._classify import PROFILES, classify_block


__all__ = ["spatial_join", "block_to_group"]


Bbox = tuple[int, int, int, int]


# Comic-DETR class names participating in clustering. ``text_bubble`` is
# kept so we can detect the "3-class cluster = caption" signal, but it
# never wins as a representative on its own.
_CLUSTER_CLASSES = frozenset({"bubble", "text_free", "text_bubble"})

# Overlap above this IoU collapses two anchors to the higher-confidence one.
_ANCHOR_DEDUP_IOU = 0.7

# Container padding = factor × median glyph short side of group members.
# Per shape:
#   dialogue → 0.20: translated text (VI/EN) is usually longer than the JP
#             source, so the bubble needs breathing room for re-wrap.
#   burst    → 0.08: SFX rarely translate into longer strings; the pad
#             only has to cover anti-aliased glyph edges. Keeping it tied
#             to ``_MASK_PAD_FACTOR`` guarantees container ⊇ mask without
#             a separate clamp. Tilted SFX with the old 0.20 factor
#             produced visible cross-group overlap (audit happymh2 #3∩#4).
_CONTAINER_PAD_FACTOR: dict[str, float] = {
    "dialogue": 0.20,
    "burst":    0.08,
}

# Floor padding when glyph size cannot be derived (no line geometry).
_CONTAINER_PAD_MIN_PX = 4

# Below this rotation magnitude the container collapses to an
# axis-aligned rectangle (cheaper, identical visual result).
_ROTATION_AABB_DEG = 1.0

# Mask dilation: enough to cover Lens word-bbox under-coverage (Lens
# tends to draw the AABB ~1-2 px inside the actual glyph stripe,
# especially on diagonal strokes and ascenders/diacritics), plus
# anti-aliased glyph edges. Was 0.08 — too tight on small fonts;
# glyph extremities regularly poked outside the mask leaving a faint
# ghost after AOT inpaint.
#
# Per shape, mirroring `_CONTAINER_PAD_FACTOR` so mask never exceeds the
# render polygon (cross-group overlap regression — audit happymh2 #3∩#4
# came from mask_pad > container_pad on tilted SFX).
_MASK_PAD_FACTOR: dict[str, float] = {
    "dialogue": 0.20,
    "burst":    0.08,
}
_MASK_PAD_MIN_PX = 2

# ─── text_bubble container hint (Layer B) ────────────────────────────────
#
# Comic-DETR ``text_bubble`` is a learned axis-aligned rect tagging the
# "safe text area" inside a speech balloon outline. When it co-exists
# with a ``bubble`` cluster mate it is a strictly better dialogue
# container than ``word_union + pad``: it accounts for the rounded
# balloon interior the glyph stripe cannot see.
#
# Union-expand policy: the hint is merged with ``word_union`` before
# acceptance, guaranteeing the renderer never clips a glyph Lens
# actually read. A separate area cap keeps DETR's occasional over-shoots
# (predictions that bleed onto neighbouring captions) from producing a
# container many times larger than the visible text.
# Reject hint when DETR over-shot onto a neighbouring balloon: the
# merged hint must not contain the centre of any block outside the
# group. Earlier code used a 4× area ratio cap which fired false-
# positives on legitimate balloons with sparse text (single-character
# interjections in a normal-sized balloon → ratio area is naturally
# 4-10×; legit, not over-shoot). The centre-containment check fires
# precisely on the real over-shoot pattern (hint bleeds onto the next
# bubble's glyphs) without penalising sparse-text balloons.
#
# `_HINT_AREA_RATIO_MAX` is kept as an emergency belt-and-braces cap:
# even when no other blocks fall inside the hint, an absurdly large
# hint (e.g. DETR returning the full page) is still rejected.
_HINT_AREA_RATIO_MAX = 100.0  # generous emergency-only cap

# Pairing bubble ↔ text_bubble cannot use IoU — the outer ``bubble`` box
# bounds the tail too, so a tight inner rect typically peaks around
# IoU ~ 0.5 with its parent. ``text_bubble`` IS the inner rect though,
# so containment of text_bubble inside bubble is near-1.
_HINT_CONTAINMENT_MIN = 0.80

# Inscribed-ellipse sampling for dialogue containers fitted in a balloon.
# 24 verts is smooth enough at print resolution (< 1 px deviation from a
# perfect ellipse for bubbles up to ~600 px wide) and keeps Rust render
# polygon hand-off cheap. Skipped when the bbox aspect is extreme — a
# very wide or very tall ellipse degenerates into a strip that wastes
# horizontal/vertical room compared to a plain rect.
_ELLIPSE_VERTICES   = 24
_ELLIPSE_ASPECT_MIN = 0.4
_ELLIPSE_ASPECT_MAX = 2.5


Polygon = tuple[tuple[float, float], ...]


# ─── Anchor record ────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _Anchor:
    cls:        str
    bbox:       Bbox
    conf:       float
    inner_bbox: Bbox | None = None   # text_bubble rect when cls=='bubble'


# ─── Geometry helpers ─────────────────────────────────────────────────────


def _area(b: Bbox) -> int:
    return max(1, (b[2] - b[0]) * (b[3] - b[1]))


def _iou(a: Bbox, b: Bbox) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    return inter / (_area(a) + _area(b) - inter)


def _center(b: Bbox) -> tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _contains_center(outer: Bbox, inner: Bbox) -> bool:
    cx, cy = _center(inner)
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def _clip(bbox: Bbox, page: tuple[int, int]) -> Bbox:
    W, H = page
    return (max(0, bbox[0]), max(0, bbox[1]),
            min(W, bbox[2]), min(H, bbox[3]))


def _union(bboxes: list[Bbox]) -> Bbox:
    return (
        min(b[0] for b in bboxes),
        min(b[1] for b in bboxes),
        max(b[2] for b in bboxes),
        max(b[3] for b in bboxes),
    )


def _bbox_to_polygon(b: Bbox) -> tuple[tuple[float, float], ...]:
    x1, y1, x2, y2 = b
    return ((float(x1), float(y1)), (float(x2), float(y1)),
            (float(x2), float(y2)), (float(x1), float(y2)))


# ─── Anchor preparation ───────────────────────────────────────────────────


def _dedup_anchors(
    anchors: list[_Anchor],
    all_text_bubbles: list[_Anchor] | None = None,
) -> list[_Anchor]:
    """Collapse overlapping anchors into one per cluster, class-aware.

    Comic-DETR routinely emits 2-3 boxes (``bubble`` + ``text_bubble`` +
    ``text_free``) covering almost the same region. The class mix tells
    us what the region actually is:

      * A real speech balloon registers as ``bubble`` (often + ``text_bubble``).
      * A free-floating caption registers as ``text_free`` *and* gets
        spurious ``bubble`` / ``text_bubble`` companions because the DETR
        heads are independent. Whenever ``text_free`` is in the cluster
        we trust that signal and treat the whole cluster as a caption,
        regardless of confidence.
      * Sometimes DETR misses the outer ``bubble`` outline entirely and
        only fires ``text_bubble`` for a real speech balloon's safe text
        area (mangabuzz 374/1/9: two tategaki strands "離せっ" + "一人で
        歩けるっ" share a single ``text_bubble`` but no ``bubble``). Without
        text_bubble as a grouping fallback those strands become
        adjacent singletons whose containers overlap.

    Algorithm:
      1. Build clusters via single-link overlap (IoU > 0.7).
      2. Per cluster, pick the dominant class:
           ``text_free`` > ``bubble`` > ``text_bubble``.
         Higher tiers win regardless of confidence — class beats score.
      3. Emit one anchor per cluster using the bbox of the highest-conf
         region of the winning class.
      4. For ``bubble`` winners, scan ``all_text_bubbles`` for the best
         inner-rect match by containment (IoU is the wrong metric — the
         outer ``bubble`` box always includes the balloon tail, so even
         a perfect inner rect peaks at IoU ~ 0.5).
      5. For standalone ``text_bubble`` winners (no ``bubble`` mate),
         reuse the bbox itself as the container hint — it IS the inner
         safe-text rect, same role text_bubble plays inside a normal
         bubble+text_bubble cluster. Existing
         ``_merge_hint_with_word_union`` expansion still applies so the
         hint covers every glyph Lens read.
    """
    clusters = _cluster_by_overlap(anchors, _ANCHOR_DEDUP_IOU)
    out: list[_Anchor] = []
    for members in clusters:
        winner = _pick_cluster_anchor(members)
        if winner is None:
            continue
        if winner.cls == "bubble" and all_text_bubbles:
            inner = _best_inner_rect(winner.bbox, all_text_bubbles)
            if inner is not None:
                winner = _Anchor(
                    cls=winner.cls, bbox=winner.bbox, conf=winner.conf,
                    inner_bbox=inner,
                )
        elif winner.cls == "text_bubble":
            # Standalone text_bubble: bbox doubles as container hint so
            # the ellipse/merged-hint path fires (same shape behaviour as
            # a bubble+text_bubble cluster).
            winner = _Anchor(
                cls=winner.cls, bbox=winner.bbox, conf=winner.conf,
                inner_bbox=winner.bbox,
            )
        out.append(winner)
    return out


def _containment(inner: Bbox, outer: Bbox) -> float:
    """Fraction of ``inner`` area that sits inside ``outer``."""
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0
    overlap = (ix2 - ix1) * (iy2 - iy1)
    return overlap / max(1, _area(inner))


def _best_inner_rect(
    bubble_bbox: Bbox, candidates: list[_Anchor],
) -> Bbox | None:
    """Pick the highest-confidence text_bubble that sits inside ``bubble_bbox``.

    Acceptance: ``_HINT_CONTAINMENT_MIN`` of the candidate's own area
    falls inside the outer bubble rect. Ties broken by confidence.
    """
    matches = [
        c for c in candidates
        if _containment(c.bbox, bubble_bbox) >= _HINT_CONTAINMENT_MIN
    ]
    if not matches:
        return None
    return max(matches, key=lambda c: c.conf).bbox


def _cluster_by_overlap(anchors: list[_Anchor], iou_thr: float) -> list[list[_Anchor]]:
    """Single-link clustering: two anchors join iff their IoU exceeds ``iou_thr``."""
    n = len(anchors)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if _iou(anchors[i].bbox, anchors[j].bbox) > iou_thr:
                union(i, j)

    buckets: dict[int, list[_Anchor]] = {}
    for i in range(n):
        buckets.setdefault(find(i), []).append(anchors[i])
    return list(buckets.values())


def _pick_cluster_anchor(members: list[_Anchor]) -> _Anchor | None:
    """Pick one representative anchor for a cluster.

    Class precedence: ``text_free`` > ``bubble`` > ``text_bubble``.
    Within the winning class, the highest-confidence region wins.
    Returns ``None`` only when the cluster is empty (impossible by
    construction; defensive).

    Standalone ``text_bubble`` clusters (no bubble / no text_free mate)
    DO win — DETR sometimes misses the outer balloon outline and only
    fires text_bubble for the inner safe-text rect. Without this
    fallback two tategaki strands sharing one balloon become adjacent
    singletons whose containers visibly overlap. See
    ``_dedup_anchors`` algorithm comment.
    """
    for cls in ("text_free", "bubble", "text_bubble"):
        same = [m for m in members if m.cls == cls]
        if same:
            return max(same, key=lambda m: m.conf)
    return None


# ─── Lens-derived container ───────────────────────────────────────────────


def _word_union(members: list[TextBlock]) -> Bbox:
    """Tightest box covering every word bbox across all members.

    Falls back to the paragraph bbox union when Lens did not emit word
    geometry for any member (rare; usually only fully decoration-only
    blocks, which the detector already filtered out).
    """
    word_boxes = [w.bbox for m in members for w in m.words]
    if word_boxes:
        return _union(word_boxes)
    return _union([m.bbox for m in members])


def _median_glyph_size(members: list[TextBlock]) -> int:
    """Approximate glyph size = shorter side of each line bbox.

    For horizontal text the shorter side is the line height (font
    ascent+descent). For vertical/tategaki text the shorter side is the
    line width (column width ≈ one glyph wide). Using the shorter side
    keeps padding proportional to actual glyph extent in both
    directions; using line height alone over-pads tategaki by the
    column length.
    """
    sizes: list[int] = []
    for m in members:
        for ln in m.lines:
            w = ln.bbox[2] - ln.bbox[0]
            h = ln.bbox[3] - ln.bbox[1]
            short = min(w, h)
            if short > 0:
                sizes.append(short)
    if not sizes:
        return 0
    return int(statistics.median(sizes))


def _group_rotation(members: list[TextBlock]) -> float:
    """Max-abs rotation across members — keeps the container wide enough
    to bound every glyph when blocks tilt different amounts."""
    if not members:
        return 0.0
    return max(members, key=lambda m: abs(m.rotation_deg)).rotation_deg


def _oriented_rect(
    x1: float, y1: float, x2: float, y2: float, rot_deg: float,
) -> Polygon:
    """4 corners of the axis-aligned rect, rotated about its centre.

    Skips rotation when ``|rot| < _ROTATION_AABB_DEG`` for cheaper output
    on axis-aligned blocks.
    """
    import math
    if abs(rot_deg) < _ROTATION_AABB_DEG:
        return (
            (float(x1), float(y1)), (float(x2), float(y1)),
            (float(x2), float(y2)), (float(x1), float(y2)),
        )
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    rad = math.radians(rot_deg)
    cos_t, sin_t = math.cos(rad), math.sin(rad)
    local = ((x1 - cx, y1 - cy), (x2 - cx, y1 - cy),
             (x2 - cx, y2 - cy), (x1 - cx, y2 - cy))
    return tuple(
        (cx + lx * cos_t - ly * sin_t,
         cy + lx * sin_t + ly * cos_t)
        for lx, ly in local
    )


def _is_column_layout(members: list[TextBlock]) -> bool:
    """True when the words across all members stack vertically.

    Collect all word centres, fit the principal axis: if the dominant
    spread is top-to-bottom (|dy| >= |dx|) the group is a character
    column and the AABB from Lens is already the right erase/container
    shape — rotating it further would double-apply the tilt.

    Falls back to bounding-box aspect when fewer than 2 word centres
    are available (degenerate block).
    """
    centres = [
        ((w.bbox[0] + w.bbox[2]) / 2.0, (w.bbox[1] + w.bbox[3]) / 2.0)
        for m in members for w in m.words
    ]
    if len(centres) >= 2:
        # Vector from the first to the last word centre (sorted top-down).
        centres.sort(key=lambda c: (c[1], c[0]))
        dx = abs(centres[-1][0] - centres[0][0])
        dy = abs(centres[-1][1] - centres[0][1])
        return dy >= dx
    # Fallback: paragraph bbox aspect.
    boxes = [m.bbox for m in members]
    x1 = min(b[0] for b in boxes); x2 = max(b[2] for b in boxes)
    y1 = min(b[1] for b in boxes); y2 = max(b[3] for b in boxes)
    return (y2 - y1) >= (x2 - x1)


def _derive_shape_kind(
    members: list[TextBlock], anchor_cls: str | None = None,
) -> str:
    """Group-level shape — DETR balloon anchors win, else fall back to members.

    DETR ``bubble`` (and standalone ``text_bubble`` — see ``_dedup_anchors``)
    is the strongest "this is a speech balloon" signal available: an
    outline / safe-text detector confirmed the shape. Per-block
    ``classify_block`` only sees text geometry, so it tags a balloon
    tilted for art style (e.g. 7° rotated tategaki) as SFX and the
    container collapses to a tight burst OBB — hiding the balloon's
    real interior. When DETR says it's a balloon we trust it and force
    ``dialogue`` so the merged-hint + ellipse path can fire.

    Without an anchor we keep the previous behaviour: any SFX member
    promotes the whole group to ``burst``.
    """
    if anchor_cls in ("bubble", "text_bubble"):
        return "dialogue"
    for m in members:
        if classify_block(m, m.text or "") == "sfx":
            return "burst"
    return "dialogue"


def _container_pad(glyph: int, shape_kind: str) -> int:
    factor = _CONTAINER_PAD_FACTOR.get(shape_kind, _CONTAINER_PAD_FACTOR["dialogue"])
    return max(_CONTAINER_PAD_MIN_PX, int(glyph * factor))


def _merge_hint_with_word_union(hint: Bbox, members: list[TextBlock]) -> Bbox:
    """Expand the DETR text_bubble hint to cover the Lens glyph stripe.

    Guarantees ``word_union ⊆ result`` so the renderer never clips a
    glyph Lens actually read. When DETR text_bubble already covers
    word_union (the common case) the result equals the input hint.
    """
    w = _word_union(members)
    return (
        min(hint[0], w[0]), min(hint[1], w[1]),
        max(hint[2], w[2]), max(hint[3], w[3]),
    )


def _hint_is_sane(
    hint: Bbox,
    members: list[TextBlock],
    shape_kind: str,
    outsiders: list[TextBlock] | None = None,
) -> bool:
    """Accept the (already merged) hint when it stays on this group only.

    Guards (real signal, not heuristic area-ratio):

      1. ``shape_kind == "dialogue"`` — SFX/burst stays on the tight
         OBB path so tilted SFX don't get squared into a balloon rect.
      2. No outsider block's centre falls inside the hint. This fires
         exactly on the DETR over-shoot pattern: hint bbox bleeds onto
         a neighbouring balloon's glyphs. Replaces the earlier
         ``area ≤ 4 × word_union`` cap which fired false-positives on
         legitimate sparse-text balloons (single-character grunts in
         normal-sized speech bubbles have natural area ratios of
         4-10× without any over-shoot).
      3. Emergency cap: hint area ≤ ``_HINT_AREA_RATIO_MAX`` ×
         word_union area. Catches degenerate cases (DETR returning
         the full page) when no outsider block happens to lie inside.

    Coverage and "hint ≥ word_union" guards are implicit from
    ``_merge_hint_with_word_union`` (it expands the hint to include
    word_union), so they're not re-checked here.

    ``outsiders`` is every TextBlock NOT in ``members``. When None
    (legacy callers without page context), the centre-containment
    check is skipped — only the emergency cap applies.
    """
    if shape_kind != "dialogue":
        return False
    if outsiders:
        member_ids = {id(m) for m in members}
        for b in outsiders:
            if id(b) in member_ids:
                continue
            if _contains_center(hint, b.bbox):
                return False
    wu_area   = max(1, _area(_word_union(members)))
    hint_area = max(1, _area(hint))
    return hint_area <= wu_area * _HINT_AREA_RATIO_MAX


def _inscribed_ellipse(bbox: Bbox, n: int = _ELLIPSE_VERTICES) -> Polygon:
    """``n``-gon inscribed in the AABB ellipse, CCW starting at angle 0.

    Sampling stays exact (no cv2/np dependency in the hot path) and the
    resulting polygon is convex, so all downstream geometry helpers
    (Sutherland-Hodgman clip, AABB derive, fill) stay valid.
    """
    import math
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    rx = (x2 - x1) / 2.0
    ry = (y2 - y1) / 2.0
    step = 2.0 * math.pi / n
    return tuple(
        (cx + rx * math.cos(i * step), cy + ry * math.sin(i * step))
        for i in range(n)
    )


def _hint_aspect_ok(hint: Bbox) -> bool:
    w = max(1, hint[2] - hint[0])
    h = max(1, hint[3] - hint[1])
    aspect = w / h
    return _ELLIPSE_ASPECT_MIN <= aspect <= _ELLIPSE_ASPECT_MAX


def _container_polygon(
    members: list[TextBlock], page: tuple[int, int],
    shape_kind: str = "dialogue",
    hint_bbox: Bbox | None = None,
    all_blocks: list[TextBlock] | None = None,
) -> Polygon:
    """Render polygon — shape-aware container around the glyph stripe.

    Strategy:

      1. ``hint_bbox`` present + dialogue → merge with word_union so the
         result is guaranteed to cover every glyph, then sanity-check by
         centre-containment against blocks outside this group (DETR
         over-shoot rejection — see ``_hint_is_sane``). Sensible aspect →
         inscribed ellipse; extreme aspect → plain rect of the merged
         hint (still tighter to the balloon than word_union + pad on
         tall narrow speech bubbles).
      2. Otherwise (no hint, hint rejected by sanity check, or burst /
         column tategaki) → fall back to the OBB / AABB path with
         shape-aware padding. Same OBB construction as
         ``_word_axis_obb`` (centres + median glyph) so container and
         mask share orientation and base geometry; only the padding
         differs.

    ``all_blocks`` lets the sanity check see outsider blocks; when
    omitted the check skips the centre-containment guard.
    """
    if hint_bbox is not None and shape_kind == "dialogue":
        merged = _merge_hint_with_word_union(hint_bbox, members)
        outsiders = (
            [b for b in all_blocks if b not in members]
            if all_blocks is not None else None
        )
        if _hint_is_sane(merged, members, shape_kind, outsiders=outsiders):
            clipped = _clip(merged, page)
            if _hint_aspect_ok(clipped):
                return _inscribed_ellipse(clipped)
            return _bbox_to_polygon(clipped)

    pad = _container_pad(_median_glyph_size(members), shape_kind)

    rot = _group_rotation(members)
    if _is_column_layout(members) or abs(rot) < _ROTATION_AABB_DEG:
        x1, y1, x2, y2 = _word_union(members)
        return _bbox_to_polygon(
            _clip((x1 - pad, y1 - pad, x2 + pad, y2 + pad), page),
        )

    # Tilted non-column: shared OBB helper, padded.
    obb = _group_obb(members, pad)
    if obb is not None:
        return obb
    # Fallback: AABB.
    x1, y1, x2, y2 = _word_union(members)
    return _bbox_to_polygon(
        _clip((x1 - pad, y1 - pad, x2 + pad, y2 + pad), page),
    )


def _group_obb(members: list[TextBlock], pad: int) -> Polygon | None:
    """OBB across all members of the group — same axis math as
    ``_word_axis_obb`` but spanning every member's words so the
    container covers a multi-block SFX consistently."""
    import math
    import statistics

    word_boxes = [w.bbox for m in members for w in m.words]
    if len(word_boxes) < 2:
        return None
    centres = sorted(
        ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in word_boxes
    )
    dx = centres[-1][0] - centres[0][0]
    dy = centres[-1][1] - centres[0][1]
    length = math.hypot(dx, dy)
    if length < 1:
        return None
    ux, uy = dx / length, dy / length
    vx, vy = -uy, ux

    glyph_half = statistics.median(
        min(max(1, b[2] - b[0]), max(1, b[3] - b[1])) for b in word_boxes
    ) / 2.0

    cx0, cy0 = centres[0]
    cxN, cyN = centres[-1]
    u_start = cx0 * ux + cy0 * uy - glyph_half - pad
    u_end   = cxN * ux + cyN * uy + glyph_half + pad
    v_mid   = (cx0 * vx + cy0 * vy + cxN * vx + cyN * vy) / 2.0
    v_min   = v_mid - glyph_half - pad
    v_max   = v_mid + glyph_half + pad

    return tuple(
        (u * ux + v * vx, u * uy + v * vy)
        for u, v in [(u_start, v_min), (u_end, v_min),
                     (u_end, v_max), (u_start, v_max)]
    )


def _erase_masks_from_words(
    members: list[TextBlock],
    *,
    shape_kind: str = "dialogue",
) -> tuple[TextMask, ...]:
    """Per-member filled mask, tight on the glyph stripe with small dilation.

    Column layout → axis-aligned word_union AABB + dilation.
    Tilted non-column text → OBB built from word-first→word-last centre
    vector and median glyph short side, then dilated by the same
    fraction. Dilation = ``_MASK_PAD_FACTOR[shape_kind] × glyph_short``
    (min 2 px) — matches the container's pad factor so masks never
    exceed the render polygon while still covering Lens word-bbox
    under-coverage + anti-aliased glyph edges.
    """
    import cv2
    import numpy as np

    glyph = _median_glyph_size(members)
    factor = _MASK_PAD_FACTOR.get(shape_kind, _MASK_PAD_FACTOR["dialogue"])
    pad = max(_MASK_PAD_MIN_PX, int(glyph * factor))

    masks: list[TextMask] = []
    for m in members:
        obb = _word_axis_obb(m, pad)
        if obb is None:
            x1, y1, x2, y2 = _word_union([m])
            x1 -= pad; y1 -= pad; x2 += pad; y2 += pad
            w, h = max(1, x2 - x1), max(1, y2 - y1)
            image = np.full((h, w), 255, dtype=np.uint8)
            masks.append(TextMask(x=x1, y=y1, image=image))
            continue
        xs = [p[0] for p in obb]
        ys = [p[1] for p in obb]
        x1 = int(min(xs)); y1 = int(min(ys))
        x2 = int(max(xs)) + 1; y2 = int(max(ys)) + 1
        w, h = max(1, x2 - x1), max(1, y2 - y1)
        image = np.zeros((h, w), dtype=np.uint8)
        local = np.array(
            [[int(p[0]) - x1, int(p[1]) - y1] for p in obb], dtype=np.int32,
        )
        cv2.fillPoly(image, [local], 255)
        masks.append(TextMask(x=x1, y=y1, image=image))
    return tuple(masks)


def _word_axis_obb(block: TextBlock, pad: int = 0) -> Polygon | None:
    """OBB tight to the glyph stripe, expanded by ``pad`` on each side.

    Long axis = vector from first to last word centre, +/- half-glyph
    + ``pad`` at each end. Short axis = median word short-side / 2 +
    ``pad``, centred on the axis line. ``pad=0`` produces the same
    polygon that probe panel 1 draws as the green oriented overlay.

    Returns ``None`` for column-layout blocks or when word geometry
    cannot yield a stable axis — caller falls back to the word_union
    AABB.

    Projecting raw word bbox corners onto the OBB frame inflates the
    rect: Lens word bboxes are axis-aligned AABBs containing rotated
    glyphs, so their corners sit well outside the visible stripe.
    Always use centres + median glyph, never corner projections.
    """
    if not block.words or _is_column_layout([block]):
        return None
    import math
    import statistics

    centres = sorted(
        [((w.bbox[0] + w.bbox[2]) / 2.0, (w.bbox[1] + w.bbox[3]) / 2.0)
         for w in block.words],
        key=lambda c: c[0],
    )
    if len(centres) < 2:
        return None
    dx = centres[-1][0] - centres[0][0]
    dy = centres[-1][1] - centres[0][1]
    length = math.hypot(dx, dy)
    if length < 1:
        return None
    ux, uy = dx / length, dy / length
    vx, vy = -uy, ux

    glyph_half = statistics.median(
        min(max(1, w.bbox[2] - w.bbox[0]),
            max(1, w.bbox[3] - w.bbox[1]))
        for w in block.words
    ) / 2.0

    cx0, cy0 = centres[0]
    cxN, cyN = centres[-1]
    u_start = cx0 * ux + cy0 * uy - glyph_half - pad
    u_end   = cxN * ux + cyN * uy + glyph_half + pad
    v_mid   = (cx0 * vx + cy0 * vy + cxN * vx + cyN * vy) / 2.0
    v_min   = v_mid - glyph_half - pad
    v_max   = v_mid + glyph_half + pad

    return tuple(
        (u * ux + v * vx, u * uy + v * vy)
        for u, v in [(u_start, v_min), (u_end, v_min),
                     (u_end, v_max), (u_start, v_max)]
    )


def _polygon_bbox(polygon: Polygon, page: tuple[int, int]) -> Bbox:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return _clip(
        (int(min(xs)), int(min(ys)), int(max(xs) + 1), int(max(ys) + 1)),
        page,
    )



# ─── Reading order ────────────────────────────────────────────────────────


def _sort_for_reading(members: list[TextBlock]) -> list[TextBlock]:
    n_vert = sum(1 for m in members if m.text_direction == "vertical")
    if n_vert * 2 >= len(members):
        return sorted(members, key=lambda m: (-m.bbox[0], m.bbox[1]))
    return sorted(members, key=lambda m: (m.bbox[1], m.bbox[0]))


def _majority_direction(members: list[TextBlock]) -> str:
    n_vert = sum(1 for m in members if m.text_direction == "vertical")
    return "vertical" if n_vert * 2 >= len(members) else "horizontal"


# ─── Typesetting hint ─────────────────────────────────────────────────────


def _typesetting_hint(block: TextBlock) -> TypesettingHint | None:
    """Build a TypesettingHint from Lens word/line geometry.

    ``font_size_px`` = median of every WORD bbox's shorter side
    (``min(w_w, w_h)``). Words give us N samples per block instead of
    1-2 from lines, and a word's shorter side directly equals one glyph
    dimension regardless of writing direction:

      * Yokogaki word: `min` = line height (font ascent+descent).
      * Tategaki word: `min` = column width (≈ one glyph wide).

    Many word samples → robust median across the page (used by render's
    ``page_body_cap`` to pick one body font size for the whole page).
    Falls back to line-level median when word geometry is missing
    (rare; recogniser-only blocks, decoration filters, etc.).
    """
    samples: list[int] = []
    for w in block.words:
        s = min(w.bbox[2] - w.bbox[0], w.bbox[3] - w.bbox[1])
        if s > 0:
            samples.append(s)
    if not samples:
        for ln in block.lines:
            s = min(ln.bbox[2] - ln.bbox[0], ln.bbox[3] - ln.bbox[1])
            if s > 0:
                samples.append(s)
    if not samples:
        return None
    font_px = int(statistics.median(samples))
    chars_per_line = [
        sum(1 for c in l.text if not c.isspace()) for l in block.lines
    ] or [sum(1 for c in (block.text or "") if not c.isspace())]
    total = sum(chars_per_line)
    line_count = max(1, len(chars_per_line))
    avg = total / line_count
    return TypesettingHint(
        font_size_px=max(1, font_px),
        line_count=line_count,
        avg_chars_per_line=avg,
    )


def _merge_typesetting(
    members: list[TextBlock], merged_text: str,
) -> TypesettingHint | None:
    hints = [h for h in (_typesetting_hint(m) for m in members) if h is not None]
    if not hints:
        return None
    anchor = max(hints, key=lambda h: h.line_count)
    total_lines = sum(h.line_count for h in hints)
    total_chars = sum(1 for c in merged_text if not c.isspace())
    avg = total_chars / total_lines if total_lines > 0 else anchor.avg_chars_per_line
    return TypesettingHint(
        font_size_px=anchor.font_size_px,
        line_count=total_lines,
        avg_chars_per_line=avg,
    )


# ─── Group assembly ───────────────────────────────────────────────────────


def _assemble_group(
    members: list[TextBlock], polygon: Polygon, bbox: Bbox,
    anchor_cls: str | None = None,
) -> BubbleGroup:
    ordered = _sort_for_reading(members)
    text = "\n".join(
        (m.text or "").strip() for m in ordered if (m.text or "").strip()
    )
    direction   = _majority_direction(ordered)
    typesetting = _merge_typesetting(ordered, text)
    confidence  = max((m.confidence for m in ordered), default=0.0)

    shape_kind = _derive_shape_kind(ordered, anchor_cls=anchor_cls)

    # text_masks and erase_masks are the same filled paragraph AABB —
    # tight on the recognised glyph extent in both axes, no per-word
    # dilation. AOT erases exactly the area the renderer will paint.
    # Pass shape_kind so mask pad matches container pad (burst uses a
    # tighter factor to avoid cross-group overlap on tilted SFX).
    masks = _erase_masks_from_words(ordered, shape_kind=shape_kind)

    return BubbleGroup(
        bbox=bbox,
        polygon=polygon,
        text=text,
        confidence=confidence,
        text_masks=masks,
        erase_masks=masks,
        source="lens",
        shape_kind=shape_kind,
        used_fallback=False,
        rotation_deg=_group_rotation(ordered),
        typesetting=typesetting,
        text_direction=direction,
    )


# ─── Public API ───────────────────────────────────────────────────────────


def spatial_join(
    blocks: tuple[TextBlock, ...],
    regions: tuple[tuple[str, Bbox, float], ...],
    page_size: tuple[int, int],
) -> tuple[BubbleGroup, ...]:
    """Group Lens TextBlocks into BubbleGroups using Comic-DETR clusters.

    Algorithm:
      1. Reduce Comic-DETR regions to one anchor per overlapping cluster
         (``_dedup_anchors``). text_free > bubble > text_bubble.
      2. Walk anchors innermost-first (smallest area). Each anchor
         claims unassigned blocks whose centre lies inside its bbox.
      3. Remaining unassigned blocks become singleton groups.
      4. Per group, the render container is the Lens word union +
         ``0.5 × median_glyph_short_side`` padding (clipped to the page).
    """
    if not blocks:
        return ()

    raw_anchors = [
        _Anchor(cls=cls, bbox=bbox, conf=conf)
        for (cls, bbox, conf) in regions
        if cls in _CLUSTER_CLASSES
    ]
    text_bubbles = [a for a in raw_anchors if a.cls == "text_bubble"]
    anchors = _dedup_anchors(raw_anchors, all_text_bubbles=text_bubbles)
    # Innermost-first so a small bubble inside a big one wins its blocks.
    anchors.sort(key=lambda a: _area(a.bbox))

    assigned: set[int] = set()
    groups: list[BubbleGroup] = []

    for anchor in anchors:
        member_ids = [
            i for i, b in enumerate(blocks)
            if i not in assigned and _contains_center(anchor.bbox, b.bbox)
        ]
        if not member_ids:
            continue
        members  = [blocks[i] for i in member_ids]
        shape    = _derive_shape_kind(members, anchor_cls=anchor.cls)
        polygon  = _container_polygon(
            members, page_size, shape_kind=shape,
            hint_bbox=anchor.inner_bbox,
            all_blocks=list(blocks),
        )
        bbox     = _polygon_bbox(polygon, page_size)
        groups.append(_assemble_group(members, polygon, bbox, anchor_cls=anchor.cls))
        assigned.update(member_ids)

    for i, block in enumerate(blocks):
        if i in assigned:
            continue
        groups.append(block_to_group(block, page_size))

    groups.sort(key=lambda g: (g.bbox[1], g.bbox[0]))
    return tuple(groups)


def block_to_group(
    block: TextBlock,
    page_size: tuple[int, int] = (10000, 10000),
) -> BubbleGroup:
    """Wrap a single block as a singleton BubbleGroup.

    Container polygon = word union + half-glyph-size padding, rotated
    by the block's tilt when non-trivial. Render receives the polygon
    directly; erase mask = filled polygon.
    """
    text = block.text or ""
    block_class = classify_block(block, text)
    profile = PROFILES[block_class]
    typesetting = _typesetting_hint(block)

    polygon = _container_polygon([block], page_size, shape_kind=profile.shape_kind)
    bbox    = _polygon_bbox(polygon, page_size)
    masks   = _erase_masks_from_words([block], shape_kind=profile.shape_kind)

    return BubbleGroup(
        bbox=bbox,
        polygon=polygon,
        text=text,
        confidence=block.confidence,
        text_masks=masks,
        erase_masks=masks,
        source="lens",
        shape_kind=profile.shape_kind,
        used_fallback=False,
        rotation_deg=block.rotation_deg,
        typesetting=typesetting,
        text_direction=block.text_direction,
    )
