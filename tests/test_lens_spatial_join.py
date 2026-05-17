"""Spatial join — Lens block → BubbleGroup assignment tests.

Synthetic regions + TextBlocks; no model, no network. Tests target the
algorithm spec:

  * Grouping anchors = ``bubble`` ∪ ``text_free`` (``text_bubble`` ignored
    because Lens word geometry is tighter than the DETR inner rect).
  * Overlapping anchors collapse to the higher-confidence one.
  * Anchors are walked innermost-first so nested bubbles claim their
    inner blocks before the outer one absorbs everything.
  * Container box = word union + 0.5 × median line height (or
    block bbox padded by the floor when word geometry is absent).
"""

from __future__ import annotations

from typoon.vision.contracts import LineBox, TextBlock, WordBox
from typoon.vision.groupers._spatial_join import spatial_join


def _block(bbox, text, *, direction="horizontal", lines=(), words=()):
    return TextBlock(
        bbox=bbox, polygon=None, confidence=1.0,
        text=text, detector="lens_blocks",
        text_direction=direction,
        lines=tuple(lines) or (LineBox(bbox=bbox, text=text),),
        words=tuple(words),
    )


def _bubble(bbox, conf=0.9):       return ("bubble",      bbox, conf)
def _text_free(bbox, conf=0.9):    return ("text_free",   bbox, conf)
def _text_bubble(bbox, conf=0.9):  return ("text_bubble", bbox, conf)


# ─── Anchoring ────────────────────────────────────────────────────────────


def test_bubble_anchors_multiple_blocks_into_one_group():
    """Two Lens blocks inside one bubble → one merged group."""
    blocks = (
        _block((110, 110, 200, 140), "hello"),
        _block((110, 150, 200, 180), "world"),
    )
    regions = (_bubble((100, 100, 220, 200)),)
    groups = spatial_join(blocks, regions, (10000, 10000))
    assert len(groups) == 1
    assert groups[0].text == "hello\nworld"
    # bbox ⊇ text union (110,110,200,180)
    bx1, by1, bx2, by2 = groups[0].bbox
    assert bx1 <= 110 and bx2 >= 200
    assert by1 <= 110 and by2 >= 180


def test_text_free_acts_as_anchor():
    """text_free behaves identically to bubble for grouping."""
    blocks = (
        _block((110, 110, 200, 140), "alpha"),
        _block((110, 150, 200, 180), "beta"),
    )
    regions = (_text_free((100, 100, 220, 200)),)
    groups = spatial_join(blocks, regions, (10000, 10000))
    assert len(groups) == 1
    assert groups[0].text == "alpha\nbeta"


def test_text_bubble_standalone_anchors_blocks():
    """Standalone text_bubble (no bubble / text_free mate) DOES group.

    DETR sometimes misses the outer bubble outline and only fires
    text_bubble for a real balloon's safe-text area. Without grouping,
    the inner Lens blocks become adjacent singletons whose containers
    visibly overlap (see mangabuzz 374/1/9: "離せっ" + "一人で歩けるっ"
    share one balloon).
    """
    blocks = (
        _block((110, 110, 200, 140), "alpha"),
        _block((110, 150, 200, 180), "beta"),
    )
    regions = (_text_bubble((100, 100, 220, 200)),)
    groups = spatial_join(blocks, regions, (10000, 10000))
    assert len(groups) == 1
    assert groups[0].text == "alpha\nbeta"
    # text_bubble forces dialogue shape so the merged-hint path can fire.
    assert groups[0].shape_kind == "dialogue"


def test_innermost_anchor_claims_block():
    """Nested anchors: smaller one claims its block before the outer."""
    blocks = (_block((110, 110, 200, 180), "inside"),)
    regions = (
        _bubble((0, 0, 1000, 1000)),     # huge outer
        _bubble((100, 100, 220, 200)),   # tighter inner
    )
    groups = spatial_join(blocks, regions, (10000, 10000))
    assert len(groups) == 1
    bx1, by1, bx2, by2 = groups[0].bbox
    # Container is word-union-derived from the block itself; it stays
    # inside the inner bubble (the outer wouldn't have created this
    # tightness).
    assert bx2 - bx1 < 200 and by2 - by1 < 200


def test_overlapping_anchors_dedup_keeps_text_free():
    """3-class cluster (bubble + text_bubble + text_free) → caption (text_free)."""
    blocks = (_block((110, 110, 200, 180), "caption"),)
    regions = (
        _bubble((100, 100, 220, 200), conf=0.55),
        _text_bubble((100, 100, 220, 200), conf=0.42),
        _text_free((100, 100, 220, 200), conf=0.66),
    )
    groups = spatial_join(blocks, regions, (10000, 10000))
    # Even with bubble outranking it on absolute confidence, text_free
    # wins because the 3-class overlap is the caption signature.
    assert len(groups) == 1


def test_overlapping_bubble_and_text_bubble_keeps_bubble():
    """2-class cluster without text_free → bubble wins (text_bubble dropped)."""
    blocks = (_block((110, 110, 200, 180), "balloon"),)
    regions = (
        _bubble((100, 100, 220, 200), conf=0.40),
        _text_bubble((100, 100, 220, 200), conf=0.92),
    )
    groups = spatial_join(blocks, regions, (10000, 10000))
    assert len(groups) == 1


def test_text_bubble_only_cluster_emits_no_anchor():
    """A lone text_bubble cluster is dropped — Lens word geometry wins."""
    blocks = (_block((110, 110, 200, 180), "lens-tight"),)
    regions = (
        _text_bubble((100, 100, 220, 200)),
        _text_bubble((105, 105, 215, 195)),    # second overlapping copy
    )
    groups = spatial_join(blocks, regions, (10000, 10000))
    # Cluster reduces to nothing → block falls through as singleton.
    # The singleton container is Lens-derived (word_union + padding),
    # never the text_bubble bbox.
    assert len(groups) == 1


# ─── Stray handling ──────────────────────────────────────────────────────


def test_stray_block_becomes_singleton():
    blocks = (
        _block((10, 10, 50, 30), "stray"),
        _block((110, 110, 200, 140), "inside"),
    )
    regions = (_bubble((100, 100, 220, 150)),)
    groups = spatial_join(blocks, regions, (10000, 10000))
    assert sorted(g.text for g in groups) == ["inside", "stray"]


def test_no_regions_all_blocks_stray():
    blocks = (
        _block((10, 10, 50, 30), "a"),
        _block((60, 60, 100, 80), "b"),
    )
    groups = spatial_join(blocks, (), (10000, 10000))
    assert sorted(g.text for g in groups) == ["a", "b"]


def test_empty_blocks_returns_empty():
    assert spatial_join((), (_bubble((0, 0, 100, 100)),), (10000, 10000)) == ()


def test_anchor_with_no_blocks_emits_nothing():
    blocks = (_block((10, 10, 50, 30), "lonely"),)
    regions = (
        _bubble((300, 300, 400, 400)),       # empty
        _bubble((0, 0, 100, 100)),           # has block
    )
    groups = spatial_join(blocks, regions, (10000, 10000))
    assert len(groups) == 1
    assert groups[0].text == "lonely"


# ─── Container box (Lens-derived) ────────────────────────────────────────


def test_container_uses_word_union_when_words_present():
    """Word geometry → container = word union + padding, not paragraph bbox."""
    words = (
        WordBox(bbox=(110, 110, 150, 130), text="hi"),
        WordBox(bbox=(160, 110, 200, 130), text="ho"),
    )
    lines = (LineBox(bbox=(110, 110, 200, 130), text="hi ho"),)
    blocks = (_block((90, 90, 220, 150), "hi ho", lines=lines, words=words),)
    groups = spatial_join(blocks, (), (10000, 10000))
    bx1, by1, bx2, by2 = groups[0].bbox
    # Container derived from word union (110..200, 110..130) + pad,
    # NOT from the loose paragraph bbox (90..220, 90..150).
    assert bx1 >= 90 and bx2 <= 220
    assert bx1 <= 110 and bx2 >= 200


def test_container_falls_back_to_block_bbox_without_words():
    """No word geometry → word_union falls back to block bbox; padding
    derives from the (default) line height = block height."""
    blocks = (_block((100, 100, 200, 150), "hi"),)
    groups = spatial_join(blocks, (), (10000, 10000))
    bx1, by1, bx2, by2 = groups[0].bbox
    # block 100×50 → default line bbox = block bbox → glyph short = 50
    # → pad = 0.2 × 50 = 10. +1 on high edge from polygon_bbox rounding.
    assert (bx1, by1, bx2, by2) == (90, 90, 211, 161)


# ─── Reading order ───────────────────────────────────────────────────────


def test_tategaki_columns_sorted_right_to_left():
    blocks = (
        _block((100, 50, 130, 250), "左列", direction="vertical"),
        _block((200, 50, 230, 250), "中列", direction="vertical"),
        _block((300, 50, 330, 250), "右列", direction="vertical"),
    )
    regions = (_bubble((90, 40, 340, 260)),)
    groups = spatial_join(blocks, regions, (10000, 10000))
    assert groups[0].text.splitlines() == ["右列", "中列", "左列"]
    assert groups[0].text_direction == "vertical"


def test_horizontal_rows_sorted_top_to_bottom():
    blocks = (
        _block((100, 200, 300, 230), "second"),
        _block((100, 50, 300, 80), "first"),
        _block((100, 350, 300, 380), "third"),
    )
    regions = (_bubble((90, 40, 310, 400)),)
    groups = spatial_join(blocks, regions, (10000, 10000))
    assert groups[0].text.splitlines() == ["first", "second", "third"]


def test_groups_emitted_top_down_left_right():
    blocks = (
        _block((10, 10, 50, 30), "A"),
        _block((10, 200, 50, 230), "C"),
        _block((400, 10, 450, 30), "B"),
    )
    groups = spatial_join(blocks, (), (10000, 10000))
    assert [g.text for g in groups] == ["A", "B", "C"]
