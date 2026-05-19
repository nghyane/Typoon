"""Tests for TiledInpainter, FullPageInpainter, TextEraser."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from typoon.vision.contracts import TextMask
from typoon.vision.erasers.inpaint import (
    FullPageInpainter,
    TiledInpainter,
    AreaGatedInpainter,
    _adaptive_tile,
    _center_tile,
    _snap_up,
)
from typoon.vision.erasers.eraser import TextEraser
from typoon.vision.erasers.backends import InpaintBackend


# ─── fake backends ────────────────────────────────────────────────────────


class _IdentityBackend:
    """Returns input unchanged — verifies paste path without mutation."""
    name = "identity"
    def inpaint(self, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return image_rgb.copy()


class _FillBackend:
    """Fills every pixel (masked or not) with a fixed colour."""
    name = "fill"
    def __init__(self, colour: tuple[int, int, int]) -> None:
        self._colour = np.array(colour, dtype=np.uint8)
    def inpaint(self, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        out = np.empty_like(image_rgb)
        out[:] = self._colour
        return out


class _RecordingBackend:
    """Records every (image, mask) pair it receives."""
    name = "recording"
    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, np.ndarray]] = []
    def inpaint(self, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        self.calls.append((image_rgb.copy(), mask.copy()))
        return image_rgb.copy()


# ─── _snap_up ─────────────────────────────────────────────────────────────


def test_snap_up_already_aligned():
    assert _snap_up(16, 8) == 16


def test_snap_up_rounds_up():
    assert _snap_up(17, 8) == 24
    assert _snap_up(1, 8) == 8


# ─── _adaptive_tile ───────────────────────────────────────────────────────


def test_adaptive_tile_snapped_to_8():
    x1, y1, x2, y2 = _adaptive_tile(100, 100, 50, 50, 500, 500, context_px=64, snap=8)
    assert (x2 - x1) % 8 == 0
    assert (y2 - y1) % 8 == 0


def test_adaptive_tile_covers_blob():
    x1, y1, x2, y2 = _adaptive_tile(100, 100, 80, 60, 500, 500, context_px=64, snap=8)
    assert x1 <= 100 and y1 <= 100
    assert x2 >= 180 and y2 >= 160


def test_adaptive_tile_has_context():
    # blob 80×60 + context=64 each side → tile at least 208×188
    x1, y1, x2, y2 = _adaptive_tile(100, 100, 80, 60, 1000, 1000, context_px=64, snap=8)
    assert x2 - x1 >= 208
    assert y2 - y1 >= 188


def test_adaptive_tile_clamped_to_page():
    x1, y1, x2, y2 = _adaptive_tile(0, 0, 30, 30, 200, 200, context_px=64, snap=8)
    assert x1 >= 0 and y1 >= 0
    assert x2 <= 200 and y2 <= 200


def test_adaptive_tile_large_blob_25pct_padding():
    # blob 400×300 — 25% pad = max(64, 100, 75)=100 each side
    x1, y1, x2, y2 = _adaptive_tile(50, 50, 400, 300, 2000, 2000, context_px=64, snap=8)
    assert x2 - x1 >= 600  # 400 + 100*2
    assert y2 - y1 >= 496  # 300 + 100*2 = 500 → snapped up, but at least 496


# ─── _center_tile ─────────────────────────────────────────────────────────


def test_center_tile_middle():
    x1, y1, x2, y2 = _center_tile(400, 400, 50, 50, 384, 1000, 1000)
    assert x2 - x1 == 384
    assert y2 - y1 == 384
    # blob centre is 425,425; tile should straddle it
    cx, cy = 425, 425
    assert x1 <= cx <= x2
    assert y1 <= cy <= y2


def test_center_tile_left_edge():
    # blob at left edge — tile clamped, still exactly T wide
    x1, y1, x2, y2 = _center_tile(0, 200, 30, 30, 384, 1000, 1000)
    assert x1 == 0
    assert x2 - x1 == 384


def test_center_tile_bottom_right_edge():
    # blob touching bottom-right corner
    x1, y1, x2, y2 = _center_tile(980, 980, 20, 20, 384, 1000, 1000)
    assert x2 == 1000
    assert x2 - x1 == 384
    assert y2 == 1000
    assert y2 - y1 == 384


def test_center_tile_page_smaller_than_tile():
    # page 200×200 with tile_size=384 — tile clips to page
    x1, y1, x2, y2 = _center_tile(50, 50, 30, 30, 384, 200, 200)
    assert x1 >= 0 and y1 >= 0
    assert x2 <= 200 and y2 <= 200


# ─── FullPageInpainter ────────────────────────────────────────────────────


def _make_canvas(H=100, W=80, alpha=True) -> np.ndarray:
    channels = 4 if alpha else 3
    c = np.zeros((H, W, channels), dtype=np.uint8)
    c[:, :, :3] = 128
    if alpha:
        c[:, :, 3] = 255
    return c


def test_full_page_no_masked_pixels_unchanged():
    canvas = _make_canvas()
    page_mask = np.zeros((100, 80), dtype=np.uint8)
    FullPageInpainter(_FillBackend((0, 0, 0))).inpaint_page(canvas, page_mask)
    assert (canvas[:, :, :3] == 128).all()


def test_full_page_only_masked_pixels_changed():
    canvas = _make_canvas()
    page_mask = np.zeros((100, 80), dtype=np.uint8)
    page_mask[20:40, 30:50] = 255
    FullPageInpainter(_FillBackend((10, 20, 30))).inpaint_page(canvas, page_mask)
    # masked region → filled colour
    assert (canvas[20:40, 30:50, :3] == [10, 20, 30]).all()
    # unmasked region → unchanged
    assert (canvas[0:20, :, :3] == 128).all()


def test_full_page_alpha_set_on_masked():
    canvas = _make_canvas()
    canvas[:, :, 3] = 0  # alpha all zero
    page_mask = np.zeros((100, 80), dtype=np.uint8)
    page_mask[10:20, 10:20] = 255
    FullPageInpainter(_IdentityBackend()).inpaint_page(canvas, page_mask)
    assert (canvas[10:20, 10:20, 3] == 255).all()
    assert (canvas[0:10, :, 3] == 0).all()


def test_full_page_identity_backend_leaves_rgb_unchanged():
    canvas = _make_canvas()
    orig_rgb = canvas[:, :, :3].copy()
    page_mask = np.ones((100, 80), dtype=np.uint8) * 255
    FullPageInpainter(_IdentityBackend()).inpaint_page(canvas, page_mask)
    np.testing.assert_array_equal(canvas[:, :, :3], orig_rgb)


# ─── TiledInpainter ───────────────────────────────────────────────────────


def _mask_at(H, W, *rects) -> np.ndarray:
    """Create page_mask with 255 in each (y1,y2,x1,x2) rect."""
    m = np.zeros((H, W), dtype=np.uint8)
    for y1, y2, x1, x2 in rects:
        m[y1:y2, x1:x2] = 255
    return m


def test_tiled_empty_mask_no_call():
    rec = _RecordingBackend()
    canvas = _make_canvas(200, 200)
    TiledInpainter(rec, context_px=32).inpaint_page(canvas, np.zeros((200,200), np.uint8))
    assert len(rec.calls) == 0


def test_tiled_single_blob_one_call():
    rec = _RecordingBackend()
    canvas = _make_canvas(200, 200)
    mask = _mask_at(200, 200, (80, 110, 80, 110))
    TiledInpainter(rec, context_px=32).inpaint_page(canvas, mask)
    assert len(rec.calls) == 1


def test_tiled_single_blob_fill_colour_applied():
    canvas = _make_canvas(200, 200)
    mask = _mask_at(200, 200, (80, 100, 80, 100))
    TiledInpainter(_FillBackend((7, 8, 9)), context_px=32).inpaint_page(canvas, mask)
    assert (canvas[80:100, 80:100, :3] == [7, 8, 9]).all()
    # outside mask unchanged
    assert (canvas[0:80, :, :3] == 128).all()


def test_tiled_two_far_blobs_two_calls():
    rec = _RecordingBackend()
    canvas = _make_canvas(500, 500)
    # two blobs far enough apart to not share a 64-tile
    mask = _mask_at(500, 500, (10, 30, 10, 30), (460, 490, 460, 490))
    TiledInpainter(rec, context_px=32).inpaint_page(canvas, mask)
    assert len(rec.calls) == 2


def test_tiled_two_adjacent_blobs_one_call():
    rec = _RecordingBackend()
    canvas = _make_canvas(200, 200)
    # two blobs very close — will share a 128-tile (covered dedup)
    mask = _mask_at(200, 200, (90, 100, 90, 100), (95, 105, 95, 105))
    TiledInpainter(rec, context_px=40).inpaint_page(canvas, mask)
    assert len(rec.calls) == 1


def test_tiled_tile_snapped_to_8():
    rec = _RecordingBackend()
    canvas = _make_canvas(400, 400)
    mask = _mask_at(400, 400, (180, 220, 180, 220))  # blob 40×40
    TiledInpainter(rec, context_px=32, snap=8).inpaint_page(canvas, mask)
    assert len(rec.calls) == 1
    tile_h, tile_w = rec.calls[0][0].shape[:2]
    # blob 40×40 + 32 pad each side = 104 → snapped to 104 (already aligned)
    assert tile_h % 8 == 0
    assert tile_w % 8 == 0


def test_tiled_unmasked_pixels_unchanged():
    canvas = _make_canvas(200, 200)
    mask = _mask_at(200, 200, (90, 110, 90, 110))
    orig = canvas.copy()
    TiledInpainter(_IdentityBackend(), context_px=32).inpaint_page(canvas, mask)
    # pixels outside mask rect unmodified
    np.testing.assert_array_equal(canvas[0:90, :], orig[0:90, :])
    np.testing.assert_array_equal(canvas[110:, :], orig[110:, :])


# ─── TextEraser ───────────────────────────────────────────────────────────


def _text_mask(x, y, w, h, value=255) -> TextMask:
    img = np.full((h, w), value, dtype=np.uint8)
    return TextMask(x=x, y=y, image=img)


def _run(coro):
    return asyncio.run(coro)


def test_text_eraser_no_masks_returns_canvas():
    canvas = _make_canvas(100, 100)
    orig = canvas.copy()
    uniform_rec = _RecordingBackend()
    complex_rec = _RecordingBackend()
    eraser = TextEraser(
        uniform_inpainter=FullPageInpainter(uniform_rec),
        complex_inpainter=TiledInpainter(complex_rec, context_px=32),
    )
    result = _run(eraser.erase(canvas, ()))
    assert result is canvas
    assert len(uniform_rec.calls) == 0
    assert len(complex_rec.calls) == 0


def test_text_eraser_uniform_mask_routes_to_uniform():
    """White canvas → low spread → uniform path."""
    canvas = np.full((200, 200, 4), 255, dtype=np.uint8)
    uniform_rec = _RecordingBackend()
    complex_rec = _RecordingBackend()
    eraser = TextEraser(
        uniform_inpainter=FullPageInpainter(uniform_rec),
        complex_inpainter=TiledInpainter(complex_rec, context_px=32),
        spread_threshold=30,
    )
    mask = (_text_mask(50, 50, 40, 40),)
    _run(eraser.erase(canvas, mask))
    assert len(uniform_rec.calls) == 1
    assert len(complex_rec.calls) == 0


def test_text_eraser_complex_mask_routes_to_complex():
    """Noisy canvas → high spread → complex path."""
    rng = np.random.default_rng(42)
    canvas = rng.integers(0, 255, (200, 200, 4), dtype=np.uint8)
    canvas[:, :, 3] = 255
    uniform_rec = _RecordingBackend()
    complex_rec = _RecordingBackend()
    eraser = TextEraser(
        uniform_inpainter=FullPageInpainter(uniform_rec),
        complex_inpainter=TiledInpainter(complex_rec, context_px=32),
        spread_threshold=30,
    )
    # mask: 255 only in a 20×20 centre of a 40×40 crop — outer ring is bg
    # partition_by_background samples bg pixels (crop_mask==0) to measure spread
    img = np.zeros((40, 40), dtype=np.uint8)
    img[10:30, 10:30] = 255
    mask = (TextMask(x=50, y=50, image=img),)
    _run(eraser.erase(canvas, mask))
    assert len(complex_rec.calls) == 1
    assert len(uniform_rec.calls) == 0


def test_text_eraser_complex_fallback_on_error():
    """If complex inpainter raises, uniform fallback runs silently."""
    class _ErrorBackend:
        name = "error"
        def inpaint(self, img, mask): raise RuntimeError("boom")

    canvas = np.full((200, 200, 4), 50, dtype=np.uint8)
    # make canvas complex (noisy)
    rng = np.random.default_rng(0)
    canvas[:, :, :3] = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
    canvas[:, :, 3] = 255
    uniform_rec = _RecordingBackend()
    eraser = TextEraser(
        uniform_inpainter=FullPageInpainter(uniform_rec),
        complex_inpainter=TiledInpainter(_ErrorBackend(), context_px=32),
        spread_threshold=30,
    )
    mask = (_text_mask(50, 50, 40, 40),)
    _run(eraser.erase(canvas, mask))  # must not raise
    assert len(uniform_rec.calls) == 1  # fallback ran


# ─── AreaGatedInpainter ───────────────────────────────────────────────────


def test_area_gated_small_blob_goes_to_small():
    small_rec = _RecordingBackend()
    large_rec = _RecordingBackend()
    canvas = _make_canvas(200, 200)
    # blob area = 20*20 = 400 < 1000 → small
    mask = _mask_at(200, 200, (90, 110, 90, 110))
    ag = AreaGatedInpainter(
        small_inpainter=FullPageInpainter(small_rec),
        large_inpainter=TiledInpainter(large_rec, context_px=32),
        area_threshold=1000,
    )
    ag.inpaint_page(canvas, mask)
    assert len(small_rec.calls) == 1
    assert len(large_rec.calls) == 0


def test_area_gated_large_blob_goes_to_large():
    small_rec = _RecordingBackend()
    large_rec = _RecordingBackend()
    canvas = _make_canvas(300, 300)
    # blob area = 50*30 = 1500 >= 1000 → large
    mask = _mask_at(300, 300, (100, 130, 100, 150))
    ag = AreaGatedInpainter(
        small_inpainter=FullPageInpainter(small_rec),
        large_inpainter=TiledInpainter(large_rec, context_px=32),
        area_threshold=1000,
    )
    ag.inpaint_page(canvas, mask)
    # small path: small_mask is empty → no call
    assert len(small_rec.calls) == 0
    # large path: TiledInpainter calls large_rec.inpaint once for the blob
    assert len(large_rec.calls) == 1


def test_area_gated_mixed_blobs_split_correctly():
    small_rec = _RecordingBackend()
    large_rec = _RecordingBackend()
    canvas = _make_canvas(400, 400)
    # small: 10*10=100, large: 50*40=2000
    mask = _mask_at(400, 400, (10, 20, 10, 20), (200, 240, 200, 250))
    ag = AreaGatedInpainter(
        small_inpainter=FullPageInpainter(small_rec),
        large_inpainter=TiledInpainter(large_rec, context_px=32),
        area_threshold=1000,
    )
    ag.inpaint_page(canvas, mask)
    assert len(small_rec.calls) == 1  # small blob → full-page TeLeA
    assert len(large_rec.calls) == 1  # large blob → tiled AOT


def test_area_gated_empty_mask_no_call():
    small_rec = _RecordingBackend()
    large_rec = _RecordingBackend()
    canvas = _make_canvas(200, 200)
    ag = AreaGatedInpainter(
        small_inpainter=FullPageInpainter(small_rec),
        large_inpainter=TiledInpainter(large_rec, context_px=32),
        area_threshold=1000,
    )
    ag.inpaint_page(canvas, np.zeros((200, 200), dtype=np.uint8))
    assert len(small_rec.calls) == 0
    assert len(large_rec.calls) == 0
