"""Tests for FullPageInpainter and TextEraser."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from typoon.vision.contracts import TextMask
from typoon.vision.erasers.inpaint import FullPageInpainter, PageInpainter
from typoon.vision.erasers.eraser import TextEraser
from typoon.vision.erasers.backends import InpaintBackend


# ─── fake backends ────────────────────────────────────────────────────────────


class _IdentityBackend:
    name = "identity"
    def inpaint(self, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return image_rgb.copy()


class _FillBackend:
    name = "fill"
    def __init__(self, colour: tuple[int, int, int]) -> None:
        self._colour = np.array(colour, dtype=np.uint8)
    def inpaint(self, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        out = np.empty_like(image_rgb)
        out[:] = self._colour
        return out


class _RecordingBackend:
    name = "recording"
    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, np.ndarray]] = []
    def inpaint(self, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        self.calls.append((image_rgb.copy(), mask.copy()))
        return image_rgb.copy()


# ─── helpers ──────────────────────────────────────────────────────────────────


def _make_canvas(H=100, W=80, alpha=True) -> np.ndarray:
    channels = 4 if alpha else 3
    c = np.zeros((H, W, channels), dtype=np.uint8)
    c[:, :, :3] = 128
    if alpha:
        c[:, :, 3] = 255
    return c


def _mask_at(H, W, *rects) -> np.ndarray:
    m = np.zeros((H, W), dtype=np.uint8)
    for y1, y2, x1, x2 in rects:
        m[y1:y2, x1:x2] = 255
    return m


def _text_mask(x, y, w, h, value=255) -> TextMask:
    img = np.full((h, w), value, dtype=np.uint8)
    return TextMask(x=x, y=y, image=img)


def _run(coro):
    return asyncio.run(coro)


# ─── FullPageInpainter ────────────────────────────────────────────────────────


def test_full_page_no_mask_unchanged():
    canvas = _make_canvas()
    page_mask = np.zeros((100, 80), dtype=np.uint8)
    FullPageInpainter(_FillBackend((0, 0, 0))).inpaint_page(canvas, page_mask)
    assert (canvas[:, :, :3] == 128).all()


def test_full_page_only_masked_changed():
    canvas = _make_canvas()
    page_mask = _mask_at(100, 80, (20, 40, 30, 50))
    FullPageInpainter(_FillBackend((10, 20, 30))).inpaint_page(canvas, page_mask)
    assert (canvas[20:40, 30:50, :3] == [10, 20, 30]).all()
    assert (canvas[0:20, :, :3] == 128).all()


def test_full_page_alpha_set_on_masked():
    canvas = _make_canvas()
    canvas[:, :, 3] = 0
    page_mask = _mask_at(100, 80, (10, 20, 10, 20))
    FullPageInpainter(_IdentityBackend()).inpaint_page(canvas, page_mask)
    assert (canvas[10:20, 10:20, 3] == 255).all()
    assert (canvas[0:10, :, 3] == 0).all()


def test_full_page_identity_leaves_rgb():
    canvas = _make_canvas()
    orig = canvas[:, :, :3].copy()
    page_mask = np.ones((100, 80), dtype=np.uint8) * 255
    FullPageInpainter(_IdentityBackend()).inpaint_page(canvas, page_mask)
    np.testing.assert_array_equal(canvas[:, :, :3], orig)


# ─── TextEraser ───────────────────────────────────────────────────────────────


def test_text_eraser_no_masks_returns_canvas():
    canvas = _make_canvas(100, 100)
    orig = canvas.copy()
    uniform_rec = _RecordingBackend()
    complex_rec = _RecordingBackend()
    eraser = TextEraser(
        uniform_inpainter=FullPageInpainter(uniform_rec),
        complex_inpainter=FullPageInpainter(complex_rec),
    )
    result = _run(eraser.erase(canvas, ()))
    assert result is canvas
    assert len(uniform_rec.calls) == 0
    assert len(complex_rec.calls) == 0


def test_text_eraser_uniform_routes_to_uniform():
    """White canvas → low spread → uniform path."""
    canvas = np.full((200, 200, 4), 255, dtype=np.uint8)
    uniform_rec = _RecordingBackend()
    complex_rec = _RecordingBackend()
    eraser = TextEraser(
        uniform_inpainter=FullPageInpainter(uniform_rec),
        complex_inpainter=FullPageInpainter(complex_rec),
        spread_threshold=30,
    )
    img = np.zeros((40, 40), dtype=np.uint8)
    img[10:30, 10:30] = 255
    _run(eraser.erase(canvas, (TextMask(x=50, y=50, image=img),)))
    assert len(uniform_rec.calls) == 1
    assert len(complex_rec.calls) == 0


def test_text_eraser_complex_routes_to_complex():
    """Noisy canvas → high spread → complex path."""
    rng = np.random.default_rng(42)
    canvas = rng.integers(0, 255, (200, 200, 4), dtype=np.uint8)
    canvas[:, :, 3] = 255
    uniform_rec = _RecordingBackend()
    complex_rec = _RecordingBackend()
    eraser = TextEraser(
        uniform_inpainter=FullPageInpainter(uniform_rec),
        complex_inpainter=FullPageInpainter(complex_rec),
        spread_threshold=30,
    )
    img = np.zeros((40, 40), dtype=np.uint8)
    img[10:30, 10:30] = 255
    _run(eraser.erase(canvas, (TextMask(x=50, y=50, image=img),)))
    assert len(complex_rec.calls) == 1
    assert len(uniform_rec.calls) == 0


def test_text_eraser_complex_fallback_on_error():
    class _ErrorBackend:
        name = "error"
        def inpaint(self, img, mask): raise RuntimeError("boom")

    rng = np.random.default_rng(0)
    canvas = rng.integers(0, 255, (200, 200, 4), dtype=np.uint8)
    canvas[:, :, 3] = 255
    uniform_rec = _RecordingBackend()
    eraser = TextEraser(
        uniform_inpainter=FullPageInpainter(uniform_rec),
        complex_inpainter=FullPageInpainter(_ErrorBackend()),
        spread_threshold=30,
    )
    img = np.zeros((40, 40), dtype=np.uint8)
    img[10:30, 10:30] = 255
    _run(eraser.erase(canvas, (TextMask(x=50, y=50, image=img),)))
    assert len(uniform_rec.calls) == 1  # fallback ran
