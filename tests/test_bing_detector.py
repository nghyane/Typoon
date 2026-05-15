"""Bing block detector — pure-function tests.

Network calls go through curl_cffi at runtime; here we verify the
preprocessing + response parsing logic that runs around the I/O.
"""

from __future__ import annotations

import numpy as np
import pytest

from typoon.vision.detectors.bing_blocks import (
    _Quad,
    _extract_lines,
    _line_to_block,
    _prepare_for_bing,
    _quad_from_bing_bbox,
)


# ─── Image preprocessing ──────────────────────────────────────────────────


def test_prepare_resizes_oversized_image():
    huge = np.zeros((6000, 8000, 3), dtype=np.uint8)
    prepared = _prepare_for_bing(huge)
    assert max(prepared.width, prepared.height) <= 4000
    assert prepared.encoded_size <= 767_772


def test_prepare_keeps_small_image_dims():
    small = np.full((400, 300, 3), 255, dtype=np.uint8)
    prepared = _prepare_for_bing(small)
    assert (prepared.width, prepared.height) == (300, 400)


def test_prepare_handles_complex_pixels_via_jpeg_fallback():
    # Random noise compresses poorly under PNG → forces JPEG ramp
    rng = np.random.default_rng(seed=42)
    noisy = rng.integers(0, 256, size=(2500, 1755, 3), dtype=np.uint8)
    prepared = _prepare_for_bing(noisy)
    assert prepared.encoded_size <= 767_772


# ─── Bbox conversion ──────────────────────────────────────────────────────


def test_quad_from_bing_bbox_extracts_axis_aligned_extents():
    quad = {
        "topLeft":     {"x": 0.10, "y": 0.20},
        "topRight":    {"x": 0.90, "y": 0.21},
        "bottomRight": {"x": 0.91, "y": 0.40},
        "bottomLeft":  {"x": 0.11, "y": 0.39},
    }
    out = _quad_from_bing_bbox(quad)
    assert out is not None
    assert out.x == pytest.approx(0.10)
    assert out.y == pytest.approx(0.20)
    assert out.w == pytest.approx(0.81)
    assert out.h == pytest.approx(0.20)


def test_quad_from_bing_bbox_rejects_malformed():
    assert _quad_from_bing_bbox({}) is None
    assert _quad_from_bing_bbox({"topLeft": {"x": 0.5}}) is None


def test_line_to_block_scales_to_page_pixels():
    quad = _Quad(x=0.10, y=0.20, w=0.30, h=0.10)
    block = _line_to_block("hello", quad, page_w=1000, page_h=2000)
    assert block.bbox == (100, 400, 400, 600)
    assert block.text == "hello"
    assert block.detector == "bing_blocks"


def test_line_to_block_clamps_to_page_bounds():
    quad = _Quad(x=0.95, y=0.95, w=0.20, h=0.20)
    block = _line_to_block("edge", quad, page_w=1000, page_h=1000)
    assert block.bbox[2] == 1000
    assert block.bbox[3] == 1000


# ─── Response parsing ─────────────────────────────────────────────────────


def _wrap_lines(lines: list[dict]) -> dict:
    return {
        "tags": [{
            "displayName": "##TextRecognition",
            "actions": [{
                "_type": "ImageKnowledge/TextRecognitionAction",
                "data": {
                    "regions": [{
                        "boundingBox": {
                            "topLeft":     {"x": 0, "y": 0},
                            "topRight":    {"x": 1, "y": 0},
                            "bottomRight": {"x": 1, "y": 1},
                            "bottomLeft":  {"x": 0, "y": 1},
                        },
                        "lines": lines,
                    }],
                },
            }],
        }],
    }


def test_extract_lines_pulls_text_and_quads():
    payload = _wrap_lines([
        {
            "text": "hello world",
            "boundingBox": {
                "topLeft":     {"x": 0.10, "y": 0.20},
                "topRight":    {"x": 0.40, "y": 0.20},
                "bottomRight": {"x": 0.40, "y": 0.30},
                "bottomLeft":  {"x": 0.10, "y": 0.30},
            },
        },
        {
            "text": "second",
            "boundingBox": {
                "topLeft":     {"x": 0.50, "y": 0.50},
                "topRight":    {"x": 0.60, "y": 0.50},
                "bottomRight": {"x": 0.60, "y": 0.55},
                "bottomLeft":  {"x": 0.50, "y": 0.55},
            },
        },
    ])
    out = _extract_lines(payload)
    assert len(out) == 2
    assert out[0][0] == "hello world"
    assert out[1][0] == "second"


def test_extract_lines_skips_blank_text():
    payload = _wrap_lines([
        {"text": "", "boundingBox": {
            "topLeft": {"x": 0, "y": 0}, "topRight": {"x": 1, "y": 0},
            "bottomRight": {"x": 1, "y": 1}, "bottomLeft": {"x": 0, "y": 1},
        }},
        {"text": "   ", "boundingBox": {
            "topLeft": {"x": 0, "y": 0}, "topRight": {"x": 1, "y": 0},
            "bottomRight": {"x": 1, "y": 1}, "bottomLeft": {"x": 0, "y": 1},
        }},
    ])
    assert _extract_lines(payload) == []


def test_extract_lines_returns_empty_when_no_text_tag():
    assert _extract_lines({"tags": []}) == []
    assert _extract_lines({}) == []


def test_extract_lines_ignores_unknown_action_types():
    payload = {
        "tags": [{
            "displayName": "##TextRecognition",
            "actions": [{
                "_type": "ImageKnowledge/SomeOtherAction",
                "data": {"regions": [{"lines": [{"text": "x"}]}]},
            }],
        }],
    }
    assert _extract_lines(payload) == []
