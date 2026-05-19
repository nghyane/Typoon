"""Vision pipeline composition + validation tests."""

from __future__ import annotations

import pytest

from typoon.vision.pipeline import PRESETS, VisionPipelineSpec


def test_default_spec_is_lens_pipeline():
    spec = VisionPipelineSpec()
    assert spec.detector == "lens_blocks"
    assert spec.grouper == "lens_native"
    assert spec.recognizer == "none"
    assert spec.eraser == "hybrid"


def test_presets_all_valid():
    """Every named preset must satisfy __post_init__ validation."""
    for name, spec in PRESETS.items():
        # Re-construct via with_overrides → triggers re-validation
        cloned = spec.with_overrides()
        assert cloned == spec, f"preset {name!r} failed round-trip"


def test_lens_native_grouper_requires_lens_detector():
    with pytest.raises(ValueError, match="lens_native"):
        VisionPipelineSpec(
            detector="ctd_blocks",
            grouper="lens_native",
            recognizer="apple_vision",
        )


def test_recognizer_none_requires_text_shipping_detector():
    with pytest.raises(ValueError, match="recognizer=none"):
        VisionPipelineSpec(
            detector="ctd_blocks",
            grouper="ctd_native",
            recognizer="none",
        )


def test_concurrency_must_be_positive():
    for field in ("page_concurrency", "detect_concurrency", "erase_concurrency"):
        with pytest.raises(ValueError, match=f"{field} must be"):
            VisionPipelineSpec(**{field: 0})


def test_with_overrides_returns_validated_copy():
    base = VisionPipelineSpec.preset("lens")
    tuned = base.with_overrides(page_concurrency=16)
    assert tuned.page_concurrency == 16
    assert tuned.detector == base.detector
    # Invalid overrides re-trigger validation
    with pytest.raises(ValueError):
        base.with_overrides(grouper="ctd_native", detector="lens_blocks")


def test_preset_unknown_raises():
    with pytest.raises(KeyError, match="unknown preset"):
        VisionPipelineSpec.preset("does_not_exist")


def test_ctd_manga_preset_uses_manga_ocr():
    spec = VisionPipelineSpec.preset("ctd_manga")
    assert spec.detector == "ctd_blocks"
    assert spec.grouper == "ctd_native"
    assert spec.recognizer == "manga_ocr"
