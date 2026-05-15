"""Empirical block classification tests (SFX / dialogue / narration).

Thresholds were derived from Lens output on ch001 fixture chapters.
These tests pin the behaviour so future tweaks to the constants can be
validated against the same evidence.
"""

from __future__ import annotations

from typoon.vision.contracts import TextBlock
from typoon.vision.groupers.lens_native import _classify_block


def _b(bbox, text, rotation_deg=0.0):
    return TextBlock(
        bbox=bbox, polygon=None, confidence=1.0,
        text=text, detector="lens_blocks",
        rotation_deg=rotation_deg,
    )


# ─── SFX detection ────────────────────────────────────────────────────────


def test_rotated_text_classified_as_sfx():
    """Any block tilted > 5° is SFX regardless of size."""
    assert _classify_block(_b((0, 0, 200, 50), "BLUSH", rotation_deg=-10.4), "BLUSH") == "sfx"
    assert _classify_block(_b((0, 0, 100, 100), "しょぼん", rotation_deg=-27.6), "しょぼん") == "sfx"
    assert _classify_block(_b((0, 0, 200, 200), "THUD", rotation_deg=38.2), "THUD") == "sfx"


def test_short_wide_text_classified_as_sfx():
    """≤ 10 chars + aspect ≥ 1.4 → SFX (horizontal SFX pattern)."""
    # "IT HURTS!" 220×97, aspect=2.27, 8 chars → SFX
    assert _classify_block(_b((0, 0, 220, 97), "IT HURTS!"), "IT HURTS!") == "sfx"
    # "A RUSTLE" 168×70, aspect=2.40, 7 chars → SFX
    assert _classify_block(_b((0, 0, 168, 70), "A RUSTLE"), "A RUSTLE") == "sfx"
    # Page number "20" 44×24, aspect=1.83, 2 chars → SFX
    assert _classify_block(_b((0, 0, 44, 24), "20"), "20") == "sfx"


def test_short_tall_text_not_sfx():
    """≤ 10 chars but tall aspect → not SFX (could be vertical dialogue)."""
    # 50×150, aspect=0.33, 4 chars → dialogue
    assert _classify_block(_b((0, 0, 50, 150), "Yes!"), "Yes!") == "dialogue"


# ─── Narration detection ──────────────────────────────────────────────────


def test_long_text_classified_as_narration():
    """> 30 chars → narration (caption blocks)."""
    text = "DAMMIT, WHY CAN'T I REMEMBER ANY OF IT?!"
    assert _classify_block(_b((0, 0, 216, 209), text), text) == "narration"

    long_text = "I'M AN UGLY ELF, SO THIS MIGHT BE A COLD COMFORT, BUT..."
    assert _classify_block(_b((0, 0, 159, 218), long_text), long_text) == "narration"


# ─── Dialogue (default) ───────────────────────────────────────────────────


def test_medium_text_in_bubble_classified_as_dialogue():
    """11-30 chars in a regular bubble shape → dialogue."""
    assert _classify_block(
        _b((0, 0, 167, 127), "YOU'RE TELLING ME WOLVES..."),
        "YOU'RE TELLING ME WOLVES...",
    ) == "dialogue"
    assert _classify_block(
        _b((0, 0, 143, 111), "HOLD IT TIGHT..."),
        "HOLD IT TIGHT...",
    ) == "dialogue"


def test_short_squareish_text_classified_as_dialogue():
    """≤ 10 chars but aspect < 1.4 (squareish bubble) → dialogue, not SFX."""
    # Squarish bubble, short text
    assert _classify_block(
        _b((0, 0, 100, 100), "Hello"),
        "Hello",
    ) == "dialogue"


# ─── Rotation override beats aspect check ─────────────────────────────────


def test_rotation_override_beats_text_length():
    """Rotated long text is still SFX (e.g. tilted dialogue → onomatopoeia)."""
    # Long text but rotated 20° → SFX wins via rotation override
    long_rotated = "This is a long rotated thing for some reason"
    assert _classify_block(
        _b((0, 0, 400, 100), long_rotated, rotation_deg=20.0),
        long_rotated,
    ) == "sfx"
