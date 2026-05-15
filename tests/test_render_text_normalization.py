"""Unicode normalisation for rendered text.

Vietnamese diacritics arriving from LLM providers can be in NFD
(decomposed) form. The embedded render font only has precomposed
glyphs, so NFD text renders with phantom spaces where the combining
mark should sit. `_normalize_for_render` forces NFC at the translate
boundary; this test pins that contract.
"""

from __future__ import annotations

import unicodedata

from typoon.stages.translate import _normalize_for_render


def test_normalize_for_render_collapses_nfd_to_nfc():
    word = unicodedata.normalize("NFD", "Tôi nghĩa")
    assert word != "Tôi nghĩa"  # sanity: NFD differs from NFC

    out = _normalize_for_render(word)
    assert out == "Tôi nghĩa"
    assert unicodedata.is_normalized("NFC", out)


def test_normalize_for_render_idempotent_on_nfc():
    nfc = "Tôi nghĩa rằng — đó là ý"  # already precomposed
    assert _normalize_for_render(nfc) == nfc


def test_normalize_for_render_handles_empty_and_whitespace():
    assert _normalize_for_render("") == ""
    assert _normalize_for_render("   ") == "   "


def test_normalize_for_render_preserves_ascii():
    assert _normalize_for_render("Hello, world!") == "Hello, world!"


def test_normalize_for_render_preserves_cjk_and_punctuation():
    """CJK + punctuation should pass through unchanged."""
    for text in ["しょぼん", "你好", "안녕", "—...!?"]:
        assert _normalize_for_render(text) == text


def test_normalize_for_render_handles_mixed_nfd_chunks():
    """Real-world LLM output can mix NFC and NFD on different words."""
    mixed = (
        "T"
        + unicodedata.normalize("NFD", "ôi")
        + " "
        + "nghĩa"            # already NFC
        + " "
        + unicodedata.normalize("NFD", "rằng")
    )
    out = _normalize_for_render(mixed)
    assert out == "Tôi nghĩa rằng"
    assert unicodedata.is_normalized("NFC", out)


def test_normalize_for_render_drops_no_visible_characters():
    """NFC normalisation must not lose any visible character."""
    src = "Đứa con của Quảng Lăng đã được đưa về."
    nfd = unicodedata.normalize("NFD", src)
    out = _normalize_for_render(nfd)
    # NFC has 1 grapheme = 1 codepoint for Vietnamese letters
    assert len(out) == len(src)
    assert out == src
