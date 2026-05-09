"""OCR backend selection — explicit config or auto-detect by platform.

`create_ocr(source_lang, backend_name)` resolves to a single backend
appropriate for the language and host:

- `source_lang == "ja"` always uses `MangaOcrCropOcr` when available;
  Apple Vision / Lens / Tesseract recognise Japanese poorly on stylised
  manga fonts.
- Other languages route through `backend_name`. Explicit names raise on
  unavailability; `"auto"` walks the priority order
  google-lens → apple-vision → windows-ocr → tesseract and returns the
  first available backend.

The pipeline calls `create_ocr` once per chapter and reuses the
returned object; backends that hold a model (manga-ocr) keep it warm.
"""

from __future__ import annotations

from .types import CropOcr, PageOcr


_AUTO_PRIORITY: tuple[str, ...] = (
    "google_lens",
    "apple_vision",
    "windows_ocr",
    "tesseract",
)


def create_ocr(
    source_lang: str | None,
    backend: str = "auto",
    *,
    lens_endpoint: str | None = None,
) -> PageOcr | CropOcr:
    """Return the OCR backend to use for a project.

    `backend` accepts: "auto", "google_lens", "apple_vision",
    "windows_ocr", "tesseract", "manga_ocr".

    `lens_endpoint` overrides the Google Lens upstream URL — used to
    point through a Discord Activity proxy or any other reverse proxy
    that fronts `lensfrontend-pa.googleapis.com`.
    """
    if (source_lang or "").lower() == "ja":
        return _build_manga_ocr_or_raise()

    if backend == "auto":
        for name in _AUTO_PRIORITY:
            built = _try_build(name, lens_endpoint=lens_endpoint)
            if built is not None:
                return built
        raise RuntimeError(
            "No OCR backend available. Install one of: "
            "chrome-lens-py (cross-platform), "
            "tesseract (apt install tesseract-ocr / brew install tesseract), "
            "or run on macOS / Windows for native OCR."
        )

    built = _try_build(backend, lens_endpoint=lens_endpoint)
    if built is None:
        raise RuntimeError(
            f"OCR backend {backend!r} not available on this host. "
            f"Install its dependency or switch to backend='auto'."
        )
    return built


def _try_build(name: str, *, lens_endpoint: str | None = None) -> PageOcr | CropOcr | None:
    if name == "google_lens":
        from . import google_lens
        if google_lens.is_available():
            return google_lens.GoogleLensPageOcr(endpoint=lens_endpoint or None)
    elif name == "apple_vision":
        from . import apple_vision
        if apple_vision.is_available():
            return apple_vision.AppleVisionPageOcr()
    elif name == "windows_ocr":
        from . import windows
        if windows.is_available():
            return windows.WindowsOcrPageOcr()
    elif name == "tesseract":
        from . import tesseract
        if tesseract.is_available():
            return tesseract.TesseractPageOcr()
    elif name == "manga_ocr":
        return _build_manga_ocr_or_raise()
    else:
        raise ValueError(f"unknown OCR backend: {name!r}")
    return None


def _build_manga_ocr_or_raise() -> CropOcr:
    from . import manga_ocr
    if not manga_ocr.is_available():
        raise RuntimeError(
            "Japanese OCR requires manga-ocr. "
            "Install: pip install transformers torch fugashi"
        )
    return manga_ocr.MangaOcrCropOcr()
