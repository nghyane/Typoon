"""CLI output — save rendered pages to disk (JPEG + PDF)."""

from __future__ import annotations

from pathlib import Path

import cv2

from ..domain.bubble import Page


def save_pages(pages: list[Page], out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered = [p for p in pages if p.rendered is not None]
    if not rendered:
        return 0

    for page in rendered:
        out = out_dir / f"p{page.index:03d}.jpg"
        cv2.imwrite(
            str(out),
            cv2.cvtColor(page.rendered, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )

    try:
        from PIL import Image
        pil_pages = [Image.fromarray(p.rendered) for p in rendered]
        pdf_path = out_dir / "chapter.pdf"
        pil_pages[0].save(
            str(pdf_path), "PDF", save_all=True,
            append_images=pil_pages[1:], resolution=150,
        )
    except ImportError:
        pass

    return len(rendered)


class SingleFileSource:
    """Wraps a single image file as a ChapterSource."""

    def __init__(self, path: Path) -> None:
        self._path = path

    async def fetch(self) -> None:
        pass

    def page_count(self) -> int:
        return 1

    def load_page(self, index: int):
        bgr = cv2.imread(str(self._path))
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
