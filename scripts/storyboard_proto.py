"""Storyboard prototype — combine N consecutive pages into one image.

Two layout strategies tried:
  * grid_2x2   — manga-ish pages, square-ish target
  * row_4x1    — webtoon-ish strips, tall target

Output written to debug-runs/storyboard_proto/ for visual inspection.
No LLM, no project deps — pure PIL.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "cache" / "benchmark_visual"
OUT = ROOT / "debug-runs" / "storyboard_proto"
OUT.mkdir(parents=True, exist_ok=True)

# Use first 4 pages
PAGES = [SRC / f"p0{i}_original.jpg" for i in range(4)]

# Vision provider limits: most providers happy with ≤2048 longest edge.
# Going higher costs tokens with no readability gain at this density.
MAX_EDGE = 2048

# Padding / page label band so the agent can reference "page 1" precisely.
LABEL_BAND = 36
PAD = 8
LABEL_FONT_SIZE = 24


def _font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    # System default; PIL falls back if not found
    for candidate in [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, LABEL_FONT_SIZE)
            except OSError:
                pass
    return ImageFont.load_default()


def _label(img: Image.Image, text: str) -> Image.Image:
    """Add a top label band with page number, so the agent can address pages by index."""
    w, h = img.size
    out = Image.new("RGB", (w, h + LABEL_BAND), (32, 32, 32))
    out.paste(img, (0, LABEL_BAND))
    draw = ImageDraw.Draw(out)
    draw.text((PAD, 4), text, fill=(240, 240, 240), font=_font())
    return out


def _fit_within(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    """Scale to fit within a box, preserving aspect."""
    w, h = img.size
    scale = min(max_w / w, max_h / h)
    if scale >= 1.0:
        return img
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def build_grid_2x2(paths: list[Path], cell_w: int = 900, cell_h: int = 1200) -> Image.Image:
    """2×2 grid. Each cell up to cell_w × cell_h, labeled."""
    assert len(paths) <= 4
    cells: list[Image.Image] = []
    for i, p in enumerate(paths):
        raw = Image.open(p).convert("RGB")
        scaled = _fit_within(raw, cell_w, cell_h)
        labeled = _label(scaled, f"page {i}")
        cells.append(labeled)

    # Each row height = max of two cells in that row
    row_h_top = max(c.height for c in cells[:2]) if cells[:2] else 0
    row_h_bot = max(c.height for c in cells[2:4]) if cells[2:4] else 0
    col_w_l = max((cells[i].width for i in (0, 2) if i < len(cells)), default=0)
    col_w_r = max((cells[i].width for i in (1, 3) if i < len(cells)), default=0)

    canvas_w = col_w_l + col_w_r + PAD * 3
    canvas_h = row_h_top + row_h_bot + PAD * 3
    canvas = Image.new("RGB", (canvas_w, canvas_h), (20, 20, 20))

    positions = [
        (PAD, PAD),
        (col_w_l + PAD * 2, PAD),
        (PAD, row_h_top + PAD * 2),
        (col_w_l + PAD * 2, row_h_top + PAD * 2),
    ]
    for cell, pos in zip(cells, positions):
        canvas.paste(cell, pos)

    return _cap_edge(canvas, MAX_EDGE)


def build_row_4x1(paths: list[Path], cell_h: int = 1600) -> Image.Image:
    """4 columns side by side. Better for tall webtoon strips."""
    cells: list[Image.Image] = []
    for i, p in enumerate(paths):
        raw = Image.open(p).convert("RGB")
        scaled = _fit_within(raw, raw.width, cell_h)
        labeled = _label(scaled, f"page {i}")
        cells.append(labeled)

    row_h = max(c.height for c in cells)
    total_w = sum(c.width for c in cells) + PAD * (len(cells) + 1)
    canvas = Image.new("RGB", (total_w, row_h + PAD * 2), (20, 20, 20))

    x = PAD
    for cell in cells:
        canvas.paste(cell, (x, PAD))
        x += cell.width + PAD

    return _cap_edge(canvas, MAX_EDGE)


def _cap_edge(img: Image.Image, max_edge: int) -> Image.Image:
    """Downscale so longest edge <= max_edge."""
    w, h = img.size
    longest = max(w, h)
    if longest <= max_edge:
        return img
    scale = max_edge / longest
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def main() -> None:
    print(f"source pages: {[p.name for p in PAGES]}")

    grid = build_grid_2x2(PAGES)
    grid_path = OUT / "grid_2x2.jpg"
    grid.save(grid_path, quality=85, optimize=True)
    print(f"grid 2x2: {grid.size}  →  {grid_path}  ({grid_path.stat().st_size // 1024} KB)")

    row = build_row_4x1(PAGES)
    row_path = OUT / "row_4x1.jpg"
    row.save(row_path, quality=85, optimize=True)
    print(f"row  4x1: {row.size}  →  {row_path}  ({row_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
