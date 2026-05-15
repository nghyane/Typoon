"""End-to-end smoke test for new vision pipeline.

Runs scan_chapter on a sample chapter using the lens preset, writes
debug-runs/poc_v4/<ch>/ artifacts, prints per-stage timing.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from typoon.adapters.mask_store import MaskStore  # noqa: E402
from typoon.adapters.prepared_reader import PreparedReader  # noqa: E402
from typoon.adapters.vision_runtime import VisionRuntimeAdapter  # noqa: E402
from typoon.domain.prepared import Chapter as PreparedChapter, Page as PreparedPage  # noqa: E402
from typoon.runs.artifacts import FileArtifactSink  # noqa: E402
from typoon.stages.scan import scan_chapter  # noqa: E402
from typoon.vision.pipeline import VisionPipelineSpec  # noqa: E402


# ─── Reader shim — feed raw webp dir as a PreparedReader-shaped object ────


class _DirReader:
    """PreparedReader-shaped adapter over a directory of webp pages."""

    def __init__(self, paths: list[Path]) -> None:
        self._paths = paths

    @property
    def page_count(self) -> int:
        return len(self._paths)

    def chapter(self) -> PreparedChapter:
        pages: list[PreparedPage] = []
        for i, p in enumerate(self._paths):
            with Image.open(p) as img:
                w, h = img.size
            pages.append(PreparedPage(index=i, width=w, height=h))
        return PreparedChapter(source=str(self._paths[0].parent), pages=tuple(pages))

    def read_rgb(self, index: int) -> np.ndarray:
        with Image.open(self._paths[index]) as img:
            return np.asarray(img.convert("RGB"))


# ─── Driver ───────────────────────────────────────────────────────────────


async def run_chapter(chapter_dir: Path, out_dir: Path, preset: str = "lens") -> None:
    paths = sorted(chapter_dir.glob("*.webp"))
    if not paths:
        print(f"no webp files in {chapter_dir}")
        return
    reader = _DirReader(paths)
    prepared = reader.chapter()
    print(f"chapter: {chapter_dir.name}  pages: {len(paths)}  preset: {preset}")

    spec = VisionPipelineSpec.preset(preset)
    print(f"  spec: {spec}")

    adapter, _, _ = VisionRuntimeAdapter.from_config(source_lang="en")
    # Override spec via fresh build (instead of touching config.toml)
    from typoon.vision.runtime import build_vision_runtime
    runtime = build_vision_runtime(
        spec, models_dir=adapter.hub.dir, source_lang="en",
    )

    sink = FileArtifactSink(out_dir.parent, out_dir.name)

    t0 = time.perf_counter()
    result = await scan_chapter(
        prepared, reader, runtime,
        source_lang="en",
        chapter_id=0,
        artifacts=sink,
    )
    elapsed = time.perf_counter() - t0
    print(f"  scan elapsed: {elapsed:.2f}s "
          f"({elapsed/len(paths):.2f}s/page)")

    bubbles = result.bubble_records()
    by_page: dict[int, list[dict]] = {}
    for b in bubbles:
        by_page.setdefault(b["page_index"], []).append(b)

    summary = {
        "chapter":      chapter_dir.name,
        "preset":       preset,
        "n_pages":      len(paths),
        "elapsed_s":    round(elapsed, 3),
        "per_page_s":   round(elapsed / len(paths), 3),
        "n_bubbles":    len(bubbles),
        "bubbles_per_page": {str(k): len(v) for k, v in sorted(by_page.items())},
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"  summary: {summary_path}")


def main() -> None:
    chapters = sys.argv[1:] or ["ch001"]
    preset = "lens"
    if chapters and chapters[-1] in ("lens", "lens_balanced", "offline", "manga_ja"):
        preset = chapters.pop()

    fixtures = ROOT / "tests" / "fixtures" / "sample_chapters"
    for ch in chapters:
        ch_dir = fixtures / ch
        out_dir = ROOT / "debug-runs" / "poc_v4" / f"{ch}_{preset}"
        out_dir.mkdir(parents=True, exist_ok=True)
        asyncio.run(run_chapter(ch_dir, out_dir, preset))


if __name__ == "__main__":
    main()
