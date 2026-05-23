"""InpaintPipeline — pulls inputs, calls Rust runtime, pushes output.

Both CLI and container entrypoints instantiate this; only the storage
and sink adapters differ.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from typoon_inpaint.artifact_sink import ArtifactSink, NullSink
from typoon_inpaint.storage import Storage

log = logging.getLogger(__name__)


class InpaintPipeline:
    def __init__(
        self,
        runtime,          # InpaintRuntime (PyO3)
        storage:  Storage,
        sink:     ArtifactSink | None = None,
        concurrency: int = 4,
    ) -> None:
        self._rt   = runtime
        self._fs   = storage
        self._sink = sink or NullSink()
        self._sem  = asyncio.Semaphore(concurrency)

    async def page(self, job_id: int, page_index: int) -> str:
        """Inpaint one page. Returns output R2 key."""
        async with self._sem:
            # Scan msgpack now embeds the InpaintPlan.
            # InpaintRuntime.inpaint_page_async accepts scan or bare plan.
            scan  = await self._fs.get(f"scan/{job_id}/{page_index:04d}.msgpack")
            jpeg  = await self._fs.get(f"prepared/{job_id}/{page_index:04d}.jpg")
            out_key = f"inpaint/{job_id}/{page_index:04d}.png"

            page_sink = self._sink.subdir(f"page_{page_index:04d}")
            debug_dir = str(page_sink.path) if not isinstance(page_sink, NullSink) else None

            png: bytes = await asyncio.to_thread(
                self._rt.inpaint_page,
                bytes(jpeg), bytes(scan), debug_dir,
            )
            await self._fs.put(out_key, bytes(png), "image/png")
            log.info("page %04d done → %s", page_index, out_key)
            return out_key

    async def chapter(self, job_id: int, page_indices: list[int]) -> list[str]:
        return list(await asyncio.gather(*[self.page(job_id, i) for i in page_indices]))
