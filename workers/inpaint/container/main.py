"""CF Container entrypoint — thin FastAPI wrapper around InpaintPipeline.

Same InpaintPipeline used by local CLI; only storage adapter differs.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

log = logging.getLogger("inpaint-container")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI()

_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    from typoon_inpaint import InpaintRuntime
    from typoon_inpaint_py.pipeline import InpaintPipeline
    from typoon_inpaint_py.storage import R2Storage

    rt = InpaintRuntime(
        os.environ.get("MODEL_PATH", "/app/model.safetensors"),
    )
    storage = R2Storage.from_env()
    _pipeline = InpaintPipeline(
        runtime=rt,
        storage=storage,
        concurrency=int(os.environ.get("CONCURRENCY", "4")),
    )
    log.info("pipeline ready")
    return _pipeline


@app.get("/health")
async def health():
    return {"ok": True, "service": "inpaint-container",
            "model_loaded": _pipeline is not None}


@app.get("/warm")
async def warm():
    _get_pipeline()
    return {"ok": True}


@app.post("/inpaint-chapter")
async def inpaint_chapter(req: Request):
    body         = await req.json()
    job_id       = body.get("job_id")
    page_indices = body.get("page_indices", [])
    if not page_indices:
        raise HTTPException(400, "page_indices required")

    pipeline = _get_pipeline()
    t0       = time.perf_counter()

    results = []
    async def one(page_index: int) -> dict:
        t = time.perf_counter()
        try:
            key = await pipeline.page(int(job_id), int(page_index))
            return {"page_index": page_index, "output_key": key,
                    "wall_ms": round((time.perf_counter() - t) * 1000)}
        except Exception as e:
            log.exception("page %d failed", page_index)
            return {"page_index": page_index, "error": str(e),
                    "wall_ms": round((time.perf_counter() - t) * 1000)}

    results = await asyncio.gather(*[one(p) for p in page_indices])
    results.sort(key=lambda r: r["page_index"])

    return JSONResponse({
        "results":         results,
        "wall_total_ms":   round((time.perf_counter() - t0) * 1000),
        "concurrency_used": pipeline._sem._value,
    })
