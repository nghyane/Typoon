"""AOT inpainting HTTP service.

Loads aot-inpaint.onnx once at startup (full float32, no WASM overhead),
exposes:
  POST /inpaint   — packed body (RGB W*H*3 ++ mask W*H), ?w=X&h=Y
                    returns RGB W*H*3 inpainted bytes
  GET  /health    — liveness + session timing

Designed to run inside a Cloudflare Container (1 GiB RAM, basic instance).
Model + activations at 384×384 peak ~200 MB — well under the 1 GiB ceiling.
Cold start: ~3-5 s (image pull + Python boot + ONNX session load).
sleepAfter=5m in the DO wrapper keeps the warm session alive across
all pages of a chapter, so cold-start cost amortises once per batch.

Wire:
  body = rgb(W*H*3) ++ mask(W*H)   uint8, NCHW float32 conversion done here
  image in [-1, 1] zeroed where mask=1 (masked region to inpaint)
  mask  in {0, 1}   float32
  output: rgb(W*H*3) uint8 clipped to [0, 255]
"""

import os
import time
import logging
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response, JSONResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("inpaint")

MODEL_PATH = os.environ.get("MODEL_PATH", "/app/model.onnx")

log.info("loading model from %s", MODEL_PATH)
t0 = time.time()
sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
_load_ms = round((time.time() - t0) * 1000)
log.info("session ready in %d ms", _load_ms)

app = FastAPI()


@app.get("/health")
def health():
    return {"ok": True, "model": os.path.basename(MODEL_PATH), "load_ms": _load_ms}


@app.post("/inpaint")
async def inpaint(req: Request):
    """Inpaint one padded tile.

    Body layout:
      bytes[0 : W*H*3]        RGB uint8 (source pixels, masked region can be anything)
      bytes[W*H*3 : W*H*4]    mask uint8 (>=127 → region to inpaint)

    Query params:
      w, h  — tile dimensions (must both be multiples of 8)

    Returns:
      W*H*3 bytes   RGB uint8 of fully inpainted tile
    """
    try:
        W = int(req.query_params["w"])
        H = int(req.query_params["h"])
    except (KeyError, ValueError):
        raise HTTPException(400, "?w and ?h required and must be integers")

    if W <= 0 or H <= 0:
        raise HTTPException(400, f"invalid dimensions W={W} H={H}")
    if W % 8 != 0 or H % 8 != 0:
        raise HTTPException(400, f"W={W} H={H} must be multiples of 8")

    rgb_len  = W * H * 3
    expected = rgb_len + W * H
    body = await req.body()
    if len(body) != expected:
        raise HTTPException(400, f"body={len(body)} expected={expected}")

    t_pre0 = time.time()
    raw  = np.frombuffer(body, dtype=np.uint8)
    rgb  = raw[:rgb_len].reshape(H, W, 3).astype(np.float32)
    mask = (raw[rgb_len:].reshape(H, W) >= 127).astype(np.float32)

    # Image in [-1, 1], zeroed inside mask (model sees only the valid context).
    img_f32 = (rgb / 127.5 - 1.0) * (1.0 - mask[:, :, None])  # (H, W, 3)
    # NCHW
    img_nchw  = img_f32.transpose(2, 0, 1)[None]               # (1, 3, H, W)
    mask_nchw = mask[None, None]                                # (1, 1, H, W)
    t_pre = (time.time() - t_pre0) * 1000

    t_inf0 = time.time()
    result = sess.run(None, {"image": img_nchw, "mask": mask_nchw})
    t_inf  = (time.time() - t_inf0) * 1000

    t_post0 = time.time()
    out_nchw = result[0][0]                                     # (3, H, W) float32
    out_hwc  = out_nchw.transpose(1, 2, 0)                     # (H, W, 3)
    out_u8   = np.clip(np.round((out_hwc + 1.0) * 127.5), 0, 255).astype(np.uint8)
    t_post   = (time.time() - t_post0) * 1000

    log.info(
        "inpaint %dx%d pre=%.0fms inf=%.0fms post=%.0fms",
        W, H, t_pre, t_inf, t_post,
    )
    return Response(
        content=out_u8.tobytes(),
        media_type="application/octet-stream",
        headers={
            "X-Timings": f"pre={t_pre:.0f},inf={t_inf:.0f},post={t_post:.0f}",
        },
    )
