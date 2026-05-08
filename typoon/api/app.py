"""FastAPI app — API only, workers run independently via `typoon work`."""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from typoon.api.deps import get_store
from typoon.api.routes import (
    auth, bubbles, glossary, me, projects, search, sse, upload, workers,
)
from typoon.config import load_config
from typoon.storage import Store

app = FastAPI(title="Typoon API")

_config, _paths = load_config()
_paths.ensure()

# CORS — closed community, only the configured web origin plus the
# Discord Activity sandbox origins (Phase 2). `allow_origins=["*"]` was
# wrong for a closed deploy and would have to be tightened anyway when
# the SPA started carrying JWTs in Authorization headers.
_origins = [
    _config.server.public_web_url,
    "https://discord.com",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_origin_regex=r"https://[a-z0-9-]+\.discordsays\.com",
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(me.router)
app.include_router(projects.router)
app.include_router(upload.router)
app.include_router(bubbles.router)
app.include_router(glossary.router)
app.include_router(workers.router)
app.include_router(search.router)
app.include_router(sse.router)


@app.get("/api/healthz")
async def healthz(db: Store = Depends(get_store)):
    """Liveness + DB connectivity. Used by reverse proxies / Tailscale
    health probes. Returns 200 on a successful Postgres ping, 503
    otherwise."""
    try:
        await db.ping()
    except Exception as e:  # asyncpg can raise many shapes
        raise HTTPException(503, f"db: {e}") from e
    return {"ok": True}


# Public render archives, dev/local serving only. Production uses an
# external CDN (bunle CDN proxying HF) and skips this mount.
# StaticFiles supports HTTP Range so the in-browser bunle reader can
# request slices without pulling the whole archive.
# Must mount BEFORE the broader /files mount so requests to
# /files/render/<token>.bnl route here, not to the project files mount.
_archive_dir = _paths.artifacts / "render"
_archive_dir.mkdir(parents=True, exist_ok=True)
app.mount("/files/render", StaticFiles(directory=str(_archive_dir)), name="render")

# Static project files (covers, future thumbnails) served via sendfile.
app.mount("/files", StaticFiles(directory=str(_paths.projects)), name="files")
