"""FastAPI app — API + optional storage role.

The mounted routers depend on `TYPOON_API_ROLE`:

  api      (default)  user/project/bubble/etc routes; no blob endpoint.
  storage             only /api/blobs/* + /api/healthz; pipeline node.
  full                api + storage; single-host deploy.

Workers (`typoon work`) run independently from this app; they reach a
storage node via `HttpBlobStore`.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from typoon.api.deps import get_store
from typoon.api.middleware import RequestIDMiddleware
from typoon.api.routes import (
    auth, blobs, bubbles, glossary, me, project_events,
    projects, search, upload, workers,
)
from typoon.config import load_config
from typoon.storage import Store

_role = os.environ.get("TYPOON_API_ROLE", "full").lower()
_serve_api     = _role in ("api", "full")
_serve_storage = _role in ("storage", "full")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Per-process startup/shutdown hooks.

    - app.state.shutdown is an asyncio.Event long-lived endpoints can
      observe (notably the SSE event stream) so they exit immediately
      when uvicorn starts a graceful shutdown rather than holding the
      process at "Waiting for connections to close".
    - The EventBus listener task is started here so reconnects stay
      transparent to subscribers; without it the first /api/events
      hit would create the listener and a Postgres restart would leave
      the bus blind until a client reconnected.
    """
    from typoon.api.deps import get_bus
    app.state.shutdown = asyncio.Event()
    bus = await get_bus()
    if _serve_api:
        # Storage-only role doesn't host SSE, no need to listen.
        await bus.start()
    try:
        yield
    finally:
        app.state.shutdown.set()
        await bus.close()


app = FastAPI(title="Typoon API", lifespan=_lifespan)

_config, _paths = load_config()
_paths.ensure()

# CORS — closed community, only the configured web origin plus the
# Discord Activity sandbox origins. `storage` role serves only worker
# traffic (no browser), so a permissive CORS doesn't widen attack
# surface there beyond what worker tokens already gate.
#
# Browser extensions (RFC-009) authenticate with `Authorization: Bearer`
# tokens — no cookie auth, no CSRF surface — so allowing the
# extension origin patterns here only widens the *advertised* CORS,
# not the actual auth surface. Token revocation in /api/me/tokens is
# the real lever.
_origins = [
    _config.server.public_web_url,
    "https://discord.com",
]
_extension_origin_re = (
    # Chrome / Edge: 32-char a-p ID.
    r"chrome-extension://[a-p]{32}"
    # Firefox: UUID with hyphens.
    r"|moz-extension://[0-9a-f-]{36}"
)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_origin_regex=(
        r"https://[a-z0-9-]+\.discordsays\.com"
        r"|" + _extension_origin_re
    ),
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "ETag"],
)

# Host header validation. Behind Cloudflare Tunnel (or any proxy) the
# app is reachable only via `_config.server.trusted_hosts` — rejecting
# other Host values blocks cache-poisoning / host-header injection.
# `["*"]` disables the check (dev only).
if _config.server.trusted_hosts and _config.server.trusted_hosts != ["*"]:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=_config.server.trusted_hosts,
    )

# Proxy header trust. Cloudflare Tunnel (and other reverse proxies)
# forward the real client IP and scheme via X-Forwarded-{For,Proto}.
# Without this middleware the app sees the tunnel daemon's loopback
# address as the client, breaking IP-based logging and any per-IP
# policy. trusted_hosts="*" is acceptable here because the tunnel is
# the only ingress — uvicorn already binds to 0.0.0.0 and we trust
# whatever proxy handed off the connection.
app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")

if _serve_api:
    app.include_router(auth.router)
    app.include_router(me.router)
    app.include_router(projects.router)
    app.include_router(upload.router)
    app.include_router(upload.local_router)
    app.include_router(bubbles.router)
    app.include_router(glossary.router)
    app.include_router(workers.router)
    app.include_router(search.router)
    app.include_router(project_events.router)

if _serve_storage:
    app.include_router(blobs.router)


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
