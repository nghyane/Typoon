"""FastAPI app — API only, workers run independently via `typoon work`."""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from typoon.adapters.event_bus import is_postgres
from typoon.api.routes import (
    bubbles, glossary, pages, projects, search, sources, sse, workers,
)
from typoon.config import load_config

app = FastAPI(title="Typoon API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(projects.router)
app.include_router(sources.router)
app.include_router(bubbles.router)
app.include_router(glossary.router)
app.include_router(workers.router)
app.include_router(search.router)
app.include_router(pages.router)
app.include_router(sse.router)

# Static project files (covers, future thumbnails) served via sendfile.
_config, _paths = load_config()
_paths.ensure()
app.mount("/files", StaticFiles(directory=str(_paths.projects)), name="files")

# Deploy-mode validation: SQLite ⇒ single-process only. If TYPOON_ROLE
# explicitly says "api" (i.e. workers run elsewhere), Postgres is required.
_role = os.environ.get("TYPOON_ROLE", "").strip().lower()
if _role == "api" and not is_postgres(_config.database_url):
    raise RuntimeError(
        "TYPOON_ROLE=api requires postgresql:// database_url. "
        "SQLite is single-process only — run `typoon work --role full` to "
        "embed workers in the API process, or switch to Postgres."
    )
