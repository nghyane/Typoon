"""FastAPI app — API only, workers run independently via `typoon work`."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from typoon.api.routes import projects, pages, sse

app = FastAPI(title="Typoon API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(projects.router)
app.include_router(pages.router)
app.include_router(sse.router)
