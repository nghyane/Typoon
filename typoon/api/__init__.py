"""Typoon API — FastAPI application."""

from .app import app
from .models import ChapterOut, Progress, ProjectOut

__all__ = ["app", "ChapterOut", "Progress", "ProjectOut"]


def serve() -> None:
    import uvicorn
    uvicorn.run("typoon.api.app:app", host="0.0.0.0", port=8000, reload=True)
