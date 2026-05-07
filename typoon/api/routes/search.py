"""Full-text search across bubbles, translations, briefs, glossary.

The DB already maintains FTS indexes for each table. This endpoint fans
out per scope and folds everything into a uniform result shape.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from typoon.api.deps import get_store, require_user
from typoon.api.models import SearchHit, SearchResults
from typoon.api.routes._shared import require_project_view
from typoon.storage import Store

router = APIRouter(
    prefix="/api", tags=["search"],
    dependencies=[Depends(require_user)],
)

_VALID_SCOPES = {"all", "translations", "briefs", "glossary"}


@router.get("/search", response_model=SearchResults)
async def search(
    q:     str = Query(..., min_length=1),
    pid:   int = Query(...,  alias="project_id"),
    scope: str = Query("all"),
    limit: int = Query(20, ge=1, le=100),
    user:  dict  = Depends(require_user),
    db:    Store = Depends(get_store),
):
    if scope not in _VALID_SCOPES:
        raise HTTPException(400, f"scope must be one of: {sorted(_VALID_SCOPES)}")
    await require_project_view(pid, user, db)

    hits: list[SearchHit] = []

    if scope in ("all", "translations"):
        for line in await db.search_context(pid, [q], scope="translations", limit=limit):
            hits.append(SearchHit(kind="translation", text=line))

    if scope in ("all", "briefs"):
        for line in await db.search_briefs(pid, [q], limit=limit):
            hits.append(SearchHit(kind="brief", text=line))

    if scope in ("all", "glossary"):
        for row in await db.glossary_search(pid, q):
            hits.append(SearchHit(
                kind="glossary",
                text=f"{row['source_term']} → {row['target_term']}"
                     + (f"  ({row['notes']})" if row.get("notes") else ""),
            ))

    return SearchResults(hits=hits[:limit])
