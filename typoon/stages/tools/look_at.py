"""look_at tool factory.

Vision lookup for ambiguous speakers/listeners. Budget is structured
around the real cost driver — page images served to the vision model —
not the number of bubble keys answered:

- Primary cap: `max_page_images` (default 6). Each page passed to
  vision counts once; pages already served in an earlier call are
  free on re-request.
- Backstop cap: `max_calls` (default 6). Only protects against
  pathological infinite loops.

The agent must commit to *which* keys are unresolved and *why text
alone is insufficient* (`tried` field). Generic filler ("text unclear",
"verify speakers") is rejected at schema or regex level. Per-key notes
from prior calls are deduplicated and carried forward, so the agent
cannot re-ask the same question to burn budget or stall the loop.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from typoon.llm.ir import ToolResponse
from typoon.llm.tool import Tool, tool

MAX_PAGE_IMAGES = 6
MAX_CALLS       = 6
MIN_TRIED_CHARS = 30

# Phrases that pretend to justify a vision call but say nothing
# concrete. Reject — force the agent to point at the actual ambiguity.
_EMPTY_TRIED_RE = re.compile(
    r"^(?:"
    r"text(?:\s+is)?\s+(?:unclear|ambiguous|insufficient|not\s+clear)|"
    r"need(?:s|ed)?\s+to\s+(?:check|verify|confirm)|"
    r"(?:to\s+)?(?:verify|confirm|check)(?:\s+speakers?)?|"
    r"speakers?\s+(?:unclear|ambiguous)|"
    r"ambiguous|unclear|unsure"
    r")[\s.]*$",
    re.IGNORECASE,
)


class LookAtArgs(BaseModel):
    keys: list[str] = Field(
        min_length=1,
        description=(
            "Bubble keys whose speaker/listener you cannot resolve from "
            "text alone. Batch ALL unresolved keys for the chapter into "
            "as few calls as possible — pages already served in a prior "
            "call are free to re-reference, so cluster keys by page."
        ),
    )
    pages: list[int] = Field(
        min_length=1,
        max_length=3,
        description=(
            "Page indices the bubbles live on. Maximum 3 pages per call. "
            "Pages count against a per-chapter image budget; re-using a "
            "page from an earlier call costs nothing."
        ),
    )
    query: str = Field(
        min_length=8,
        description=(
            "Concrete visual question, e.g. 'Identify which child says #ABC123 "
            "vs #DEF456 — sister or mother?'. Not a paraphrase of 'check speakers'."
        ),
    )
    tried: str = Field(
        min_length=MIN_TRIED_CHARS,
        description=(
            "Required: explain why text alone is insufficient for these "
            "specific keys (≥30 chars). Cite the bubble content or the "
            "missing signal. Generic phrases like 'text unclear' or "
            "'verify speakers' are rejected."
        ),
    )


@tool
async def look_at(args: LookAtArgs, run) -> ToolResponse:
    """Inspect page images when speaker/listener is genuinely unresolvable from text.

    Page images are the real cost driver — the per-chapter budget is
    counted in pages, not calls. Cluster bubbles by page, batch ALL
    unresolved keys into as few calls as possible.
    """
    return await run(args)


def make_look_at(
    ctx, prepared, reader, keyed,
    *,
    max_page_images: int = MAX_PAGE_IMAGES,
    max_calls: int = MAX_CALLS,
) -> Tool:
    """Build a disciplined look_at tool bound to one chapter.

    State (per chapter):
      calls         — number of vision calls made so far
      served_pages  — page indices already shown to the vision model;
                      free to re-reference
      notes         — accumulated {key: observation} from prior calls;
                      used both as the de-dup table and as carry-forward
                      context when the agent mixes old + new keys
    """
    calls = 0
    served_pages: set[int] = set()
    notes: dict[str, str] = {}

    async def run(args: LookAtArgs) -> ToolResponse:
        nonlocal calls

        # Reject filler `tried` content. Pydantic min_length already
        # filters short strings; this catches the "text unclear" /
        # "verify speakers" pattern at full length.
        if _EMPTY_TRIED_RE.match(args.tried.strip()):
            return ToolResponse(
                "Rejected: `tried` reads as generic filler. "
                "Quote the specific bubble content or missing signal "
                "that text alone cannot supply."
            )

        new_keys  = [k for k in args.keys if k not in notes]
        new_pages = [p for p in args.pages if p not in served_pages]

        # Pure dedup: every requested key already answered, no new
        # page to serve. Return prior notes; do not tick budgets.
        if not new_keys and not new_pages:
            prior = "\n".join(f"#{k}: {notes[k]}" for k in args.keys if k in notes)
            return ToolResponse(
                "All requested keys already resolved by an earlier "
                "look_at call, and no new pages requested. Use these "
                "notes; do not call again for the same keys.\n" + prior
            )

        # Backstops. Set high enough that legitimate chapters never
        # hit them; if they do, that's a signal something is wrong
        # upstream (e.g. scan over-segmented bubbles) — fall back to
        # neutral phrasing for remaining unresolved bubbles.
        if calls >= max_calls:
            return ToolResponse(
                f"look_at call cap reached ({max_calls}). For unresolved "
                "keys, write 'Uncertain speaker; use neutral phrasing' "
                "in bubble_notes and submit."
            )
        # Budget gate: only NEW pages cost. If serving them all would
        # blow the cap, reject before running vision.
        if new_pages and len(served_pages) + len(new_pages) > max_page_images:
            remaining = max_page_images - len(served_pages)
            return ToolResponse(
                f"Page-image budget would be exceeded "
                f"({len(served_pages)}/{max_page_images} served, "
                f"{len(new_pages)} new requested, {remaining} remaining). "
                "Drop pages or reuse already-served pages: "
                f"{sorted(served_pages) or '∅'}. For bubbles you can no "
                "longer inspect, write 'Uncertain speaker; use neutral "
                "phrasing' in bubble_notes."
            )

        # Run vision. Request notes for the new keys only; vision sees
        # the union of pages (new + already-served re-shown so the
        # model has the full context). Note: prepared.read_rgb is
        # idempotent; re-shown pages don't double-count budget.
        from typoon.stages.look_at import look_at as _look_at
        result = await _look_at(
            ctx, prepared, reader,
            pages=args.pages, keys=new_keys, query=args.query, keyed=keyed,
        )
        calls += 1
        served_pages.update(new_pages)

        accepted: dict[str, str] = {}
        for k, v in result.items():
            if k not in notes:
                accepted[k] = v
                notes[k] = v

        # Build response: new observations + prior notes for any
        # already-resolved keys the agent re-included.
        already = [k for k in args.keys if k in notes and k not in accepted]
        parts: list[str] = []
        if accepted:
            parts.append("\n".join(f"#{k}: {v}" for k, v in accepted.items()))
        if already:
            parts.append(
                "Already resolved (carried forward):\n"
                + "\n".join(f"#{k}: {notes[k]}" for k in already)
            )
        unanswered = [k for k in new_keys if k not in accepted]
        if unanswered:
            parts.append(
                "Vision returned no observation for: "
                + ", ".join(f"#{k}" for k in unanswered)
                + ". Treat as unresolved; use neutral phrasing in bubble_notes."
            )

        # Soft nudge once two or more vision calls have been spent.
        nudge = ""
        if calls >= 2:
            nudge = (
                f"\n[note: {calls}/{max_calls} calls, "
                f"{len(served_pages)}/{max_page_images} page-images served. "
                "Cluster remaining keys onto already-served pages where "
                "possible — those re-references are free.]"
            )

        body = "\n\n".join(p for p in parts if p) or "No visual notes returned."
        return ToolResponse(body + nudge)

    return look_at(run=run)
