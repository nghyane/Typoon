"""look_at — single vision query: page images + question → observations."""

from __future__ import annotations

from typoon.adapters.prepared_reader import PreparedReader
from typoon.stages.image import encode_page_jpeg
from typoon.domain.prepared import Chapter as PreparedChapter
from typoon.domain.scan import BubbleKey
from typoon.llm.ir import ContentPart, Message

_MAX_PAGES = 3


async def look_at(
    ctx,  # TranslateCtx — avoid circular import
    prepared: PreparedChapter,
    reader: PreparedReader,
    *,
    pages: list[int],
    keys: list[str],
    query: str,
    keyed: list[BubbleKey],
) -> dict[str, str]:
    """Query vision model with page images. Returns {key: observation}."""
    key_map = {bk.key: bk for bk in keyed}
    relevant = {k: key_map[k] for k in keys if k in key_map}
    pages = pages[:_MAX_PAGES]

    if not relevant:
        return {}

    related = "\n".join(f"#{k}: {bk.source_text}" for k, bk in relevant.items())
    parts: list[ContentPart] = [ContentPart.of_text(
        f"Pages: {', '.join(str(p) for p in pages)}\n"
        f"Query: {query}\n\n"
        f"Bubbles to observe:\n{related}\n\n"
        f"For each bubble key, note: speaker identity/gender, emotion/tone, "
        f"whether text is dialogue or SFX. Reply as: #KEY: observation"
    )]

    labels = {k: bk.box.polygon for k, bk in relevant.items()}
    for pi in pages:
        if pi < 0 or pi >= prepared.page_count:
            continue
        img = reader.read_rgb(pi)
        parts.append(ContentPart.of_text(f"--- Page {pi} ---"))
        parts.append(ContentPart.of_image(encode_page_jpeg(img, labels=labels)))

    if len(parts) == 1:
        return {}  # no images loaded

    system = (
        "You are a visual assistant for comic translation. "
        "Inspect the page images and answer questions about speech bubbles. "
        "Be concise — one line per key."
    )
    messages = [Message.system(system), Message.user_parts(parts)]
    resp = await ctx.vision_provider.call(messages, [])
    text = resp.text or ""

    notes: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("#"):
            continue
        key_part, _, note = line.partition(":")
        key = key_part.lstrip("#").strip()
        if key in key_map:
            notes[key] = note.strip()
    return notes
