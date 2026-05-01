"""look_at — single vision query: page images + question → answer."""

from __future__ import annotations

from typoon.adapters.session import Session
from typoon.domain.prepared import Chapter
from typoon.domain.scan import Bubble as ScannedBubble
from typoon.llm.ir import ContentPart, Message


async def look_at(
    session: Session,
    prepared: Chapter,
    *,
    pages: list[int],
    keys: list[str],
    query: str,
    key_map: dict[str, ScannedBubble],
) -> dict[str, str]:
    """Query vision model with page images. Returns {key: note} observations."""
    source_by_key = {k: key_map[k].source_text for k in keys if k in key_map}
    polygon_by_key = {k: key_map[k].box.polygon for k in keys if k in key_map}

    parts: list[ContentPart] = []

    # Build text context
    related = "\n".join(f"#{k}: {source_by_key.get(k, '')}" for k in keys)
    page_label = ", ".join(str(p) for p in pages)
    parts.append(ContentPart.of_text(
        f"Pages: {page_label}\n"
        f"Query: {query}\n\n"
        f"Bubbles to observe:\n{related}\n\n"
        f"For each bubble key, note: speaker identity/gender, emotion/tone, "
        f"whether text is dialogue or SFX, pronoun/address cues visible in the image. "
        f"Reply as a list: #KEY: observation"
    ))

    # Attach page images
    for pi in pages:
        try:
            import cv2
            from .image import encode_page_jpeg
            path = prepared.page_path(pi)
            bgr = cv2.imread(str(path))
            if bgr is None:
                continue
            import numpy as np
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            labels = {k: polygon_by_key[k] for k in keys if k in polygon_by_key}
            parts.append(ContentPart.of_text(f"--- Page {pi} ---"))
            parts.append(ContentPart.of_image(encode_page_jpeg(img, labels=labels)))
        except Exception:
            continue

    if len(parts) == 1:
        # No images loaded — skip vision call
        return {}

    system = (
        "You are a visual assistant for comic translation. "
        "Inspect the page images and answer questions about speech bubbles. "
        "Be concise — one line per key."
    )
    messages = [Message.system(system), Message.user_parts(parts)]

    resp = await session.vision_provider.call(messages, [])
    text = resp.text or ""

    # Parse "#KEY: note" lines
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
