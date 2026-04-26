"""LookAt agent — inspects page images via Agent protocol."""

from __future__ import annotations

from typoon.domain.bubble import Bubble, Session
from typoon.llm.ir import ContentPart, Message, ToolCallMsg, ToolDef, ToolResponse

from . import prompt
from .image import encode_page_jpeg
from .tools.visual_notes import VisualNotesArgs, submit_visual_notes


class LookAtAgent:
    """Inspects page images with key overlays and submits visual notes."""

    def __init__(
        self, session: Session, *, pages: list[int], keys: list[str],
        query: str, source_by_key: dict[str, str],
        polygon_by_key: dict[str, list[list[float]]],
    ) -> None:
        self._session = session
        self._pages = pages
        self._keys = keys
        self._query = query
        self._source_by_key = source_by_key
        self._polygon_by_key = polygon_by_key
        self._notes: dict[str, str] | None = None
        self._text_retries = 0

    def name(self) -> str:
        return "translate/lookat"

    def system_prompt(self) -> str:
        return prompt.LOOKAT_SYSTEM

    def user_message(self) -> Message:
        related = "\n".join(
            f"#{k}: {self._source_by_key.get(k, '')}" for k in self._keys
        )
        page_label = ", ".join(str(p) for p in self._pages)
        text = prompt.LOOKAT_USER.format(
            page_index=page_label, query=self._query, related_text=related,
        )
        parts: list[ContentPart] = [ContentPart.of_text(text)]
        source = self._session.source
        if source is not None and hasattr(source, "load_page"):
            for pi in self._pages:
                try:
                    img = source.load_page(pi)
                    labels = {k: self._polygon_by_key[k] for k in self._keys
                              if k in self._polygon_by_key}
                    parts.append(ContentPart.of_text(f"--- Page {pi} ---"))
                    parts.append(ContentPart.of_image(encode_page_jpeg(img, labels=labels)))
                except Exception:
                    continue
        return Message.user_parts(parts)

    def tools(self) -> list[ToolDef]:
        return [submit_visual_notes.definition]

    async def dispatch(self, call: ToolCallMsg) -> ToolResponse:
        if call.name != "submit_visual_notes":
            return ToolResponse(f"Unknown tool: {call.name}")
        try:
            args = VisualNotesArgs.model_validate_json(call.arguments)
        except Exception as e:
            return ToolResponse(f"Invalid: {e}")
        allowed = set(self._keys)
        self._notes = {n.key: n.note for n in args.notes if n.key in allowed}
        return ToolResponse("ok")

    def on_text(self, text: str | None) -> None:
        self._text_retries += 1

    def is_done(self) -> bool:
        return self._notes is not None

    def retry_prompt(self) -> str | None:
        if self._notes is not None:
            return None
        if self._text_retries >= 2:
            return None
        return "Do not respond with text. Call submit_visual_notes."

    def into_output(self) -> dict[str, str]:
        return self._notes or {}


async def look_at(
    session: Session, *, pages: list[int], keys: list[str],
    query: str, source_by_key: dict[str, str],
    key_map: dict[str, Bubble],
) -> dict[str, str]:
    """Run LookAt agent with key overlays on page images."""
    from typoon.llm.agent import run as agent_run
    polygon_by_key = {k: key_map[k].polygon for k in keys if k in key_map}
    agent = LookAtAgent(
        session, pages=pages, keys=keys,
        query=query, source_by_key=source_by_key,
        polygon_by_key=polygon_by_key,
    )
    result = await agent_run(session.context_provider, agent, hook=session.hook, max_turns=2)
    return result.output or {}
