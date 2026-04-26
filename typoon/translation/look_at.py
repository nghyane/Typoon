"""LookAt agent — inspects page images via Agent protocol."""

from __future__ import annotations

from typoon.domain.bubble import Session
from typoon.llm.ir import ContentPart, Message, ToolCallMsg, ToolDef, ToolResponse

from . import prompt
from .tools.view_page import encode_page_jpeg
from .tools.visual_notes import VisualNotesArgs, submit_visual_notes


class LookAtAgent:
    """Inspects page images and submits keyed visual notes."""

    def __init__(
        self, session: Session, *, pages: list[int], keys: list[str],
        query: str, source_by_key: dict[str, str],
    ) -> None:
        self._session = session
        self._pages = pages
        self._keys = keys
        self._query = query
        self._source_by_key = source_by_key
        self._notes: dict[str, str] | None = None

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
                    parts.append(ContentPart.of_text(f"--- Page {pi} ---"))
                    parts.append(ContentPart.of_image(encode_page_jpeg(img)))
                except Exception:
                    continue
        return Message.user_parts(parts)

    def tools(self) -> list[ToolDef]:
        return [submit_visual_notes.definition]

    async def dispatch(self, call: ToolCallMsg) -> ToolResponse:
        if call.name == "submit_visual_notes":
            try:
                args = VisualNotesArgs.model_validate_json(call.arguments)
            except Exception as e:
                return ToolResponse(f"Invalid: {e}")
            allowed = set(self._keys)
            self._notes = {n.key: n.note for n in args.notes if n.key in allowed}
            return ToolResponse("ok")
        return ToolResponse(f"Unknown tool: {call.name}")

    def on_text(self, text: str | None) -> None:
        pass

    def is_done(self) -> bool:
        return self._notes is not None

    def retry_prompt(self) -> str | None:
        if self._notes is None:
            return "You must call submit_visual_notes with your observations."
        return None

    def into_output(self) -> dict[str, str]:
        return self._notes or {}


async def look_at(
    session: Session, *, pages: list[int], keys: list[str],
    query: str, source_by_key: dict[str, str], turn: int,
) -> dict[str, str]:
    """Run LookAt agent and return keyed visual notes."""
    from typoon.llm.agent import run as agent_run
    agent = LookAtAgent(
        session, pages=pages, keys=keys,
        query=query, source_by_key=source_by_key,
    )
    result = await agent_run(session.context_provider, agent, hook=session.hook)
    return result.output or {}
