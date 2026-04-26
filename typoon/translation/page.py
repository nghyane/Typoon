"""Page/window translation agent — implements Agent protocol."""

from __future__ import annotations

from typoon.domain.bubble import Bubble, Session
from typoon.llm.ir import Message, ToolCallMsg, ToolDef, ToolResponse

from . import prompt
from .brief import ChapterBrief, brief_slice
from .protocol import TranslationOp, validate_ops
from .tools.submit import SubmitArgs, submit_translations


class PageAgent:
    """Translates a page-window of keyed bubbles."""

    def __init__(
        self, session: Session, *, brief: ChapterBrief,
        bubbles: list[Bubble], key_map: dict[str, Bubble],
    ) -> None:
        self._session = session
        self._brief = brief
        self._bubbles = bubbles
        self._key_map = key_map
        self._keys = [b.translation_key or "" for b in bubbles]
        self._active = set(self._keys)
        self._accepted: list[TranslationOp] = []
        self._pending_feedback: str = ""

    def name(self) -> str:
        return "translate/page"

    def system_prompt(self) -> str:
        return prompt.PAGE_SYSTEM.format(
            source_lang=self._session.source_lang,
            target_lang=self._session.target_lang,
            source_policy=prompt.load_policy(f"source_{self._session.source_lang}.md"),
            target_policy=prompt.load_policy(f"target_{self._session.target_lang}.md"),
        )

    def user_message(self) -> Message:
        page_indices = {b.page_index for b in self._bubbles}
        return Message.user_text(prompt.PAGE_USER.format(
            brief_slice=brief_slice(self._brief, page_indices, self._keys),
            feedback_block="",
            keys="\n".join(f"#{b.translation_key} {b.source_text}" for b in self._bubbles),
        ))

    def tools(self) -> list[ToolDef]:
        return [submit_translations.definition]

    async def dispatch(self, call: ToolCallMsg) -> ToolResponse:
        if call.name != "submit_translations":
            return ToolResponse(f"Unknown tool: {call.name}")
        try:
            args = SubmitArgs.model_validate_json(call.arguments)
        except Exception as e:
            return ToolResponse(f"Invalid: {e}")
        ops = [TranslationOp(key=it.key, status=it.status.value, text=it.text) for it in args.items]
        result = validate_ops(ops, active=self._active, key_map=self._key_map)
        for op in result.accepted:
            self._accepted.append(op)
            self._active.discard(op.key)
        if not result.invalid and not self._active:
            return ToolResponse("ok")
        parts = [f"#{k}: {r}" for k, r in result.invalid.items()]
        missing = self._active - set(result.invalid)
        if missing:
            parts.append(f"Missing keys: {', '.join(sorted(missing))}")
        self._pending_feedback = "\n".join(parts)
        return ToolResponse(f"Validation errors:\n{self._pending_feedback}")

    def on_text(self, text: str | None) -> None:
        pass

    def is_done(self) -> bool:
        return not self._active

    def retry_prompt(self) -> str | None:
        if self._active:
            return f"Fix these keys and call submit_translations again:\n{self._pending_feedback}"
        return None

    def into_output(self) -> list[TranslationOp]:
        return self._accepted


async def translate_window(
    session: Session, *, brief: ChapterBrief,
    bubbles: list[Bubble], key_map: dict[str, Bubble],
) -> list[TranslationOp]:
    """Run PageAgent and return accepted translations."""
    from typoon.llm.agent import run as agent_run
    agent = PageAgent(session, brief=brief, bubbles=bubbles, key_map=key_map)
    result = await agent_run(session.provider, agent, hook=session.hook)
    if result.error:
        raise result.error
    if agent._active:
        raise RuntimeError(f"Page agent did not resolve all keys. Missing: {', '.join(sorted(agent._active))}")
    return result.output or []
