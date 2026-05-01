"""Page/window translation agent — implements Agent protocol."""

from __future__ import annotations

from dataclasses import dataclass

from typoon.adapters.session import Session
from typoon.domain.scan import Bubble as ScannedBubble
from typoon.llm.ir import Message, ToolCallMsg, ToolDef, ToolResponse

from . import prompt
from .brief import ChapterBrief, annotated_chapter_text, brief_slice
from .tools.submit import SubmitArgs, TextKind, submit_translations


@dataclass(slots=True)
class TranslationOp:
    key: str
    kind: str   # dialogue | sfx | skip
    text: str = ""


class PageAgent:
    """Translates a page-window of keyed bubbles."""

    def __init__(
        self,
        session: Session,
        *,
        brief: ChapterBrief,
        window_keys: list[str],
        key_map: dict[str, ScannedBubble],
    ) -> None:
        self._session = session
        self._brief = brief
        self._key_map = key_map
        self._keys = window_keys
        self._active = set(window_keys)
        self._accepted: list[TranslationOp] = []
        self._pending_feedback: str = ""
        self._text_retries = 0

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
        page_indices = {self._key_map[k].page_index for k in self._keys if k in self._key_map}
        return Message.user_text(prompt.PAGE_USER.format(
            brief_slice=brief_slice(self._brief, page_indices, self._keys),
            feedback_block="",
            annotated_text=annotated_chapter_text(self._key_map, self._active),
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
        errors: list[str] = []
        for item in args.items:
            key, kind, text = item.key, item.kind, item.text.strip()
            if key not in self._key_map:
                errors.append(f"#{key}: unknown key")
                continue
            if key not in self._active:
                errors.append(f"#{key}: already accepted or not in active set")
                continue
            if kind != TextKind.skip and not text:
                errors.append(f"#{key}: {kind.value} requires translated text")
                continue
            if text and (key in text or f"#{key}" in text):
                errors.append(f"#{key}: translation contains key marker")
                continue
            self._accepted.append(TranslationOp(key=key, kind=kind.value,
                                                 text=text if kind != TextKind.skip else ""))
            self._active.discard(key)
        if self._active:
            errors.append(f"Missing keys: {', '.join(sorted(self._active))}")
            self._pending_feedback = "\n".join(errors)
            return ToolResponse(f"Validation errors:\n{self._pending_feedback}")
        return ToolResponse("ok")

    def on_text(self, text: str | None) -> None:
        self._text_retries += 1

    def is_done(self) -> bool:
        return not self._active

    def retry_prompt(self) -> str | None:
        if not self._active:
            return None
        if self._text_retries >= 2:
            return None
        if self._pending_feedback:
            return f"Fix these keys and call submit_translations again:\n{self._pending_feedback}"
        return "Do not respond with text. Call submit_translations."

    def into_output(self) -> list[TranslationOp]:
        return self._accepted


async def translate_window(
    session: Session,
    *,
    brief: ChapterBrief,
    window_keys: list[str],
    key_map: dict[str, ScannedBubble],
) -> tuple[list[TranslationOp], int]:
    """Run PageAgent. Returns (accepted ops, turns used)."""
    from typoon.llm.agent import run as agent_run
    agent = PageAgent(session, brief=brief, window_keys=window_keys, key_map=key_map)
    result = await agent_run(session.provider, agent, hook=session.hook, max_turns=4)
    if result.error:
        raise result.error
    if agent._active:
        raise RuntimeError(f"Page agent incomplete. Missing: {', '.join(sorted(agent._active))}")
    return result.output or [], result.turns
