"""Two-pass translation: text-only first, images for unclear bubbles second.

No agent loop — translate is a pure function, not agentic exploration.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from typoon.app.events import (
    Hook,
    LLMCall,
    LLMResponse,
    LLMText,
    LLMThinking,
    PipelineError,
    ToolCallStart,
    ToolResult,
)
from typoon.domain.bubble import Bubble, Page, Session
from typoon.llm.ir import (
    CallResponse,
    ContentPart,
    Message,
    Provider,
    StreamEvent,
    StreamEventType,
    ToolCallMsg,
)

from . import prompt
from .tools.submit import SubmitArgs, submit_translations
from .tools.view_bubble import encode_bubble_jpeg
from .tools.view_page import encode_page_jpeg

_LOW_CONF = 0.6
_MAX_PASSES = 3


@dataclass
class _State:
    source: dict[str, Bubble]
    translated: dict[str, str] = field(default_factory=dict)
    unclear: set[str] = field(default_factory=set)
    turns: int = 0
    error: Exception | None = None

    def missing(self) -> list[str]:
        return sorted(set(self.source) - set(self.translated))


async def translate_pages(
    pages: list[Page],
    session: Session,
) -> tuple[int, Exception | None]:
    """Fill bubble.translated_text for all bubbles across pages.

    Returns (total_turns, error_or_none). Partial state is kept on error —
    caller can check how many bubbles got translated.
    """
    bubbles = [b for p in pages for b in p.bubbles]
    if not bubbles:
        return 0, None

    state = _State(source={b.id: b for b in bubbles})

    # ── Pass 1: text only ────────────────────────────────
    try:
        await _call(
            state,
            session,
            user_msg=_build_pass1_user(bubbles, session),
            pass_label="pass1",
        )
    except Exception as e:
        state.error = e
        session.hook.on(PipelineError(stage="translate/pass1", error=e))

    # ── Pass 2: image for unclear ────────────────────────
    if state.unclear and state.error is None:
        try:
            await _call(
                state,
                session,
                user_msg=_build_pass2_user(state, session),
                pass_label="pass2",
            )
        except Exception as e:
            state.error = e
            session.hook.on(PipelineError(stage="translate/pass2", error=e))

    # ── Pass 3: missing (rare) ───────────────────────────
    missing = state.missing()
    if missing and state.error is None:
        try:
            await _call(
                state,
                session,
                user_msg=_build_pass3_user(missing, state, session),
                pass_label="pass3",
            )
        except Exception as e:
            state.error = e
            session.hook.on(PipelineError(stage="translate/pass3", error=e))

    # ── Write back ───────────────────────────────────────
    for b in bubbles:
        if b.id in state.translated:
            b.translated_text = state.translated[b.id]

    return state.turns, state.error


# ── Single LLM call with streaming tool dispatch ────────────────────


async def _call(
    state: _State,
    session: Session,
    user_msg: Message,
    pass_label: str,
) -> None:
    state.turns += 1
    hook = session.hook
    provider = session.provider
    system = _system_prompt(session)
    tools = [submit_translations.definition]
    messages = [Message.system(system), user_msg]

    hook.on(LLMCall(agent=f"translate/{pass_label}", turn=state.turns))
    t0 = time.monotonic()

    if hasattr(provider, "stream") and callable(getattr(provider, "stream", None)):
        resp = await _consume_stream(
            provider, messages, tools, pass_label, state.turns, hook,
        )
    else:
        resp = await provider.call(messages, tools)

    ms = (time.monotonic() - t0) * 1000
    n_tools = len(resp.tool_calls) if resp.tool_calls else 0
    hook.on(LLMResponse(
        agent=f"translate/{pass_label}", turn=state.turns,
        tool_calls=n_tools, ms=ms,
    ))

    if not resp.tool_calls:
        return

    for tc in resp.tool_calls:
        _apply(tc, state)
        hook.on(ToolResult(
            agent=f"translate/{pass_label}", turn=state.turns,
            tool=tc.name, result=f"{len(state.translated)}/{len(state.source)}",
        ))


def _apply(tc: ToolCallMsg, state: _State) -> None:
    """Merge one submit_translations call into state. Tolerates malformed args."""
    if tc.name != "submit_translations":
        return
    try:
        args = SubmitArgs.model_validate_json(tc.arguments)
    except Exception:
        return  # skip corrupt call; missing-ids retry pass will catch it
    for edit in args.edits:
        if edit.id not in state.source:
            continue
        if edit.unclear:
            state.unclear.add(edit.id)
            # don't commit text yet — pass 2 will replace
        else:
            state.translated[edit.id] = edit.text
            state.unclear.discard(edit.id)


# ── Streaming consumer (copy of llm.agent logic, standalone) ────────


async def _consume_stream(
    provider: Provider,
    messages: list[Message],
    tools: list,
    agent_name: str,
    turn: int,
    hook: Hook,
) -> CallResponse:
    text_parts: list[str] = []
    pending: dict[int, list] = {}  # tool_index -> [id, name, arg_chunks]

    async for event in provider.stream(messages, tools):
        match event.type:
            case StreamEventType.THINKING_DELTA:
                hook.on(LLMThinking(agent=agent_name, turn=turn, delta=event.text))
            case StreamEventType.TEXT_DELTA:
                text_parts.append(event.text)
                hook.on(LLMText(agent=agent_name, turn=turn, delta=event.text))
            case StreamEventType.TOOL_CALL_START:
                pending[event.tool_index] = [event.tool_id, event.tool_name, []]
                hook.on(ToolCallStart(
                    agent=agent_name, turn=turn, tool_name=event.tool_name,
                ))
            case StreamEventType.TOOL_CALL_DELTA:
                if event.tool_index in pending:
                    pending[event.tool_index][2].append(event.text)
            case StreamEventType.DONE:
                break

    tool_calls = [
        ToolCallMsg(id=entry[0], name=entry[1], arguments="".join(entry[2]))
        for _, entry in sorted(pending.items())
    ]
    text = "".join(text_parts) if text_parts else None
    return CallResponse(tool_calls=tool_calls, text=text)


# ── Prompt building ─────────────────────────────────────────────────


def _system_prompt(s: Session) -> str:
    return prompt.SYSTEM.format(
        source_lang=s.source_lang,
        target_lang=s.target_lang,
        source_policy=prompt.load_policy(f"source_{s.source_lang}.md"),
        target_policy=prompt.load_policy(f"target_{s.target_lang}.md"),
    )


def _build_pass1_user(bubbles: list[Bubble], s: Session) -> Message:
    """Full bubble list with OCR text and confidence. No images."""
    glossary_block = _glossary_block(s.glossary)
    knowledge_block = _knowledge_block(s.knowledge)
    bubble_list = "\n".join(_format_bubble(b) for b in bubbles)

    text = prompt.PASS1_USER.format(
        source_lang=s.source_lang,
        target_lang=s.target_lang,
        count=len(bubbles),
        glossary_block=glossary_block,
        knowledge_block=knowledge_block,
        bubble_list=bubble_list,
    )
    return Message.user_text(text)


def _build_pass2_user(state: _State, s: Session) -> Message:
    """Follow-up: resolve unclear bubbles with image crops attached."""
    unclear_ids = sorted(state.unclear)
    # Show everything translated so far (context for disambiguation)
    done_lines = [
        f"[{bid}] \"{state.source[bid].source_text}\" → \"{state.translated[bid]}\""
        for bid in sorted(state.translated)
    ]
    unclear_lines = [
        f"[{bid}] \"{state.source[bid].source_text}\""
        for bid in unclear_ids
    ]

    text = prompt.PASS2_USER.format(
        source_lang=s.source_lang,
        target_lang=s.target_lang,
        done_count=len(state.translated),
        unclear_count=len(unclear_ids),
        done_block="\n".join(done_lines) if done_lines else "(none yet)",
        unclear_block="\n".join(unclear_lines),
    )

    parts: list[ContentPart] = [ContentPart.of_text(text)]
    # Attach bubble crops for each unclear id
    source = s.source
    if source is not None and hasattr(source, "load_page"):
        for bid in unclear_ids:
            b = state.source[bid]
            try:
                page_img = source.load_page(b.page_index)
                parts.append(ContentPart.of_text(f"\nImage for [{bid}]:"))
                parts.append(ContentPart.of_image(encode_bubble_jpeg(page_img, b.polygon)))
            except Exception:
                continue  # skip missing pages silently
    return Message.user_parts(parts)


def _build_pass3_user(missing: list[str], state: _State, s: Session) -> Message:
    """Short retry for bubbles that slipped through passes 1+2."""
    missing_lines = [
        f"[{bid}] \"{state.source[bid].source_text}\""
        for bid in missing
    ]
    text = prompt.PASS3_USER.format(
        source_lang=s.source_lang,
        target_lang=s.target_lang,
        missing_count=len(missing),
        missing_block="\n".join(missing_lines),
    )
    return Message.user_text(text)


# ── Formatting ──────────────────────────────────────────────────────


def _format_bubble(b: Bubble) -> str:
    conf = f"{b.ocr_confidence:.0%}"
    warn = " ⚠OCR" if b.ocr_confidence < _LOW_CONF else ""
    return f"[{b.id} conf={conf}{warn}] \"{b.source_text}\""


def _glossary_block(glossary: dict[str, str] | None) -> str:
    if not glossary:
        return ""
    lines = [f"  {src} → {tgt}" for src, tgt in glossary.items()]
    return "Glossary (use these exact translations):\n" + "\n".join(lines) + "\n"


def _knowledge_block(knowledge: str | None) -> str:
    if not knowledge:
        return ""
    return f"Series knowledge:\n{knowledge}\n"
