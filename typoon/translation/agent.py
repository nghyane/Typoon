"""Translation agent — fills bubble.translated_text via LLM tool-calling loop."""

from __future__ import annotations

from typoon.llm.ir import ContentPart, Message, ToolCallMsg, ToolDef, ToolResponse
from typoon.llm.agent import run as agent_run
from typoon.domain.bubble import Bubble, Page, Session

from . import prompt
from .tools import build_tools
from .tools.translate import TranslateArgs
from .tools.view_page import ViewPageArgs, encode_page_jpeg
from .tools.view_bubble import ViewBubbleArgs, encode_bubble_jpeg
from .tools.search_glossary import SearchGlossaryArgs
from .tools.get_context import GetContextArgs

async def translate_pages(pages: list[Page], session: Session) -> tuple[int, Exception | None]:
    """Translate all bubbles in-place. Returns (turns, error_or_none)."""
    all_bubbles = [b for p in pages for b in p.bubbles]
    if not all_bubbles:
        return 0, None

    agent = _Agent(all_bubbles, session)
    result = await agent_run(session.provider, agent, hook=session.hook)

    for b in all_bubbles:
        if b.id in result.output:
            b.translated_text = result.output[b.id]

    return result.turns, result.error


# ── Prompt ───────────────────────────────────────────────────────────


def _build_system(s: Session) -> str:
    return prompt.SYSTEM.format(
        source_lang=s.source_lang,
        target_lang=s.target_lang,
        source_policy=prompt.load_policy(f"source_{s.source_lang}.md"),
        target_policy=prompt.load_policy(f"target_{s.target_lang}.md"),
    )


def _build_user(bubbles: list[Bubble], s: Session) -> str:
    glossary_block = ""
    if s.glossary:
        lines = [f"  {src} → {tgt}" for src, tgt in s.glossary.items()]
        glossary_block = "Glossary (canon — use these, do not search_glossary for them):\n" + "\n".join(lines) + "\n"

    knowledge_block = ""
    if s.knowledge:
        knowledge_block = f"Series knowledge (use to infer pronouns and relationships):\n{s.knowledge}\n"

    return prompt.USER.format(
        source_lang=s.source_lang,
        target_lang=s.target_lang,
        count=len(bubbles),
        ids=", ".join(b.id for b in bubbles),
        glossary_block=glossary_block,
        knowledge_block=knowledge_block,
        bubble_list="\n".join(_format_bubble(b) for b in bubbles),
    )


_LOW_CONF_THRESHOLD = 0.6


def _format_bubble(b: Bubble) -> str:
    if b.ocr_confidence < _LOW_CONF_THRESHOLD:
        return f'[{b.id}] "{b.source_text}" ⚠️ OCR unreliable (conf={b.ocr_confidence:.0%}) — use view_page to read actual text'
    return f'[{b.id}] "{b.source_text}"'


# ── Agent ────────────────────────────────────────────────────────────


class _Agent:
    def __init__(self, bubbles: list[Bubble], session: Session) -> None:
        self._session = session
        self._translated: dict[str, str] = {}
        self._source = {b.id: b.source_text for b in bubbles}
        self._bubbles = {b.id: b for b in bubbles}
        self._total = len(self._source)

        has_images = hasattr(session.source, 'load_page')
        self._system = _build_system(session)
        user_text = _build_user(bubbles, session)

        # Single page → attach image inline (load on demand, release after encode)
        if has_images and session.source.page_count() == 1 and all(b.page_index == 0 for b in bubbles):
            img = session.source.load_page(0)
            self._user_msg = Message.user_parts([
                ContentPart.of_text(user_text),
                ContentPart.of_image(encode_page_jpeg(img)),
            ])
        else:
            self._user_msg = Message.user_text(user_text)

        self._tools = build_tools(
            has_images=has_images,
            has_glossary=bool(session.glossary),
            has_context=session.store is not None,
        )

    def name(self) -> str:
        return "translate"

    def system_prompt(self) -> str:
        return self._system

    def user_message(self) -> Message:
        return self._user_msg

    def tools(self) -> list[ToolDef]:
        return self._tools

    async def dispatch(self, call: ToolCallMsg) -> ToolResponse:
        s = self._session
        try:
            match call.name:
                case "translate":
                    args = TranslateArgs.model_validate_json(call.arguments)
                    for item in args.translations:
                        self._translated[item.id] = item.translated_text
                    missing = sorted(set(self._source) - set(self._translated))
                    if missing:
                        ids = ", ".join(missing[:20])
                        return ToolResponse(f"ok ({len(args.translations)} bubbles). Still missing {len(missing)}: {ids}")
                    return ToolResponse(f"ok ({len(args.translations)} bubbles). All done.")

                case "view_page":
                    args = ViewPageArgs.model_validate_json(call.arguments)
                    if args.page_index >= s.source.page_count():
                        return ToolResponse(f"Error: page {args.page_index} out of range")
                    img = s.source.load_page(args.page_index)
                    return ToolResponse(f"Page {args.page_index}:", image_data_uri=encode_page_jpeg(img))

                case "view_bubble":
                    args = ViewBubbleArgs.model_validate_json(call.arguments)
                    b = self._bubbles.get(args.bubble_id)
                    if b is None:
                        return ToolResponse(f"Error: bubble {args.bubble_id} not found")
                    img = s.source.load_page(b.page_index)
                    uri = encode_bubble_jpeg(img, b.polygon)
                    return ToolResponse(f"Bubble {b.id}:", image_data_uri=uri)

                case "search_glossary":
                    args = SearchGlossaryArgs.model_validate_json(call.arguments)
                    hits = await s.store.glossary_search(s.project_id, args.query)
                    if not hits:
                        return ToolResponse("No matching glossary entries.")
                    return ToolResponse("\n".join(f"  {h['source_term']} → {h['target_term']}" for h in hits))

                case "get_context":
                    args = GetContextArgs.model_validate_json(call.arguments)
                    from .context import ask
                    answer = await ask(s.context_provider, s.store, s.project_id, args.question)
                    return ToolResponse(answer or "No relevant context found.")

                case _:
                    return ToolResponse(f"Unknown tool: {call.name}")
        except Exception as e:
            return ToolResponse(f"Error: {e}")

    def on_text(self, text: str | None) -> None:
        pass

    def is_done(self) -> bool:
        return len(self._translated) >= self._total

    def retry_prompt(self) -> str | None:
        return None  # tool response already guides LLM to continue

    def into_output(self) -> dict[str, str]:
        return dict(self._translated)
