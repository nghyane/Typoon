"""Knowledge agent — extracts series knowledge after each chapter translation."""

from __future__ import annotations

from typoon.llm.ir import Message, ToolCallMsg, ToolDef, ToolResponse
from typoon.llm.agent import run as agent_run, RunResult
from typoon.domain.bubble import Session

from .tools.update_snapshot import UpdateSnapshotArgs, update_snapshot
from .tools.add_note import AddNoteArgs, add_note
from .tools.upsert_glossary import UpsertGlossaryArgs, upsert_glossary

SYSTEM = """\
You extract and consolidate series knowledge from translated manga dialogue.

Update the snapshot and record notable facts.
- Snapshot must be self-contained — a new reader should understand the series state.
- Keep concise. Drop old events (>5 chapters ago) unless plot-critical.
- Only add glossary entries for recurring proper nouns.
- Call all tools in one message if possible."""


async def consolidate(session: Session, chapter: int, pairs: list[tuple[str, str]]) -> RunResult:
    agent = _Agent(session, chapter, pairs)
    return await agent_run(session.context_provider, agent, hook=session.hook)


class _Agent:
    def __init__(self, session: Session, chapter: int, pairs: list[tuple[str, str]]) -> None:
        self._session = session
        self._chapter = chapter
        self._done = False

        dialogue = "\n".join(f'"{src}" → "{tgt}"' for src, tgt in pairs if tgt)
        user_parts = [f"Chapter {chapter} translations ({session.source_lang} → {session.target_lang}):\n{dialogue}"]
        if session.knowledge:
            user_parts.append(f"\nPrevious knowledge snapshot:\n{session.knowledge}")
        self._user_text = "\n".join(user_parts)

    def name(self) -> str:
        return "knowledge"

    def system_prompt(self) -> str:
        return SYSTEM

    def user_message(self) -> Message:
        return Message.user_text(self._user_text)

    def tools(self) -> list[ToolDef]:
        return [update_snapshot.definition, add_note.definition, upsert_glossary.definition]

    async def dispatch(self, call: ToolCallMsg) -> ToolResponse:
        s = self._session
        try:
            match call.name:
                case "update_snapshot":
                    args = UpdateSnapshotArgs.model_validate_json(call.arguments)
                    await s.store.save_knowledge(s.project_id, self._chapter, args.snapshot)
                    return ToolResponse("ok")
                case "add_note":
                    args = AddNoteArgs.model_validate_json(call.arguments)
                    await s.store.add_note(s.project_id, self._chapter, args.note_type, args.content)
                    return ToolResponse("ok")
                case "upsert_glossary":
                    args = UpsertGlossaryArgs.model_validate_json(call.arguments)
                    await s.store.glossary_upsert(s.project_id, args.source_term, args.target_term, args.notes or None)
                    return ToolResponse("ok")
                case _:
                    return ToolResponse(f"Unknown tool: {call.name}")
        except Exception as e:
            return ToolResponse(f"Error: {e}")

    def on_text(self, text: str | None) -> None:
        self._done = True

    def is_done(self) -> bool:
        return self._done

    def retry_prompt(self) -> str | None:
        return None

    def into_output(self) -> None:
        return None
