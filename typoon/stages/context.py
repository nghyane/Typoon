"""Chapter context analysis — build ChapterBrief before translation."""

from __future__ import annotations

import re
from pydantic import BaseModel, Field

from typoon.adapters.ctx import TranslateCtx
from typoon.stages.brief import ChapterBrief, AddressRule, chapter_text
from typoon.stages.skills import LOAD_SKILL_TOOL, SkillLibrary
from typoon.stages.tools.brief import ChapterBriefArgs, make_submit_chapter_brief
from typoon.stages.tools.look_at import make_look_at
from typoon.stages.tools.search_knowledge import make_search_knowledge
from typoon.stages.image import encode_page_jpeg
from typoon.domain.prepared import Chapter as PreparedChapter
from typoon.domain.scan import BubbleKey
from typoon.llm.ir import ContentPart, Message, ToolResponse
from typoon.llm.loop import tool_loop
from typoon.llm.tool import Tool
from . import prompt

_PLACEHOLDER_NAMES  = {"", "?", "unknown", "unclear", "someone", "speaker", "listener"}
_UNCERTAIN_WORDS    = ("likely", "maybe", "unclear", "unknown", "uncertain", "probably", "possibly")
_ADDRESS_SENSITIVE_RE = re.compile(
    r"\b(my girl|my boy|my son|my daughter|that's my girl|that's my boy)\b",
    re.IGNORECASE,
)


class _LoadSkillArgs(BaseModel):
    name: str = Field(description="Skill name exactly as listed in the available skill catalog")


async def build_chapter_brief(
    ctx: TranslateCtx,
    prepared: PreparedChapter,
    keyed: list[BubbleKey],
) -> ChapterBrief:
    sensitive        = _address_sensitive(keyed)
    context_snapshot = await _context_snapshot(ctx, sensitive)

    skills = SkillLibrary(ctx.target_lang)
    system = prompt.CONTEXT_SYSTEM.format(
        source_lang=ctx.source_lang,
        target_lang=ctx.target_lang,
        source_policy=prompt.load_source_policy(ctx.source_lang),
        target_policy=prompt.load_target_policy(ctx.target_lang),
        skill_catalog=skills.catalog(),
    )
    user     = prompt.CONTEXT_USER.format(
        context_snapshot=context_snapshot,
        chapter_text=chapter_text(keyed),
    )
    messages = [Message.system(system), _context_user_message(user, prepared, sensitive, keyed)]

    brief: ChapterBrief | None = None

    async def on_submit(args: ChapterBriefArgs) -> ToolResponse:
        nonlocal brief
        address, address_error = _clean_address(args.address)
        if address_error:
            return ToolResponse(address_error)
        brief = ChapterBrief(
            summary=args.summary,
            facts=args.facts,
            glossary={g.source: g.target for g in args.glossary},
            address=address,
            style_notes=args.style_notes,
            page_notes={pn.page: pn.note for pn in args.page_notes},
            key_notes={bn.key: bn.note for bn in args.bubble_notes},
        )
        return ToolResponse("ok")

    tools = [
        make_search_knowledge(ctx),
        Tool(LOAD_SKILL_TOOL, _LoadSkillArgs, lambda args: _handle_load_skill(args, skills)),
        make_look_at(ctx, prepared, keyed),
        make_submit_chapter_brief(on_submit),
    ]

    await tool_loop(
        ctx.context_provider,
        messages,
        tools,
        is_done=lambda: brief is not None,
        agent="context",
        max_turns=12,
        hook=ctx.hook,
    )

    if brief is None:
        raise RuntimeError("Context agent did not submit chapter brief")
    return brief


async def _handle_load_skill(args: _LoadSkillArgs, skills: SkillLibrary) -> ToolResponse:
    return ToolResponse(skills.load(args.name))


async def _context_snapshot(ctx: TranslateCtx, sensitive: list[BubbleKey]) -> str:
    store  = ctx.store
    parts: list[str] = []

    glossary = await store.get_glossary(ctx.project_id)
    if glossary:
        parts.append(
            f"## Glossary\n{len(glossary)} terms available. "
            "Call search_knowledge(scope=glossary, queries=[<term>]) to look up."
        )
    else:
        parts.append("## Glossary\nEmpty.")

    recent = await store.get_recent_chapter_briefs(
        ctx.project_id, before_chapter_idx=ctx.chapter_idx, limit=10
    )
    if recent:
        lines = [
            f"  Ch{r.get('chapter','?')}: {(r.get('brief') or {}).get('summary','')[:120]}"
            for r in recent
        ]
        parts.append(
            "## Prior chapters (briefs available)\n" + "\n".join(lines) +
            "\nCall search_knowledge(scope=briefs, queries=[<topic>]) for details."
        )
    else:
        parts.append("## Prior chapters\nNone.")

    if sensitive:
        lines = [f'  p{bk.page_index} #{bk.key}: "{bk.source_text}"' for bk in sensitive[:20]]
        parts.append(
            "## Address-sensitive bubbles\n"
            "These affect Vietnamese xưng hô/family pronouns. Storyboard images around the "
            "relevant pages are attached below; inspect local story context before deciding "
            "speaker/listener. If still uncertain from the storyboard, write bubble_notes as "
            "'Uncertain speaker; use neutral phrasing' rather than guessing.\n"
            + "\n".join(lines)
        )

    return "\n\n".join(parts)


def _address_sensitive(keyed: list[BubbleKey]) -> list[BubbleKey]:
    return [bk for bk in keyed if _ADDRESS_SENSITIVE_RE.search(bk.source_text)]


def _context_user_message(
    text: str,
    prepared: PreparedChapter,
    sensitive: list[BubbleKey],
    keyed: list[BubbleKey],
) -> Message:
    if not sensitive:
        return Message.user_text(text)

    parts: list[ContentPart] = [ContentPart.of_text(text)]
    ordered = sorted(keyed, key=lambda bk: (bk.page_index, bk.idx))
    for target in sensitive[:2]:
        storyboard = _storyboard_image(prepared, ordered, target)
        if storyboard is None:
            continue
        parts.append(ContentPart.of_text(
            f"--- Address-sensitive dialogue-neighborhood storyboard: "
            f"inspect speaker/listener for #{target.key} ---"
        ))
        parts.append(ContentPart.of_image(encode_page_jpeg(storyboard)))
    return Message.user_parts(parts)


def _storyboard_image(
    prepared: PreparedChapter,
    ordered: list[BubbleKey],
    target: BubbleKey,
):
    import cv2
    import numpy as np

    try:
        pos = next(i for i, bk in enumerate(ordered) if bk.key == target.key)
    except StopIteration:
        return None

    window = ordered[max(0, pos - 3): min(len(ordered), pos + 4)]
    by_page: dict[int, list[BubbleKey]] = {}
    for bk in window:
        by_page.setdefault(bk.page_index, []).append(bk)

    pages = sorted(by_page, key=lambda p: (abs(p - target.page_index), p))[:6]
    if not pages:
        return None

    PANEL_W  = 280
    PADDING  = 8
    GAP      = 4
    TOP_BAR_H = 30

    panel_h   = 0
    page_info = []
    for page_index in pages:
        bgr = cv2.imread(str(prepared.page_path(page_index)))
        if bgr is None:
            continue
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        bks   = by_page[page_index]
        h, w  = image.shape[:2]
        scale = PANEL_W / w
        ph    = int(h * scale)
        panel_h = max(panel_h, ph)
        page_info.append((image, bks, scale, ph))

    cols     = len(page_info)
    CANVAS_W = PADDING + cols * PANEL_W + (cols - 1) * GAP + PADDING
    CANVAS_H = PADDING + TOP_BAR_H + panel_h + PADDING
    canvas   = np.full((CANVAS_H, CANVAS_W, 3), 250, dtype=np.uint8)

    cv2.putText(
        canvas,
        f"Target #{target.key} | p{target.page_index} | {target.source_text}  |  "
        "Red=target Blue=nearby  |  Identify speaker/listener for Vietnamese pronouns",
        (PADDING, PADDING + 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 2, cv2.LINE_AA,
    )

    for i, (image, bks, scale, ph) in enumerate(page_info):
        pw      = int(image.shape[1] * scale)
        top_pad = (panel_h - ph) // 2
        x = PADDING + i * (PANEL_W + GAP)
        y = PADDING + TOP_BAR_H

        canvas[y + top_pad: y + top_pad + ph, x: x + pw] = cv2.resize(image, (pw, ph))
        cv2.putText(
            canvas, f"p{pages[i]}",
            (x + 3, y - 3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1, cv2.LINE_AA,
        )
        for bk in bks:
            poly  = [[px * scale + x, py * scale + y + top_pad] for px, py in bk.box.polygon]
            pts   = np.array(poly, dtype=np.int32)
            color = (230, 40, 40) if bk.key == target.key else (40, 110, 220)
            cv2.polylines(canvas, [pts], True, color, 3 if bk.key == target.key else 2)

    return canvas


def _clean_address(address) -> tuple[list[AddressRule], str]:
    seen:   dict[tuple[str, str], AddressRule] = {}
    errors: list[str] = []

    for a in address:
        speaker   = _clean_text(a.speaker)
        listener  = _clean_text(a.listener)
        self_ref  = _clean_text(a.self_ref)
        other_ref = _clean_text(a.other_ref)
        note      = _clean_text(a.note)
        pair      = (_norm_name(speaker), _norm_name(listener))

        if _is_placeholder(speaker):
            errors.append(f"placeholder speaker is not allowed: {a.speaker!r}")
            continue
        if listener != "*" and _is_placeholder(listener):
            errors.append(f"placeholder listener is not allowed: {a.listener!r}")
            continue
        if not self_ref or not other_ref:
            errors.append(f"empty xưng hô is not allowed for {speaker} -> {listener}")
            continue
        uncertain_text = f"{speaker} {listener} {note}".casefold()
        if any(word in uncertain_text for word in _UNCERTAIN_WORDS):
            errors.append(
                f"uncertain address rule is not allowed for {speaker} -> {listener}; "
                "put uncertainty in bubble_notes"
            )
            continue
        if speaker and listener and pair[0] == pair[1]:
            errors.append(f"self-address rule is not allowed: {speaker} -> {listener}")
            continue

        rule = AddressRule(speaker=speaker, listener=listener,
                           self_ref=self_ref, other_ref=other_ref, note=note)
        old  = seen.get(pair)
        if old is not None and (old.self_ref, old.other_ref) != (self_ref, other_ref):
            errors.append(
                f"conflicting address rules for {speaker} -> {listener}: "
                f"{old.self_ref}/{old.other_ref} vs {self_ref}/{other_ref}"
            )
        else:
            seen[pair] = rule

    if errors:
        return [], (
            "Address rules are invalid. Submit chapter_brief again with exactly one "
            "non-conflicting rule per speaker->listener pair. Put uncertain speaker/listener "
            "details in bubble_notes, not address. Errors:\n- " + "\n- ".join(errors)
        )
    return list(seen.values()), ""


def _clean_text(text: str) -> str:
    return " ".join(text.strip().split())


def _norm_name(name: str) -> str:
    return _clean_text(name).casefold()


def _is_placeholder(name: str) -> bool:
    return _norm_name(name) in _PLACEHOLDER_NAMES
