"""Chapter context analysis — build ChapterBrief before translation."""

from __future__ import annotations

from pydantic import BaseModel, Field

from typoon.adapters.ctx import TranslateCtx
from typoon.adapters.prepared_reader import PreparedReader
from typoon.stages.brief import ChapterBrief, AddressRule, chapter_text
from typoon.stages.skills import LOAD_SKILL_TOOL, SkillLibrary
from typoon.stages.tools.brief import ChapterBriefArgs, make_submit_chapter_brief
from typoon.stages.tools.look_at import make_look_at
from typoon.stages.tools.mark_noise import MarkNoiseArgs, make_mark_noise
from typoon.stages.tools.mark_noise_page import MarkNoisePageArgs, make_mark_noise_page
from typoon.stages.tools.search_knowledge import make_search_knowledge
from typoon.stages.page import _is_auto_skip
from typoon.domain.prepared import Chapter as PreparedChapter
from typoon.domain.scan import BubbleKey
from typoon.llm.ir import Message, ToolResponse
from typoon.llm.loop import tool_loop
from typoon.llm.tool import Tool
from . import prompt

_PLACEHOLDER_NAMES  = {"", "?", "unknown", "unclear", "someone", "speaker", "listener"}
_UNCERTAIN_WORDS    = ("likely", "maybe", "unclear", "unknown", "uncertain", "probably", "possibly")
# Targets whose grammar requires explicit speaker/listener decisions for
# almost every line — pronouns, honorifics, kinship terms, register
# levels. For these, mounting look_at by default is correct: the agent
# can opt out (one tool def in the system prompt is cheap), but losing
# the option silently degrades every chapter.
_HIGH_CONTEXT_TARGETS = {"vi", "ja", "ko", "zh", "th"}


class _LoadSkillArgs(BaseModel):
    name: str = Field(description="Skill name exactly as listed in the available skill catalog")


async def build_chapter_brief(
    ctx: TranslateCtx,
    prepared: PreparedChapter,
    reader: PreparedReader,
    keyed: list[BubbleKey],
) -> ChapterBrief:
    # Pre-fold deterministic noise: bubbles that match _is_auto_skip
    # (whitelist regex, single digits, OCR rubble) never reach the agent.
    # The translator already drops them; surfacing them here only burns
    # tokens and tempts the agent to "explain" each watermark string.
    pre_noise = {bk.key for bk in keyed if _is_auto_skip(bk.source_text)}
    visible_keyed = [bk for bk in keyed if bk.key not in pre_noise]

    # Per-user translator memory drives the long-lived context: per-
    # material character/world/style/glossary cards plus a sliding
    # window of briefs from prior chapters this user already translated.
    # Auto-create on first touch so subsequent appends have a row to
    # hang off; the cards stay empty until the agent / user populates
    # them.
    memory = await ctx.store.get_translator_memory(
        user_id=ctx.owner_id,
        material_id=ctx.material_id,
        target_lang=ctx.target_lang,
    )
    if memory is None:
        memory = await ctx.store.upsert_translator_memory(
            user_id=ctx.owner_id,
            material_id=ctx.material_id,
            source_lang=ctx.source_lang,
            target_lang=ctx.target_lang,
        )

    # Glossary: user_glossary (lang-pair scoped) merged with memory's
    # per-material glossary cards. Memory takes precedence — it is the
    # narrowest scope and the one the agent actively learns into.
    user_terms = await ctx.store.list_user_glossary(
        ctx.owner_id,
        source_lang=ctx.source_lang,
        target_lang=ctx.target_lang,
    )
    glossary_terms: dict[str, str] = {
        r["source_term"]: r["target_term"] for r in user_terms
    }
    for term in memory.get("glossary", []) or []:
        src = term.get("source_term")
        tgt = term.get("target_term")
        if src and tgt:
            glossary_terms[src] = tgt

    # Sliding window of prior briefs for this (user, material, lang).
    # Replaces the slice-1 placeholder `prior_briefs = []`. Window of 5
    # chapters keeps the prompt bounded; the agent can pull older
    # entries on demand via search_knowledge once that's wired to
    # memory briefs (TODO once the M2 storage move settles).
    prior_brief_rows = await ctx.store.list_recent_memory_briefs(
        memory_id=memory["id"],
        before_chapter_id=ctx.chapter_id,
        limit=5,
    )
    prior_briefs: list[dict] = [
        {
            "chapter": r.get("number"),
            "brief":   r.get("brief_json") or {},
            "summary": r.get("summary"),
        }
        for r in prior_brief_rows
    ]
    has_knowledge = bool(glossary_terms) or bool(prior_briefs)
    context_snapshot = _context_snapshot(
        glossary_terms=glossary_terms,
        prior_briefs=prior_briefs,
    )

    skills = SkillLibrary(ctx.target_lang)
    system = prompt.CONTEXT_SYSTEM.format(
        source_lang=ctx.source_lang,
        target_lang=ctx.target_lang,
        source_policy=prompt.load_source_policy(ctx.source_lang),
        target_policy=prompt.load_target_policy(ctx.target_lang),
        skill_catalog=skills.catalog(),
    )
    user = prompt.CONTEXT_USER.format(
        context_snapshot=context_snapshot,
        chapter_text=chapter_text(visible_keyed) or "(all bubbles already pre-filtered as deterministic noise — submit an empty brief)",
    )
    messages = [Message.system(system), Message.user_text(user)]

    brief: ChapterBrief | None = None
    valid_keys: set[str] = {bk.key for bk in visible_keyed}
    noise_keys: set[str] = set(pre_noise)
    noise_pages: set[int] = set()
    valid_pages: set[int] = {bk.page_index for bk in keyed}
    keys_by_page: dict[int, list[str]] = {}
    for bk in keyed:
        keys_by_page.setdefault(bk.page_index, []).append(bk.key)

    async def on_mark_noise(args: MarkNoiseArgs) -> ToolResponse:
        unknown = [k for k in args.keys if k not in valid_keys]
        if unknown:
            return ToolResponse(
                f"Unknown bubble keys (not in chapter): {unknown[:8]}. "
                "Copy keys exactly from after '#' in the chapter text. "
                "Note: deterministic-noise bubbles are NOT shown — do not invent keys."
            )
        # Cap is computed against the visible (non-pre-folded) chapter.
        # Pre-folded keys don't count — they're a deterministic baseline
        # the agent can't undo or amplify.
        agent_added = (noise_keys - pre_noise) | set(args.keys)
        if len(agent_added) > max(1, int(len(valid_keys) * 0.6)):
            return ToolResponse(
                "Refusing to mark this many bubbles as noise (>60% of visible chapter). "
                "Only flag platform chrome / watermarks / counters; everything "
                "else must be translated."
            )
        noise_keys.update(args.keys)
        return ToolResponse(f"Marked {len(args.keys)} as noise. Total noise so far: {len(noise_keys)}.")

    async def on_mark_noise_page(args: MarkNoisePageArgs) -> ToolResponse:
        unknown = [p for p in args.pages if p not in valid_pages]
        if unknown:
            return ToolResponse(
                f"Unknown page indices (not in chapter): {unknown[:8]}. "
                f"Valid pages are 0..{max(valid_pages) if valid_pages else 0}."
            )
        # Cap at 50% of pages so a misfire can't drop most of the chapter.
        cap = max(1, int(len(valid_pages) * 0.5))
        if len(noise_pages | set(args.pages)) > cap:
            return ToolResponse(
                "Refusing to drop this many pages (>50% of chapter). "
                "Only drop pages where EVERY bubble is platform/scanlator "
                "chrome with no story content."
            )
        noise_pages.update(args.pages)
        # Pages dropped at render time also bypass the translator: every
        # bubble on those pages is implicitly noise.
        added = 0
        for p in args.pages:
            for k in keys_by_page.get(p, []):
                if k not in noise_keys:
                    noise_keys.add(k)
                    added += 1
        return ToolResponse(
            f"Marked {len(args.pages)} pages as full-page noise "
            f"(+{added} bubble keys auto-added). Total noise pages: {len(noise_pages)}."
        )

    async def on_submit(args: ChapterBriefArgs) -> ToolResponse:
        nonlocal brief
        address, address_error = _clean_address(args.address)
        if address_error:
            return ToolResponse(address_error)
        brief = ChapterBrief(
            summary=args.summary,
            glossary={g.source: g.target for g in args.glossary},
            address=address,
            style_notes=args.style_notes,
            page_notes={pn.page: pn.note for pn in args.page_notes},
            key_notes={bn.key: bn.note for bn in args.bubble_notes},
            noise_keys=noise_keys,
            noise_pages=noise_pages,
        )
        return ToolResponse("ok")

    tools = [
        Tool(LOAD_SKILL_TOOL, _LoadSkillArgs, lambda args: _handle_load_skill(args, skills)),
        make_mark_noise(on_mark_noise),
        make_mark_noise_page(on_mark_noise_page),
        make_submit_chapter_brief(on_submit),
    ]
    # search_knowledge is dead weight when the project has no glossary
    # and no prior briefs. Skip mounting it — fewer tool defs in the
    # system prompt, and the agent can't be tempted into a no-op call.
    if has_knowledge:
        tools.insert(0, make_search_knowledge(ctx))
    # look_at is mounted by target language. Targets that require explicit
    # speaker/listener decisions for almost every line (VI, JA, KO, ZH,
    # TH) get the tool by default; the agent decides per chapter whether
    # to actually call it. Targets where text alone usually carries
    # enough context (EN, ES, FR, …) do not get it — saves a tool def.
    # The wrapper enforces a per-chapter call cap so a misbehaving agent
    # cannot chain "let me verify" round trips.
    if ctx.target_lang.lower() in _HIGH_CONTEXT_TARGETS:
        tools.append(make_look_at(ctx, prepared, reader, visible_keyed))

    await tool_loop(
        ctx.context_provider,
        messages,
        tools,
        is_done=lambda: brief is not None,
        agent="context",
        # Generous cap — context output is bounded by ~16k tokens, but
        # tool ping-pong (search_knowledge → load_skill → mark_noise →
        # submit) plus retries on validation errors can chain. We'd
        # rather burn turns than fail the chapter.
        max_turns=16,
        hook=ctx.hook,
    )

    if brief is None:
        raise RuntimeError("Context agent did not submit chapter brief")
    return brief


async def _handle_load_skill(args: _LoadSkillArgs, skills: SkillLibrary) -> ToolResponse:
    return ToolResponse(skills.load(args.name))


def _context_snapshot(
    *,
    glossary_terms: list,
    prior_briefs: list,
) -> str:
    """Emit only sections that carry information.

    Empty glossary / no prior briefs → omit. The agent doesn't need to
    be told it has nothing to look up. Tools for lookup are mounted
    conditionally in build_chapter_brief.
    """
    parts: list[str] = []

    if glossary_terms:
        parts.append(
            f"## Glossary\n{len(glossary_terms)} terms available. "
            "Call search_knowledge(scope=glossary, queries=[<term>]) to look up."
        )

    if prior_briefs:
        lines = [
            f"  Ch{r.get('chapter','?')}: {(r.get('brief') or {}).get('summary','')[:120]}"
            for r in prior_briefs
        ]
        parts.append(
            "## Prior chapters (briefs available)\n" + "\n".join(lines) +
            "\nCall search_knowledge(scope=briefs, queries=[<topic>]) for details."
        )

    return "\n\n".join(parts) if parts else "(no prior context — translate from chapter text alone)"


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
