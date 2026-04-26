# RFC-008: Chapter Brief Artifacts and Parallel Keyed Page Translation

## Status

Draft. Full rebuild of translation knowledge/storage architecture. No backward
compatibility with the old knowledge snapshot flow is required.

## Problem

The current translation flow uses fixed passes:

```text
pass 1: translate text in chunks
pass 2: resolve unclear bubbles with images
pass 3: retry missing bubbles
```

This fails in production with weaker models:

- line-order output shifts when one line is omitted,
- sequential IDs invite renumbering or hallucinated IDs,
- tool calls are not reliable across all providers,
- each retry may resend too much context,
- page-by-page translation without a shared brief causes inconsistent terms,
  xưng hô, tone, and speaker assumptions.

## Decision

Use a context-first workflow with saved brief artifacts:

```text
1. Chapter context stage
   Read the full keyed chapter text once.
   Produce one ChapterBrief artifact for the current chapter.

2. Page translation stage
   Clarify requested visual context with LookAt.
   Fork page/page-window translators in parallel.
   Each worker receives only the relevant brief slice + page text.
```

The goal is to pay the full-chapter context cost once, then avoid looping the
whole context through every page translation call. The ChapterBrief is saved
after successful translation and becomes the knowledge artifact for future
chapters. There is no separate end-of-chapter summary agent.

This RFC intentionally replaces the old knowledge architecture:

```text
remove: knowledge_snapshots, notes, post-translation knowledge agent
add:    chapter_briefs as the source of chapter knowledge
change: translations persist opaque key + status
```

## Pipeline

```text
Key all bubbles
  -> KnowledgeProvider loads relevant previous ChapterBrief artifacts
  -> ChapterContextAgent reads compact full keyed text + selected prior context
  -> emits one ChapterBrief
  -> Controller calls LookAt grouped by page
  -> merge LookAt notes into ChapterBrief
  -> fork PageTranslator workers per page/page-window
  -> validate keyed outputs
  -> repair failed pages/keys only
  -> write translations back by key
  -> save ChapterBrief artifact after successful translation
```

## Core rules

1. Every bubble gets an opaque key, e.g. `#K7Q9M2`.
2. Output mapping is always `key -> translation`; output order is irrelevant.
3. Reading order is heuristic, especially for manga.
4. The controller is the source of truth for completion.
5. `done=true` / `DONE` from a model is only a hint.
6. LookAt returns visual notes, not final translations.
7. Full chapter text is only sent to the context stage, not to page workers.
8. Page workers receive brief slices, not the whole brief if avoidable.
9. Retry only missing/invalid keys or failed pages.
10. ChapterBrief is the only context artifact produced by the model.
11. Previous context is retrieved through a KnowledgeProvider, not built into
    every translation prompt.

Completion:

```python
complete = set(translated) == set(expected)
```

## Components

```text
typoon/translation/
  translate.py        public translate_pages() and orchestration
  keys.py             opaque key generation and key maps
  brief.py            ChapterBrief model and slicing
  context.py          ChapterContextAgent prompt/call/parser
  look_at.py          LookAt prompt/call/parser
  page.py             PageTranslator prompt/call/parser
  protocol.py         keyed ops and validation
  tools/submit.py     keyed submit tool
```

Keep the public API unchanged:

```python
await translate_pages(pages, session)
```

## Key input

Generate one short opaque key per bubble. Seed from stable bubble context:

```text
project_id | chapter_index | page_index | bubble_index |
normalized_source_text | rounded_polygon
```

Use a 7-character alphabet without ambiguous characters:

```text
ABCDEFGHJKLMNPQRSTUVWXYZ23456789
```

Controller keeps private maps:

```python
key_to_bubble: dict[str, Bubble]
key_to_page: dict[str, int]
page_to_keys: dict[int, list[str]]
```

Agents receive sanitized keyed input:

```python
@dataclass
class KeyInput:
    key: str
    page_index: int
    order: int          # best-effort reading order
    source_text: str
    bbox: list[int] | None = None  # only for LookAt overlays
```

Translator prompts should not include verbose bbox data. Use `bbox` to render
light hash overlays on LookAt page images.

## KnowledgeProvider

`KnowledgeProvider` is the retrieval layer for the context stage. It reads
chapter brief artifacts, glossary entries, and prior translations. It does not
read legacy knowledge snapshots or notes because those tables are removed by this
architecture.

`KnowledgeProvider` is a small retrieval layer used by the context stage:

```python
previous_context = await knowledge_provider.get_context(
    project_id,
    before_chapter=chapter,
    queries=current_chapter_terms,
)
```

Implementation should use the new store primitives:

- recent `chapter_briefs` before the current chapter,
- `search_briefs(...)` for relevant brief facts/terms/rules,
- `search_translations(...)` for prior source -> target examples,
- glossary lookup for hard terms.

The context stage receives formatted prior context from `KnowledgeProvider` and
does not know the storage layout.

## Stage 1: Chapter context

The context agent reads the full current chapter OCR text once plus selected
prior context from `KnowledgeProvider`. It does not translate the chapter. It
extracts the context needed for consistent page translation.

Input format should be compact:

```text
[p0] #K7Q9M2 なんでここに
[p0] #A3F8TX ドン
[p1] #P9Q2MA ...
```

Context prompt intent:

```text
Read this keyed chapter text and prior context.
Do not translate every line.
Produce one ChapterBrief artifact with:
- chapter/page situation summaries,
- names/terms/glossary candidates,
- xưng hô / speaker style rules if inferable,
- tone/style constraints,
- keys/pages that need visual clarification before translation.
```

Output model:

```python
@dataclass
class ChapterBrief:
    summary: str
    facts: list[str]
    glossary: dict[str, str]
    style_rules: list[str]
    pronoun_rules: list[str]
    page_notes: dict[int, str]
    key_notes: dict[str, str]
    look_requests: list[LookRequest]
```

```python
@dataclass
class LookRequest:
    page_index: int
    keys: list[str]
    query: str
```

Brief facts are guidance, not ground truth. Existing project glossary and prior
knowledge override inferred brief content.

`ChapterBrief` is both working context for the current translation and the saved
artifact for future retrieval. Do not ask the model to output a separate memory
object with duplicated summary/rules.

## Stage 2: LookAt clarification

Controller groups `look_requests` by page and calls LookAt once per page when
possible.

LookAt input:

```text
Page: 4
Query: Determine speaker/tone and whether marked keys are dialogue or SFX.

Attached: full page image with light hash overlays for #K7Q9M2, #A3F8TX.

Related text:
#K7Q9M2: なんでここに
#A3F8TX: ドン
```

LookAt output:

```text
#K7Q9M2: dialogue; girl on left, angry, confronting boy entering.
#A3F8TX: large impact SFX; skip candidate.
```

Controller merges notes into the brief:

```python
brief.key_notes[key] = lookat_note
```

LookAt does not produce final translations. Page translators decide final wording
or skip based on the brief and notes.

## Stage 3: Parallel page translation

Fork one worker per page or page-window after the brief is ready.

Worker input must be a slice, not the full chapter:

```text
Translate page 4.

Global rules:
- 先輩 => senpai / anh chị khóa trên, depending on tone
- A calls B "cậu"; B calls A "anh"
- Tone: tense, casual

Page note:
- Girl confronts boy entering the room.

Visual/key notes:
#K7Q9M2: girl on left, angry, confronting boy.
#A3F8TX: impact SFX; skip candidate.

Keys:
#K7Q9M2 なんでここに
#A3F8TX ドン
#P9Q2MA ...
```

Output uses the same keyed protocol for every page worker:

```text
#K7Q9M2 | ok | Sao cậu lại ở đây?
#A3F8TX | skip |
#P9Q2MA | ok | ...
```

Tool form should use the same fields:

```json
{
  "items": [
    {"key": "K7Q9M2", "status": "ok", "text": "Sao cậu lại ở đây?"},
    {"key": "A3F8TX", "status": "skip", "text": ""}
  ],
  "done": false
}
```

Status values for page translation:

| Status | Meaning |
| --- | --- |
| `ok` | final target-language translation |
| `skip` | do not render this source text |
| `need_look` | brief lacked visual context; controller may call LookAt and repair |

After LookAt has answered a key, `need_look` for that key is invalid.

## Brief slicing

Never send the full brief to every worker by default.

Each page worker gets:

- global glossary and hard rules,
- compact style/pronoun rules,
- page note for its page,
- key notes for keys on its page,
- optional previous/next page one-line context,
- the page/page-window keys.

Do not include:

- full chapter OCR,
- all page notes,
- all LookAt notes,
- long free-form summaries when hard rules are enough.

This is the main token-saving property of the design.

## Brief persistence

`ChapterBrief` is saved as the chapter knowledge artifact after translation
succeeds. There is no separate `ChapterMemory` object and no final summarization
call.

```text
context stage creates ChapterBrief
LookAt notes merge into ChapterBrief.key_notes
page translation validates all keys
store saves ChapterBrief for this chapter
```

Future context stages retrieve previous brief artifacts through
`KnowledgeProvider`. The provider may return whole brief artifacts or selected
sections, depending on token budget and query relevance.

`ChapterBrief` is stored in `chapter_briefs`. Do not store it in a legacy
knowledge snapshot table.

## Storage redesign

This RFC is a clean break from the old SQLite schema. Delete/recreate the local
DB or run a destructive migration when adopting it.

Remove:

```text
knowledge_snapshots
notes
notes_fts
post-translation knowledge consolidation flow
```

Keep:

```text
projects
chapters
glossary
translations
```

Add:

```sql
CREATE TABLE chapter_briefs (
    project_id    INTEGER NOT NULL,
    chapter       REAL NOT NULL,
    brief_json    TEXT NOT NULL,
    summary       TEXT,
    terms_text    TEXT,
    facts_text    TEXT,
    rules_text    TEXT,
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at    TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (project_id, chapter),
    FOREIGN KEY (project_id, chapter)
      REFERENCES chapters(project_id, idx)
      ON DELETE CASCADE
);
```

Update `translations` to persist the key protocol:

```sql
CREATE TABLE translations (
    project_id       INTEGER NOT NULL,
    chapter          REAL NOT NULL,
    page             INTEGER NOT NULL,
    idx              INTEGER NOT NULL,
    key              TEXT NOT NULL,
    source_text      TEXT NOT NULL,
    translated_text  TEXT NOT NULL,
    status           TEXT NOT NULL DEFAULT 'ok',
    polygon          TEXT,
    font_size_px     INTEGER,
    created_at       TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (project_id, chapter, page, idx),
    UNIQUE(project_id, chapter, key),
    FOREIGN KEY (project_id, chapter)
      REFERENCES chapters(project_id, idx)
      ON DELETE CASCADE
);
```

`brief_json` is the source of truth. `summary`, `terms_text`, `facts_text`, and
`rules_text` are searchable projections for `KnowledgeProvider`. Add FTS for
`chapter_briefs` later if needed; do not block the initial rebuild on it.

New store API:

```python
async def save_chapter_brief(project_id, chapter, brief) -> None: ...
async def get_chapter_brief(project_id, chapter) -> dict | None: ...
async def get_recent_chapter_briefs(project_id, before_chapter, limit=3) -> list[dict]: ...
async def search_briefs(project_id, queries, limit=10) -> list[str]: ...
```

Remove store API:

```python
get_knowledge
save_knowledge
add_note
```

Delete/retranslate must remove `chapter_briefs` and keyed `translations` for the
chapter.

## Validation and repair

The controller validates every page output before applying it.

Reject or normalize:

1. unknown keys,
2. keys outside the page worker's active set,
3. duplicate conflicting keys,
4. `ok` with empty text,
5. `skip` with non-empty text (normalize to empty or reject),
6. `need_look` after LookAt notes exist,
7. text containing protocol markers or opaque keys,
8. source-copy output when source and target languages differ,
9. hard glossary violations where available.

Repair only failed keys/pages:

```text
Repair page 4.

Validation errors:
#P9Q2MA failed: ok text was empty

Keys:
#P9Q2MA source...
```

If repair still fails after LookAt clarification, return a clear
`TranslationStalled` error for those keys rather than silently accepting a bad
translation.

## Concurrency and cost

After the brief is built, page workers may run in parallel with a bounded
concurrency limit:

```python
await gather_limited(page_jobs, concurrency=3)
```

Recommended policy:

```text
weak model:   one page per worker
medium model: 2-3 pages or max 20-25 keys per worker
strong model: larger page-window if validation remains clean
```

The context stage is one larger call; page translation calls stay small. This is
expected to cost fewer tokens than repeatedly sending chapter context through a
page-by-page agent loop.

## Reading order policy

Reading order is best-effort. It helps build page/window inputs but is not used
to apply translations.

- Page workers may infer a better local dialogue order from text and brief.
- LookAt can clarify local order/speaker when requested.
- Output order is irrelevant; controller maps by key.

## Cache and conversation policy

Do not keep one long chapter chat.

Use stateless calls:

```text
context stage: stable context prompt + full keyed chapter text
page stage:    stable page prompt + brief slice + page keys
repair stage:  stable repair prompt + failed keys + brief slice
```

Controller state carries progress and accepted translations.

## Logging

Replace pass logs with workflow logs:

```text
context chapter bubbles=110 look_requests=4
brief created facts=5 glossary=3 page_notes=17
lookat page=4 keys=2
lookat page=8 keys=2
translate page=4 keys=7 ok=6 skip=1 invalid=0 parser=tool
translate page=5 keys=5 ok=4 need_look=1 parser=text
repair page=5 keys=1 ok=1
brief saved chapter=12
complete pages=17 bubbles=110
```

## Production scope

Required:

- opaque keyed bubbles,
- redesigned SQLite schema with `chapter_briefs` and keyed `translations`,
- ChapterContextAgent,
- ChapterBrief emitted by the context stage,
- KnowledgeProvider retrieval of previous brief artifacts,
- ChapterBrief page/key slicing,
- page-level LookAt clarification,
- parallel PageTranslator workers,
- keyed tool output + keyed text fallback,
- deterministic validation,
- page/key repair only,
- save ChapterBrief after successful translation,
- removal of legacy knowledge snapshots/notes flow,
- bounded concurrency,
- stateless calls.

Not core:

- long chapter conversation,
- direct LookAt final translations,
- dynamic JSON schema / active-key enum,
- semantic QA pass over every bubble,
- extra end-of-chapter summary call,
- complex scene planning before the brief pipeline is stable.

## Required tests

1. Context agent output can be parsed into `ChapterBrief`.
2. SQLite schema creates `chapter_briefs` and keyed `translations`.
3. Legacy `knowledge_snapshots` / `notes` tables and APIs are removed.
4. KnowledgeProvider can return previous brief context from `chapter_briefs`.
5. LookAt requests are grouped by page.
6. LookAt notes merge into `brief.key_notes`.
7. Page worker receives only its brief slice, not full chapter OCR.
8. Shuffled keyed page output maps to correct bubbles.
9. Unknown or out-of-page keys are rejected.
10. Missing keys trigger page repair only.
11. `need_look` after LookAt notes is rejected.
12. Page translations can run with bounded concurrency.
13. Completion requires every key to be accepted as `ok` or `skip`.
14. `ChapterBrief` is saved only after successful translation.
15. No extra end-of-chapter summary call is required.

## Decision summary

Adopt:

```text
Chapter Brief
+ LookAt clarification
+ parallel keyed page translation
+ controller validation and repair
```

This preserves chapter-level consistency without repeatedly sending full context
to every translation call. It also saves the same brief as future knowledge
without another model call, while keeping final rendering safe through keyed
mapping.
