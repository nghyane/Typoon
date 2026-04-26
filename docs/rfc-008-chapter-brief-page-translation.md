# RFC-008: Chapter Brief Artifacts and Parallel Keyed Page Translation

## Status

Implemented. Full rebuild of translation and knowledge architecture.

## Problem

The old translation flow used fixed passes (text chunks, image follow-up, retry
missing) with line-order parsing, sequential IDs, and a post-translation
knowledge agent. This was fragile with weaker models and caused inconsistent
terms, xưng hô, and tone across pages.

## Decision

Context-first workflow with saved brief artifacts:

```text
1. ContextAgent reads full keyed chapter text once.
   Queries stored knowledge and inspects page images on demand.
   Submits one ChapterBrief artifact.

2. PageAgent per page-window translates using brief slices.
   Validation errors returned as tool responses for in-conversation retry.
```

Old knowledge architecture replaced:

```text
remove: knowledge_snapshots, notes, post-translation knowledge agent
add:    chapter_briefs as the knowledge artifact
change: translations persist opaque key + status
```

## Pipeline

```text
assign opaque keys (blake2s, 7-char)
-> ContextAgent (Agent protocol)
     tools: search_knowledge, look_at, submit_chapter_brief
     queries knowledge on demand, inspects pages with key overlays
     submits ChapterBrief
-> PageAgent per window (Agent protocol)
     tool: submit_translations (enum ok|skip)
     receives brief slice only
     retry via conversation (validation errors as tool response)
-> save translations + ChapterBrief artifact
```

## Agents

All implement Agent protocol and run through `agent.run()`.

| Agent | Tools | Completion |
| --- | --- | --- |
| ContextAgent | search_knowledge, look_at, submit_chapter_brief | brief submitted |
| LookAtAgent | submit_visual_notes | notes submitted |
| PageAgent | submit_translations | all keys accepted |

Submit = done, no extra round trip.

## Tools (all strict Pydantic schema)

| Tool | Fields |
| --- | --- |
| submit_chapter_brief | summary, facts, glossary[], rules[], page_notes[], bubble_notes[] |
| search_knowledge | query, scope: enum(all/glossary/briefs/translations) |
| look_at | pages: list[int], keys[], query |
| submit_visual_notes | notes: list[VisualNote(key, note)] |
| submit_translations | items: list[TranslationEdit(key, status: enum ok/skip, text)] |

## Key overlays

Page images sent to LookAt have `#KEY` labels rendered at bubble polygon
centroids. The vision model sees which key maps to which bubble region without
needing bbox data in the prompt.

## Knowledge flow

Before translation: ContextAgent calls `search_knowledge` to query glossary,
previous chapter briefs, and past translations on demand.

After successful translation: ChapterBrief saved as artifact in
`chapter_briefs` table. No separate summary call.

## Storage

```sql
chapter_briefs  (brief_json, summary, terms_text, facts_text, rules_text)
translations    (key, source_text, translated_text, status, polygon, font_size_px)
glossary        (source_term, target_term, notes)
```

Removed: `knowledge_snapshots`, `notes`, `notes_fts`.

## Components

```text
typoon/translation/
  translate.py        orchestration
  context.py          ContextAgent
  look_at.py          LookAtAgent
  page.py             PageAgent + TranslationOp
  brief.py            ChapterBrief model + slicing
  keys.py             opaque key generation
  image.py            page image encoding + key overlay
  prompt.py           all prompts
  tools/
    brief.py          submit_chapter_brief
    submit.py         submit_translations
    visual_notes.py   submit_visual_notes
    look_at.py        look_at
    search_knowledge.py  search_knowledge
```

## Design rules

1. All output via strict tool calls. No text fallback. No regex parsing.
2. No pre-loaded context dump. Agent queries on demand.
3. Submit = done, no extra round trip.
4. Retry via conversation — validation errors as tool response.
5. Key overlays on images — LookAt sees labels at polygon positions.
6. One ChapterBrief artifact — working context and saved knowledge.
7. Brief slicing — page workers get only relevant sections.
8. Opaque keys — model never sees storage IDs.
9. Page windows respect page boundaries.
