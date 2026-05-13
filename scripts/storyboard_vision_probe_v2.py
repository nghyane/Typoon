"""Step 3 — feed deterministic name candidates + glossary into vision prompt.

Tests whether including a candidate list (from frequency-based extraction
of OCR text, here mocked) helps the model BIND visual characters to names
rather than emitting generic descriptors.

Uses project [vision_agent] config — same provider as production.
"""

from __future__ import annotations

import asyncio
import base64
import time
from pathlib import Path

from typoon.config import load_config
from typoon.llm.ir import ContentPart, Message
from typoon.providers import make_vision_provider

ROOT = Path(__file__).resolve().parents[1]
STORYBOARD = ROOT / "debug-runs" / "storyboard_proto" / "grid_2x2.jpg"
OUT = ROOT / "debug-runs" / "storyboard_proto" / "vision_response_v2.md"

# Mock name candidates — in production these come from frequency-based
# extraction over the chapter OCR text + community_glossary aggregate.
# Picked to test:
#   - "Saran" (made up — should NOT appear if model doesn't see evidence)
#   - "elf" (matches the pointed-ear figure visible in storyboard)
#   - "Brandi" (made up — should be rejected if not present)
NAME_CANDIDATES = ["elf", "Saran", "Brandi"]

# Mock prior glossary from community (empty in this test — first material translation)
COMMUNITY_GLOSSARY: dict[str, str] = {}


SYSTEM = """\
You are analyzing a manga storyboard — a single image containing 4 consecutive pages
of a manga chapter, laid out 2×2. Each page is labelled "page 0" through "page 3".

Your job is to extract material-level context: characters, their visual gender,
scene mood, and register. The goal is to feed a downstream translator stable
character info so they pick correct pronouns and honorifics.

You will be given a list of NAME CANDIDATES extracted from chapter text. For each
character you see, decide:
  - if a candidate name matches what you see visually → use that name
  - if a candidate is provided but no character in the image matches → mark it "not visible"
  - if you see a character that has no candidate name → use a short visual descriptor

Do NOT translate. Do NOT transcribe bubble text. Visual signal only.
"""


def build_user(name_candidates: list[str], glossary: dict[str, str]) -> str:
    candidates_block = (
        "\n".join(f"- {n}" for n in name_candidates) if name_candidates
        else "(none — first chapter, no extracted names)"
    )
    glossary_block = (
        "\n".join(f"- {k} → {v}" for k, v in glossary.items()) if glossary
        else "(empty — this is a fresh material)"
    )

    return f"""\
## Name candidates
{candidates_block}

## Existing glossary (from community translations)
{glossary_block}

## Task
Reply in this exact structure:

### Characters
For each distinct character visible (max 6), one bullet:
- **<name>** (gender: male | female | unknown) — visual: 1-line description (clothing, hairstyle, age). Where <name> is either a candidate from the list above, or a short descriptor if no candidate matches.

### Candidate verdicts
For EACH candidate in the list, one line: "candidate: visible | not visible | uncertain".

### Scene
2 sentences. Setting and what's happening.

### Register & mood
2 sentences. Tone the translator should preserve.

### Pacing
1 sentence.

### Page-by-page note
List page 0..3 only if they differ in setting/mood.
"""


async def main() -> None:
    config, _ = load_config()
    provider = make_vision_provider(config)
    model = config.vision_agent.model

    raw = STORYBOARD.read_bytes()
    data_uri = "data:image/jpeg;base64," + base64.b64encode(raw).decode()

    user = build_user(NAME_CANDIDATES, COMMUNITY_GLOSSARY)

    messages = [
        Message.system(SYSTEM),
        Message.user_parts([
            ContentPart.of_text(user),
            ContentPart.of_image(data_uri),
        ]),
    ]

    t0 = time.monotonic()
    resp = await provider.call(messages, [])
    elapsed = time.monotonic() - t0

    text = resp.text or "(empty response)"
    print(f"=== vision response v2 ({elapsed:.1f}s, {len(text)} chars) ===\n")
    print(text)

    OUT.write_text(
        f"# Vision probe v2 — with name candidates\n\n"
        f"- model: {model}\n"
        f"- elapsed: {elapsed:.2f}s\n"
        f"- candidates given: {NAME_CANDIDATES}\n"
        f"- glossary given: {COMMUNITY_GLOSSARY or '(empty)'}\n\n"
        f"## Response\n\n{text}\n",
        encoding="utf-8",
    )
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
