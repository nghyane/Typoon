"""Step 2 of storyboard prototype — send grid_2x2 to vision and check signal.

Uses the project's own [vision_agent] provider from config.toml. No env-var
overrides, no hacky model pinning — same path the real pipeline uses.

Output written to debug-runs/storyboard_proto/vision_response.md
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
OUT = ROOT / "debug-runs" / "storyboard_proto" / "vision_response.md"


SYSTEM = """\
You are analyzing a manga storyboard — a single image containing 4 consecutive pages
of a manga chapter, laid out 2×2. Each page is labelled "page 0" through "page 3"
at the top of its panel.

Your job is to extract material-level context that will help downstream translators
maintain consistency: characters, their visual gender, scene mood, and overall register.

Do NOT translate anything. Do NOT transcribe bubble text. The OCR pass is separate.
Focus on visual signal only.
"""

USER = """\
Analyze this storyboard. Reply in this exact structure (markdown):

## Characters
For each distinct character visible (max 6), one bullet:
- **<name-or-descriptor>** (gender: male | female | unknown): 1-line visual description (clothing, hairstyle, age range). If the character has no on-page name, use a short descriptor like "blonde girl" or "older man with glasses".

## Scene
2-3 sentences. Setting, time of day, indoor/outdoor, what's happening at a high level.

## Register & mood
2-3 sentences. Is this action, comedy, drama, romance? Formal speech or casual? Tense or relaxed? What style/tone should the translator preserve?

## Pacing
1 sentence. Is this a talking-heads sequence, an action burst, a transition, or a montage?

## Page-by-page note (only if pages differ)
Skip if all 4 pages share the same setting/mood. Otherwise list page 0..3 with the differentiator.
"""


async def main() -> None:
    config, _ = load_config()
    provider = make_vision_provider(config)
    model = config.vision_agent.model

    raw = STORYBOARD.read_bytes()
    data_uri = "data:image/jpeg;base64," + base64.b64encode(raw).decode()

    messages = [
        Message.system(SYSTEM),
        Message.user_parts([
            ContentPart.of_text(USER),
            ContentPart.of_image(data_uri),
        ]),
    ]

    t0 = time.monotonic()
    resp = await provider.call(messages, [])
    elapsed = time.monotonic() - t0

    text = resp.text or "(empty response)"
    print(f"=== vision response ({elapsed:.1f}s, {len(text)} chars) ===\n")
    print(text)

    OUT.write_text(
        f"# Vision probe — storyboard 2x2\n\n"
        f"- model: {model}\n"
        f"- elapsed: {elapsed:.2f}s\n"
        f"- image: {STORYBOARD.relative_to(ROOT)} ({STORYBOARD.stat().st_size // 1024} KB)\n\n"
        f"## Response\n\n{text}\n",
        encoding="utf-8",
    )
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
