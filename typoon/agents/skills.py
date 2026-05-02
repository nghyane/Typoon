"""Translation skill discovery and loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

from typoon.llm.ir import ToolDef

_SKILLS_DIR = Path(__file__).parent / "skills"

LOAD_SKILL_TOOL = ToolDef(
    name="load_skill",
    description="Load the full SKILL.md instructions for one relevant skill.",
    parameters={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Skill name exactly as listed in the available skill catalog",
            }
        },
        "required": ["name"],
    },
)


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str


class SkillLibrary:
    """Skill catalog bound to a target language."""

    def __init__(self, target_lang: str) -> None:
        self._lang = target_lang
        self._skills = _load_skills(target_lang)

    def catalog(self) -> str:
        lines = ["<available_skills>"]
        for skill in self._skills:
            lines.append(
                f"  <skill name={quoteattr(skill.name)}>{escape(skill.description)}</skill>"
            )
        lines.append("</available_skills>")
        return "\n".join(lines)

    def load(self, name: str) -> str:
        safe = name.strip().lower()
        for skill in self._skills:
            if skill.name == safe:
                meta = f"name: {skill.name}\ndescription: {skill.description}"
                return f"---\n{meta}\n---\n\n{skill.body}".strip()
        return f"Translation skill not found: {name}"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_skills(target_lang: str) -> list[Skill]:
    skills: list[Skill] = []
    for path in _skill_paths(target_lang):
        text = path.read_text("utf-8")
        frontmatter, body = _split_frontmatter(text)
        name = frontmatter.get("name") or path.parent.name
        description = frontmatter.get("description", "")
        if name and description:
            skills.append(Skill(name, description, body.strip()))
    return skills


def _skill_paths(target_lang: str) -> list[Path]:
    roots = [_SKILLS_DIR / "global", _SKILLS_DIR / target_lang]
    paths: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.glob("*/SKILL.md")):
            if path not in seen:
                paths.append(path)
                seen.add(path)
    return paths


def _split_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end == -1:
        return {}, text
    raw = text[4:end]
    body = text[end + len("\n---\n"):]
    return _parse_simple_yaml(raw), body


def _parse_simple_yaml(raw: str) -> dict:
    data: dict = {}
    current_map: str | None = None
    for line in raw.splitlines():
        if not line.strip():
            continue
        if line.startswith("  ") and current_map:
            key, value = _key_value(line.strip())
            if key:
                data[current_map][key] = value
            continue
        key, value = _key_value(line)
        if not key:
            continue
        if value == "":
            data[key] = {}
            current_map = key
        else:
            data[key] = value
            current_map = None
    return data


def _key_value(line: str) -> tuple[str, str]:
    if ":" not in line:
        return "", ""
    key, value = line.split(":", 1)
    return key.strip(), value.strip().strip('"')
