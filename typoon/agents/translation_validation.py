"""Validation for translated bubble operations."""

from __future__ import annotations

import re
from dataclasses import dataclass

from typoon.agents.brief import ChapterBrief
from typoon.domain.scan import BubbleKey


_TOKEN_RE = re.compile(r"(?<!\w)(?:[A-Z]{1,4}\d[A-Z0-9:/.,+%-]*|\d{1,4}(?:[:/%$¥₩€£]|[A-Za-z]{1,4})[A-Z0-9:/.,+%-]*)(?!\w)")


@dataclass(frozen=True)
class TranslationIssue:
    key: str
    message: str


def validate_translation_ops(
    ops,
    key_map: dict[str, BubbleKey],
    brief: ChapterBrief,
) -> list[TranslationIssue]:
    issues: list[TranslationIssue] = []
    for op in ops:
        if op.kind == "skip":
            continue
        source = key_map[op.key].source_text
        issues.extend(_missing_source_tokens(op.key, source, op.text))
        issues.extend(_missing_glossary_terms(op.key, source, op.text, brief))
    return issues


def _missing_source_tokens(key: str, source: str, translated: str) -> list[TranslationIssue]:
    issues: list[TranslationIssue] = []
    for token in sorted(set(_TOKEN_RE.findall(source))):
        if token not in translated:
            issues.append(TranslationIssue(key, f'preserve token "{token}"'))
    return issues


def _missing_glossary_terms(
    key: str,
    source: str,
    translated: str,
    brief: ChapterBrief,
) -> list[TranslationIssue]:
    issues: list[TranslationIssue] = []
    source_fold = source.casefold()
    translated_fold = translated.casefold()
    for src, target in brief.glossary.items():
        if not src or not target:
            continue
        if src.casefold() in source_fold and target.casefold() not in translated_fold:
            issues.append(TranslationIssue(key, f'use glossary "{src}" -> "{target}"'))
    return issues


def issue_prompt(issues: list[TranslationIssue]) -> str:
    lines = [f"- {issue.key}: {issue.message}" for issue in issues]
    keys = ", ".join(sorted({issue.key for issue in issues}))
    return (
        "Validation failed for these ids. Reply with a <translations> block for ONLY "
        f"these ids: {keys}\n" + "\n".join(lines)
    )
