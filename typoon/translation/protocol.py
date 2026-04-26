"""Keyed translation protocol parsing and validation."""

from __future__ import annotations

import re
from dataclasses import dataclass

from typoon.domain.bubble import Bubble
from typoon.llm.ir import CallResponse

from .tools.submit import SubmitArgs

_LINE_RE = re.compile(r"^\s*#?([A-Z2-9]{6,8})\s*\|\s*(ok|skip|need_look)\s*\|?\s*(.*)$", re.I)
_LEGACY_RE = re.compile(r"^\s*#?([A-Z2-9]{6,8})\s*:\s*(.*)$")


@dataclass(slots=True)
class TranslationOp:
    key: str
    status: str
    text: str = ""


@dataclass(slots=True)
class ValidationResult:
    accepted: list[TranslationOp]
    need_look: list[TranslationOp]
    invalid: dict[str, str]


def parse_response(resp: CallResponse) -> list[TranslationOp]:
    ops = _parse_tools(resp)
    if ops:
        return ops
    return parse_text(resp.text or "")


def parse_text(text: str) -> list[TranslationOp]:
    ops: list[TranslationOp] = []
    for line in text.splitlines():
        m = _LINE_RE.match(line)
        if m:
            key, status, value = m.group(1), m.group(2).lower(), m.group(3).strip()
            ops.append(TranslationOp(key=key, status=status, text=_strip_quotes(value)))
            continue
        m = _LEGACY_RE.match(line)
        if m:
            key, value = m.group(1), _strip_quotes(m.group(2).strip())
            marker = value.lower()
            if marker == "[skip]":
                ops.append(TranslationOp(key=key, status="skip"))
            elif marker == "[need_look]":
                ops.append(TranslationOp(key=key, status="need_look"))
            elif value:
                ops.append(TranslationOp(key=key, status="ok", text=value))
    return ops


def validate_ops(
    ops: list[TranslationOp],
    *,
    active: set[str],
    key_map: dict[str, Bubble],
    look_notes: dict[str, str],
) -> ValidationResult:
    accepted: list[TranslationOp] = []
    need_look: list[TranslationOp] = []
    invalid: dict[str, str] = {}
    seen: set[str] = set()
    for op in ops:
        key = op.key
        status = op.status.lower()
        text = op.text.strip()
        if key not in key_map:
            invalid[key] = "unknown key"
            continue
        if key not in active:
            invalid[key] = "key outside active page/window"
            continue
        if key in seen:
            invalid[key] = "duplicate key"
            continue
        seen.add(key)
        if status not in {"ok", "skip", "need_look"}:
            invalid[key] = f"invalid status {status}"
            continue
        if status == "ok":
            if not text:
                invalid[key] = "ok text was empty"
                continue
            if key in text or f"#{key}" in text:
                invalid[key] = "translation contains key marker"
                continue
            accepted.append(TranslationOp(key=key, status="ok", text=text))
        elif status == "skip":
            accepted.append(TranslationOp(key=key, status="skip", text=""))
        else:
            if key in look_notes:
                invalid[key] = "need_look already answered by LookAt"
                continue
            need_look.append(TranslationOp(key=key, status="need_look", text=text))
    return ValidationResult(accepted=accepted, need_look=need_look, invalid=invalid)


def _parse_tools(resp: CallResponse) -> list[TranslationOp]:
    ops: list[TranslationOp] = []
    for tc in resp.tool_calls or []:
        if tc.name != "submit_translations":
            continue
        try:
            args = SubmitArgs.model_validate_json(tc.arguments)
        except Exception:
            continue
        for item in args.items:
            ops.append(TranslationOp(key=item.key, status=item.status.value, text=item.text))
    return ops


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1].strip()
    return value
