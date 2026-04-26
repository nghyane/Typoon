"""Keyed translation protocol: tool-call parsing and validation."""

from __future__ import annotations

from dataclasses import dataclass

from typoon.domain.bubble import Bubble
from typoon.llm.ir import CallResponse

from .tools.submit import SubmitArgs


@dataclass(slots=True)
class TranslationOp:
    key: str
    status: str  # ok | skip | need_look
    text: str = ""


@dataclass(slots=True)
class ValidationResult:
    accepted: list[TranslationOp]
    need_look: list[TranslationOp]
    invalid: dict[str, str]


def parse_response(resp: CallResponse) -> list[TranslationOp]:
    """Parse submit_translations tool calls. Raises if no valid tool call."""
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
    if not ops:
        raise ValueError(
            f"Model did not call submit_translations. "
            f"Text response: {(resp.text or '')[:200]}"
        )
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
