"""Resume policy — decide what action to take per chapter status."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResumePolicy:
    force: bool = False
    resume_translated: bool = True
    retry_failed: bool = True
    max_retries: int = 0


def _decide(status: str | None, policy: ResumePolicy, retry_count: int = 0) -> str:
    match status:
        case "done":
            return "clean" if policy.force else "skip"
        case "translated":
            return "render" if policy.resume_translated else "skip"
        case "rendering":
            return "render"
        case "translating":
            return "clean"
        case "failed":
            if not policy.retry_failed:
                return "skip"
            if policy.max_retries > 0 and retry_count >= policy.max_retries:
                return "skip"
            return "clean"
        case _:
            return "translate"
