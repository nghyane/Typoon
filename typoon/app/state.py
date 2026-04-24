"""Chapter state machine.

Pure data — no logic, no dependencies.
"""

from __future__ import annotations

from enum import StrEnum


class ChapterState(StrEnum):
    PENDING = "pending"
    SCANNING = "scanning"
    TRANSLATING = "translating"
    TRANSLATED = "translated"
    RENDERING = "rendering"
    DONE = "done"
    FAILED = "failed"
