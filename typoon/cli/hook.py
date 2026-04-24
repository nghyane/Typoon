"""Inline CLI progress hook.

Plain scrollback output — one line per event boundary. No alternate
screen, no carriage-return overwriting, no raw key handling. Everything
survives in the terminal history after the command exits.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from typoon.app.events import (
    ChapterDone,
    ChapterSkipped,
    ChapterStart,
    Event,
    Hook,
    KnowledgeDone,
    KnowledgeStart,
    ModelsUnloaded,
    PageErased,
    PageRendered,
    PageScanned,
    PipelineError,
    SeriesProgress,
    ToolResult,
    TranslateDone,
    TranslateStart,
    TranslationReady,
)


class CliHook(Hook):
    """One-line-per-event progress reporter.

    Coarse granularity on purpose: scan / translate / render each
    emit a single summary line when they complete, not per page. This
    keeps the output short and greppable for long runs.
    """

    def __init__(self, log_file: Path | None = None) -> None:
        self._console = Console(stderr=True, highlight=False)
        self._log_file = log_file
        self._log_lines: list[str] = []

        self._chapter: float = 0
        self._pages_total = 0
        self._scan_bubbles = 0
        self._scan_pages_seen = 0
        self._translate_total = 0

    # ── Public logging ───────────────────────────────────────────

    def log(self, msg: str) -> None:
        self._println(msg)

    # ── Event dispatch ───────────────────────────────────────────

    def on(self, event: Event) -> None:  # noqa: C901
        match event:
            case ChapterStart(chapter=ch, pages=n):
                self._chapter = ch
                self._pages_total = n
                self._scan_bubbles = 0
                self._scan_pages_seen = 0
                self._translate_total = 0
                self._println(f"\n[bold]ch{_ch(ch)}[/]  {n} pages")

            case ChapterSkipped(chapter=ch, reason=r):
                self._println(f"  [dim]ch{_ch(ch)} skipped ({r})[/]")

            case ChapterDone(chapter=ch, pages=p, bubbles=b, elapsed=s):
                self._println(f"  [green]✓[/] ch{_ch(ch)}  {p}p {b}b  [dim]{s:.1f}s[/]")

            case PageScanned(page=p, total=t, bubbles=b):
                self._scan_bubbles += b
                self._scan_pages_seen += 1
                if self._scan_pages_seen == t:
                    self._println(f"  [cyan]scan[/]      {self._scan_bubbles} bubbles")

            case TranslateStart(total_bubbles=n):
                self._translate_total = n
                self._println(f"  [yellow]translate[/] {n} bubbles…")

            case TranslateDone():
                pass  # TranslationReady gives the same info with a clearer label

            case TranslationReady(translated=tr, total=n):
                self._println(f"  [green]⚡[/]         {tr}/{n} translated")

            case PageErased(page=p, total=t):
                if p + 1 == t:
                    self._println(f"  [magenta]render[/]    {t}/{t}")

            case PageRendered():
                pass

            case KnowledgeStart():
                pass
            case KnowledgeDone():
                pass

            case ModelsUnloaded(stage=s):
                # Log only — not worth a console line.
                self._log_lines.append(f"unloaded {s}")

            case ToolResult(tool=t):
                self._log_lines.append(f"tool {t}")

            case SeriesProgress():
                pass

            case PipelineError(stage=s, error=e):
                self._println(f"  [red]✗[/] {s}: {e}")

    # ── Lifecycle ────────────────────────────────────────────────

    def start(self) -> None:
        pass

    def stop(self) -> None:
        if self._log_file and self._log_lines:
            self._log_file.parent.mkdir(parents=True, exist_ok=True)
            self._log_file.write_text("\n".join(self._log_lines) + "\n")

    # ── Internals ────────────────────────────────────────────────

    def _println(self, msg: str) -> None:
        self._console.print(msg)
        self._log_lines.append(_strip_markup(msg))


# ── Helpers ──────────────────────────────────────────────────────


def _ch(ch: float) -> str:
    return str(int(ch)) if ch == int(ch) else str(ch)


def _strip_markup(s: str) -> str:
    import re
    return re.sub(r"\[/?[^\]]*\]", "", s)
