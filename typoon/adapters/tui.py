"""Unified TUI — single fullscreen shell for browse + pipeline views.

Owns one Live, one key reader, one layout chrome.
Implements Hook so pipeline events render in the same screen.

┌─ typoon ─────────────────────────────────────────────────────┐
│ [left panel]                         │ [right panel]         │
│  browse: project table               │  browse: chapters     │
│  pipeline: full-width activity view                          │
├──────────────────────────────────────────────────────────────┤
│ status bar / input bar                                       │
└──────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import sys
import termios
import threading
import time
import tty
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from typoon.events import (
    ChapterDone,
    ChapterSkipped,
    ChapterStart,
    Event,
    Hook,
    KnowledgeDone,
    KnowledgeStart,
    LLMCall,
    LLMResponse,
    LLMText,
    LLMThinking,
    ModelsUnloaded,
    PageErased,
    PageRendered,
    PageScanned,
    PipelineError,
    SeriesProgress,
    ToolCallStart,
    ToolResult,
    TranslateDone,
    TranslateStart,
    TranslationReady,
)

console = Console()

_CH_ICONS = {"done": "[green]✓[/]", "failed": "[red]✗[/]", "running": "[yellow]⟳[/]",
             "pending": "[dim]·[/]", "skipped": "[dim]−[/]"}


# ── Result from browse mode ──────────────────────────────────────

@dataclass
class BrowseResult:
    action: str = "quit"  # quit | resume | add_url | add_path
    project: dict | None = None
    input_value: str = ""
    from_ch: float = 0
    to_ch: float = 0
    force: bool = False


# ── TUI ──────────────────────────────────────────────────────────

class TUI(Hook):
    """Single fullscreen app — browse projects and monitor pipeline."""

    def __init__(self, log_file: Path | None = None) -> None:
        # ── Terminal ─────────────────────────────────────
        self._live: Live | None = None
        self._old_termios = None
        self._key_thread: threading.Thread | None = None
        self._running = False

        # ── Mode ─────────────────────────────────────────
        self._mode = "browse"  # browse | pipeline

        # ── Browse state ─────────────────────────────────
        self._projects: list[dict] = []
        self._chapters_map: dict[int, list[dict]] = {}
        self._cursor = 0
        self._input_mode = ""   # "" | "url" | "path"
        self._input_buf = ""
        self._input_prompt = ""
        self._browse_result = BrowseResult()
        self._browse_event: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        # ── Pipeline state ───────────────────────────────
        self._log: deque[Text] = deque(maxlen=50)
        self._full_log: list[str] = []
        self._log_file = log_file
        self._tool_queue: list[tuple[str, str]] = []
        self._t0 = time.monotonic()

        self._project_name = ""
        self._project_lang = ""
        self._project_provider = ""
        self._project_done = 0
        self._project_failed = 0
        self._project_skipped = 0
        self._project_total = 0

        self._chapter: float = 0
        self._phase = "init"
        self._pages_total = 0
        self._bubbles = 0
        self._translated = 0
        self._total_bubbles = 0
        self._pages_rendered = 0
        self._render_total = 0
        self._llm_turns = 0
        self._llm_ms = 0.0
        self._scan_ms = 0.0
        self._erase_ms = 0.0

        self._streaming_agent = ""
        self._streaming_turn = 0
        self._thinking_buf = ""
        self._text_buf = ""
        self._stream_line = ""  # current LLM streaming preview

        # Chapter timing
        self._ch_t0 = 0.0
        self._chapter_times: list[float] = []

        # Chapter queue: (ch_num, status) for grid display
        self._ch_queue: list[tuple[float, str]] = []

        self.quit_requested = False
        self.paused = False

    # ── Lifecycle ────────────────────────────────────────

    def start(self) -> None:
        self._running = True
        self._live = Live(self._build(), console=console, screen=True, refresh_per_second=8)
        self._live.start()
        fd = sys.stdin.fileno()
        self._old_termios = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        self._key_thread = threading.Thread(target=self._key_loop, daemon=True)
        self._key_thread.start()
        self._tick_thread = threading.Thread(target=self._tick_loop, daemon=True)
        self._tick_thread.start()

    def stop(self) -> None:
        self._running = False
        if self._old_termios is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_termios)
            self._old_termios = None
        if self._live:
            self._live.stop()
            self._live = None
        if self._log_file and self._full_log:
            self._log_file.parent.mkdir(parents=True, exist_ok=True)
            self._log_file.write_text("\n".join(self._full_log) + "\n")

    def _tick_loop(self) -> None:
        """Rebuild layout 8fps — smooth streaming text."""
        while self._running:
            self._refresh()
            time.sleep(0.125)

    def _refresh(self) -> None:
        if self._live:
            try:
                self._live.update(self._build())
            except Exception:
                pass

    # ── Mode switching ───────────────────────────────────

    def set_projects(self, projects: list[dict], chapters_map: dict[int, list[dict]]) -> None:
        self._projects = projects
        self._chapters_map = chapters_map
        self._cursor = min(self._cursor, max(0, len(projects) - 1))

    async def browse(self) -> BrowseResult:
        """Switch to browse mode. Awaits until user picks an action."""
        self._mode = "browse"
        self._browse_result = BrowseResult()
        self._loop = asyncio.get_running_loop()
        self._browse_event = asyncio.Event()
        await self._browse_event.wait()
        return self._browse_result

    def _signal_browse(self) -> None:
        """Thread-safe: signal the async browse() to return."""
        if self._loop and self._browse_event:
            self._loop.call_soon_threadsafe(self._browse_event.set)

    async def wait_for_key(self) -> None:
        """Show 'press any key' and wait. Used after pipeline finishes."""
        self._mode = "done"
        self._loop = asyncio.get_running_loop()
        self._browse_event = asyncio.Event()
        await self._browse_event.wait()

    def switch_to_pipeline(self, name: str = "", lang: str = "", provider: str = "") -> None:
        """Switch to pipeline view."""
        self._flush_stdin()
        self._mode = "pipeline"
        self._t0 = time.monotonic()
        self._log.clear()
        self._reset_pipeline()
        self._project_name = name
        self._project_lang = lang
        self._project_provider = provider

    @staticmethod
    def _flush_stdin() -> None:
        """Drain any buffered input so stale keys don't leak into new mode."""
        import select
        while select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.read(1)

    def _reset_pipeline(self) -> None:
        self._project_done = 0
        self._project_failed = 0
        self._project_skipped = 0
        self._project_total = 0
        self._ch_queue.clear()
        self._full_log.clear()
        self._tool_queue.clear()
        self._stream_line = ""
        self._thinking_buf = ""
        self._text_buf = ""
        self._chapter_times.clear()
        self._reset_chapter()
        self.quit_requested = False
        self.paused = False

    def _reset_chapter(self) -> None:
        self._pages_total = 0
        self._bubbles = 0
        self._scan_ms = 0.0
        self._translated = 0
        self._total_bubbles = 0
        self._llm_turns = 0
        self._llm_ms = 0.0
        self._pages_rendered = 0
        self._render_total = 0
        self._erase_ms = 0.0

    # ── Log ──────────────────────────────────────────────

    def log(self, msg: str) -> None:
        self._emit(msg)

    def _emit(self, line: str | Text) -> None:
        if isinstance(line, str):
            line = Text.from_markup(line)
        self._log.append(line)
        self._full_log.append(line.plain)

    def _flush_tools(self) -> None:
        if not self._tool_queue:
            return
        groups: list[tuple[str, int]] = []
        for tool, _ in self._tool_queue:
            if groups and groups[-1][0] == tool:
                groups[-1] = (tool, groups[-1][1] + 1)
            else:
                groups.append((tool, 1))
        parts = ", ".join(f"{t} ×{c}" if c > 1 else t for t, c in groups)
        self._emit(f"           [dim]→ {parts}[/]")
        self._tool_queue.clear()

    # ── Key handling ─────────────────────────────────────

    def _key_loop(self) -> None:
        while self._running:
            try:
                ch = sys.stdin.read(1)
            except (OSError, ValueError):
                break
            if not ch:
                break
            if self._mode == "done":
                self._signal_browse()  # any key → back to browse
            elif self._mode == "browse":
                self._handle_browse_key(ch)
            else:
                self._handle_pipeline_key(ch)

    def _handle_browse_key(self, ch: str) -> None:
        if self._input_mode:
            self._handle_input_key(ch)
            return

        n = len(self._projects)

        if ch == "\x1b":
            seq = sys.stdin.read(2)
            if seq == "[A":
                self._cursor = max(0, self._cursor - 1)
            elif seq == "[B":
                self._cursor = min(n - 1, self._cursor + 1) if n else 0
            else:
                self._browse_result = BrowseResult(action="quit")
                self._signal_browse()
        elif ch in ("q", "Q"):
            self._browse_result = BrowseResult(action="quit")
            self._signal_browse()
        elif ch in ("k", "K"):
            self._cursor = max(0, self._cursor - 1)
        elif ch in ("j", "J"):
            self._cursor = min(n - 1, self._cursor + 1) if n else 0
        elif ch in ("\r", "\n") and n > 0:
            self._browse_result = BrowseResult(
                action="resume", project=self._projects[self._cursor],
            )
            self._signal_browse()
        elif ch in ("r", "R") and n > 0:
            self._input_mode = "range"
            self._input_buf = ""
            self._input_prompt = "Range (e.g. 5-10): "
        elif ch in ("a", "A"):
            self._input_mode = "url"
            self._input_buf = ""
            self._input_prompt = "URL: "
        elif ch in ("d", "D"):
            self._input_mode = "path"
            self._input_buf = ""
            self._input_prompt = "Path: "
        elif ch in ("s", "S") and n > 0:
            self._browse_result = BrowseResult(
                action="resume", project=self._projects[self._cursor],
                force=True,
            )
            self._signal_browse()

    def _handle_input_key(self, ch: str) -> None:
        if ch == "\x1b":
            self._input_mode = ""
            self._input_buf = ""
        elif ch in ("\r", "\n"):
            value = self._input_buf.strip()
            if value:
                if self._input_mode == "range":
                    from_ch, to_ch = self._parse_range(value)
                    self._browse_result = BrowseResult(
                        action="resume", project=self._projects[self._cursor],
                        from_ch=from_ch, to_ch=to_ch,
                    )
                else:
                    action = "add_url" if self._input_mode == "url" else "add_path"
                    self._browse_result = BrowseResult(action=action, input_value=value)
                self._signal_browse()
            self._input_mode = ""
            self._input_buf = ""
        elif ch in ("\x7f", "\x08"):
            self._input_buf = self._input_buf[:-1]
        elif ch == "\x15":
            self._input_buf = ""
        elif ch == "\x17":
            self._input_buf = self._input_buf.rsplit(" ", 1)[0] if " " in self._input_buf else ""
        elif ch >= " ":
            self._input_buf += ch

    @staticmethod
    def _parse_range(s: str) -> tuple[float, float]:
        """Parse '5-10', '5-', '-10', '5'. Returns (from, to) with 0=unbounded."""
        import re
        s = s.strip()
        m = re.match(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)?", s)
        if m:
            return float(m.group(1)), float(m.group(2)) if m.group(2) else 0
        m = re.match(r"-\s*(\d+(?:\.\d+)?)", s)
        if m:
            return 0, float(m.group(1))
        m = re.match(r"(\d+(?:\.\d+)?)", s)
        if m:
            v = float(m.group(1))
            return v, v
        return 0, 0

    def _handle_pipeline_key(self, ch: str) -> None:
        if ch in ("q", "Q"):
            self.quit_requested = True
            self._emit("[yellow]⏹ quitting after current chapter…[/]")
        elif ch in ("p", "P"):
            self.paused = not self.paused
            if self.paused:
                self._emit("[yellow]⏸ paused — press p to resume[/]")
            else:
                self._emit("[green]▶ resumed[/]")

    # ── Layout (shared chrome) ───────────────────────────

    def _build(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=1),
            Layout(name="body", ratio=1),
            Layout(name="bar", size=1),
        )
        layout["header"].update(self._build_header())
        layout["bar"].update(self._build_bar())

        if self._mode == "browse":
            layout["body"].split_row(
                Layout(self._build_project_list(), name="left", ratio=3),
                Layout(self._build_chapter_detail(), name="right", size=32),
            )
        else:
            layout["body"].update(self._build_pipeline())
        return layout

    def _build_header(self) -> Text:
        if self._mode in ("pipeline", "done"):
            elapsed = time.monotonic() - self._t0
            mins, secs = divmod(int(elapsed), 60)
            t_str = f"{mins}m{secs:02d}s" if mins else f"{secs}s"
            progress = self._project_done + self._project_failed + self._project_skipped
            parts = [f" [bold #58a6ff]typoon[/]"]
            if self._project_name:
                parts.append(f"  [bold]{self._project_name}[/]")
            if self._project_lang:
                parts.append(f"  [dim]{self._project_lang}[/]")
            if self._project_provider:
                parts.append(f"  [dim]{self._project_provider}[/]")
            right = ""
            if self._project_total:
                right = f"  {progress}/{self._project_total} ch"
                eta = self._eta_str()
                if eta:
                    right += f"  [cyan]{eta}[/]"
                right += f"  [dim]{t_str}[/]"
            else:
                right = f"  [dim]{t_str}[/]"
            return Text.from_markup("".join(parts) + right)
        return Text.from_markup(
            " [bold #58a6ff]typoon[/]  [dim]projects[/]"
        )

    def _build_bar(self) -> Text:
        if self._mode == "done":
            return Text.from_markup(" [bold]press any key[/] to continue")
        if self._mode == "browse":
            if self._input_mode:
                return Text.from_markup(
                    f" {self._input_prompt}[bold]{self._input_buf}[/]█"
                    "  [dim]enter[/] submit  [dim]esc[/] cancel"
                )
            return Text.from_markup(
                " [dim]↑↓[/] nav  [dim]enter[/] resume  [dim]r[/] range  "
                "[dim]a[/] add url  [dim]d[/] add dir  "
                "[dim]s[/] sync  [dim]q[/] quit"
            )
        status = ""
        if self.paused:
            status = "  [yellow]⏸ paused[/]"
        elif self.quit_requested:
            status = "  [yellow]⏹ stopping[/]"
        avg = self._avg_str()
        if avg:
            status += f"  [dim]{avg}[/]"
        return Text.from_markup(
            f" [dim]q[/] quit  [dim]p[/] pause{status}"
        )

    # ── Browse views ─────────────────────────────────────

    def _build_project_list(self) -> Panel:
        if not self._projects:
            return Panel(
                "[dim]No projects yet\n\n"
                "  [bold]a[/] add URL    [bold]d[/] add local path[/]",
                title="[bold]Projects[/]", border_style="dim",
            )

        lines: list[Text] = []
        for i, p in enumerate(self._projects):
            chs = self._chapters_map.get(p["id"], [])
            done = sum(1 for c in chs if c["status"] == "done")
            failed = sum(1 for c in chs if c["status"] == "failed")
            total = len(chs)
            icon = "✗" if failed else ("✓" if done == total and total > 0 else "⟳")
            lang = f"{p.get('source_lang', '?')}→{p.get('target_lang', '?')}"
            title = p.get("title", "")[:36]

            if i == self._cursor:
                line = Text(f"  ▸ {title:<38s} {lang:<8s} {done}/{total:<4} {icon}", style="bold")
            else:
                line = Text(f"    {title:<38s} ", style="")
                line.append(f"{lang:<8s} ", style="dim")
                line.append(f"{done}/{total:<4} ", style="dim")
                line.append(icon, style="green" if icon == "✓" else "red" if icon == "✗" else "dim")

            lines.append(line)

        content = Group(*lines)
        return Panel(content, title="[bold]Projects[/]", border_style="dim")

    def _build_chapter_detail(self) -> Panel:
        if not self._projects:
            return Panel("[dim]—[/]", title="[bold]Chapters[/]", border_style="dim")

        p = self._projects[self._cursor]
        chs = self._chapters_map.get(p["id"], [])
        lines: list[str] = [
            f"[bold]{p.get('title', '')}[/]",
            f"[dim]{p.get('source_lang', '?')} → {p.get('target_lang', '?')}[/]",
        ]
        url = p.get("source_url", "")
        if url:
            lines.append(f"[dim]{url[:28]}…[/]" if len(url) > 28 else f"[dim]{url}[/]")
        lines.append("")

        for c in chs[:20]:
            n = int(c["idx"]) if c["idx"] == int(c["idx"]) else c["idx"]
            icon = _CH_ICONS.get(c["status"], "?")
            lines.append(f"  {icon} ch{n}")
        if len(chs) > 20:
            lines.append(f"  [dim]… +{len(chs) - 20} more[/]")

        return Panel("\n".join(lines), title="[bold]Chapters[/]", border_style="dim")

    # ── Pipeline view (full-width) ───────────────────────

    def _build_pipeline(self) -> Panel:
        parts: list[Text | str] = []

        # ── Header line ──────────────────────────────────
        if self._phase != "init":
            ch = self._ch_label(self._chapter)
            parts.append(Text.from_markup(f" [bold]ch{ch}[/]  {self._pages_total} pages"))
        parts.append(Text(""))

        # ── 3-phase status ───────────────────────────────
        parts.append(Text.from_markup(self._phase_line_preprocess()))
        parts.append(Text.from_markup(self._phase_line_translate()))
        parts.append(Text.from_markup(self._phase_line_render()))
        parts.append(Text(""))

        # ── LLM streaming preview (up to 4 lines) ────────
        if self._stream_line:
            parts.append(Text.from_markup(f" {self._stream_line}"))
        buf = self._thinking_buf or self._text_buf
        if buf:
            icon = "[dim italic]💭[/] " if self._thinking_buf else "   "
            stream_lines = buf.replace("\r", "").split("\n")
            for sl in stream_lines[-4:]:
                parts.append(Text.from_markup(f" {icon}[dim]{sl[:90]}[/]"))

        # ── Tool queue ───────────────────────────────────
        if self._tool_queue:
            groups: list[tuple[str, int]] = []
            for tool, _ in self._tool_queue:
                if groups and groups[-1][0] == tool:
                    groups[-1] = (tool, groups[-1][1] + 1)
                else:
                    groups.append((tool, 1))
            tparts = ", ".join(f"{t} ×{c}" if c > 1 else t for t, c in groups)
            parts.append(Text.from_markup(f" [dim]→ {tparts}[/]"))

        # ── Chapter timeline ─────────────────────────────
        if self._ch_queue:
            parts.append(Text(""))
            parts.append(Text.from_markup(f" {self._build_chapter_grid()}"))

        # ── Log ──────────────────────────────────────────
        log_entries = list(self._log)
        if log_entries:
            parts.append(Text(""))
            for entry in log_entries[-10:]:
                parts.append(entry if isinstance(entry, Text) else Text(str(entry)))

        return Panel(Group(*parts), border_style="dim")

    # ── Phase lines ──────────────────────────────────────

    @staticmethod
    def _progress_bar(current: int, total: int, width: int = 20) -> str:
        if total <= 0:
            return "[dim]" + "░" * width + "[/]"
        filled = int(width * current / total)
        empty = width - filled
        return f"[green]{'█' * filled}[/][dim]{'░' * empty}[/]"

    def _phase_line_preprocess(self) -> str:
        p = self._phase
        if p == "init":
            return " [dim]preprocess[/]  [dim]waiting[/]"
        if p == "preprocess":
            return f" [bold cyan]preprocess[/]  [yellow]⟳[/] {self._pages_total}p…"
        return (f" [dim]preprocess[/]  [green]✓[/] "
                f"{self._pages_total}p {self._bubbles}b "
                f"[dim]{self._scan_ms:.0f}ms[/]")

    def _phase_line_translate(self) -> str:
        p = self._phase
        phase_order = {"preprocess": 0, "translate": 1, "render": 2, "knowledge": 2, "done": 3}
        rank = phase_order.get(p, 0)
        if rank < 1:
            return " [dim]translate[/]   [dim]waiting[/]"
        if p == "translate":
            bar = self._progress_bar(self._translated, self._total_bubbles)
            return (f" [bold yellow]translate[/]   {bar}  "
                    f"{self._translated}/{self._total_bubbles}  "
                    f"t{self._llm_turns}  [dim]{self._llm_ms / 1000:.1f}s[/]")
        return (f" [dim]translate[/]   [green]✓[/] "
                f"{self._translated}/{self._total_bubbles}  "
                f"t{self._llm_turns}  [dim]{self._llm_ms / 1000:.1f}s[/]")

    def _phase_line_render(self) -> str:
        p = self._phase
        phase_order = {"preprocess": 0, "translate": 1, "render": 2, "knowledge": 2, "done": 3}
        rank = phase_order.get(p, 0)
        if rank < 2:
            return " [dim]render[/]      [dim]waiting[/]"
        tot = self._render_total or self._pages_total
        if p in ("render", "knowledge"):
            bar = self._progress_bar(self._pages_rendered, tot)
            return (f" [bold magenta]render[/]      {bar}  "
                    f"{self._pages_rendered}/{tot}  "
                    f"[dim]{self._erase_ms:.0f}ms[/]")
        return (f" [dim]render[/]      [green]✓[/] "
                f"{self._pages_rendered}/{tot}  "
                f"[dim]{self._erase_ms:.0f}ms[/]")

    # ── Chapter timeline ─────────────────────────────────

    def _build_chapter_grid(self) -> str:
        parts: list[str] = []
        for ch_num, status in self._ch_queue:
            n = int(ch_num) if ch_num == int(ch_num) else ch_num
            icon = _CH_ICONS.get(status, "[dim]·[/]")
            parts.append(f"{icon} {n}")
        return "  ".join(parts)

    # ── ETA ──────────────────────────────────────────────

    def _eta_str(self) -> str:
        if not self._chapter_times:
            return ""
        avg = sum(self._chapter_times) / len(self._chapter_times)
        remaining = self._project_total - (self._project_done + self._project_failed + self._project_skipped)
        if remaining <= 0:
            return ""
        eta_s = avg * remaining
        if eta_s < 60:
            return f"ETA ~{int(eta_s)}s"
        return f"ETA ~{int(eta_s / 60)}m"

    def _avg_str(self) -> str:
        if not self._chapter_times:
            return ""
        avg = sum(self._chapter_times) / len(self._chapter_times)
        return f"avg {avg:.0f}s/ch"

    # ── Hook: pipeline event dispatch ────────────────────

    def _ch_label(self, ch: float) -> str:
        return str(int(ch)) if ch == int(ch) else str(ch)

    def _set_ch_status(self, ch: float, status: str) -> None:
        for i, (num, _) in enumerate(self._ch_queue):
            if num == ch:
                self._ch_queue[i] = (num, status)
                return
        self._ch_queue.append((ch, status))

    def on(self, event: Event) -> None:  # noqa: C901
        match event:
            # ── Chapter lifecycle ─────────────────────────
            case ChapterStart(chapter=ch, pages=n):
                self._reset_chapter()
                self._chapter = ch
                self._pages_total = n
                self._phase = "preprocess"
                self._ch_t0 = time.monotonic()
                self._stream_line = ""
                self._set_ch_status(ch, "running")
                self._emit(f"\n [bold]ch{self._ch_label(ch)}[/]  {n} pages")

            case ChapterSkipped(chapter=ch, reason=r):
                self._set_ch_status(ch, "skipped")
                self._emit(f"  [dim]ch{self._ch_label(ch)} skipped ({r})[/]")

            case ChapterDone(chapter=ch, pages=p, bubbles=b, elapsed=s):
                self._set_ch_status(ch, "done")
                if ch == self._chapter:
                    self._phase = "done"
                    self._chapter_times.append(s)
                self._emit(f"  [green]✓[/] ch{self._ch_label(ch)}  {p}p {b}b  [dim]{s:.1f}s[/]")

            # ── Preprocess ────────────────────────────────
            case PageScanned(page=p, total=t, bubbles=b, det_ms=d, ocr_ms=o):
                self._pages_total = t
                self._bubbles += b
                self._scan_ms += d + o

            # ── Translate ─────────────────────────────────
            case TranslateStart(total_bubbles=n):
                self._phase = "translate"
                self._total_bubbles = n

            case LLMCall(agent=a, turn=t):
                self._flush_tools()
                self._llm_turns = t
                self._streaming_agent = a
                self._streaming_turn = t
                self._thinking_buf = ""
                self._text_buf = ""
                self._stream_line = f"[dim]{a}[/] t{t} [dim]…[/]"

            case LLMThinking(delta=d):
                self._thinking_buf += d
                preview = self._thinking_buf[-60:].replace("\n", " ")
                a, t = self._streaming_agent, self._streaming_turn
                self._stream_line = f"[dim]{a}[/] t{t} [dim italic]💭 {preview}[/]"

            case LLMText(delta=d):
                self._text_buf += d
                preview = self._text_buf[-60:].replace("\n", " ")
                a, t = self._streaming_agent, self._streaming_turn
                self._stream_line = f"[dim]{a}[/] t{t} [dim]{preview}[/]"

            case LLMResponse(turn=t, tool_calls=tc, ms=ms):
                self._llm_ms += ms
                s = ms / 1000
                label = f"{tc} tools" if tc else "done"
                a = self._streaming_agent
                self._stream_line = f"[dim]{a}[/] t{t}  [green]{s:.1f}s[/]  [dim]{label}[/]"

            case ToolResult(tool=tool, result=r):
                self._tool_queue.append((tool, r))

            case TranslateDone(translated=tr, total=n):
                self._flush_tools()
                self._translated = tr
                self._stream_line = ""

            case TranslationReady(translated=tr, total=n):
                self._phase = "render"
                self._render_total = self._pages_total
                self._stream_line = ""
                self._emit(f"  [green]⚡ {tr}/{n} translated[/]")

            # ── Render ────────────────────────────────────
            case PageErased(page=p, total=t, ms=ms):
                self._erase_ms += ms
                self._pages_rendered = p + 1
                self._render_total = t

            case PageRendered():
                pass

            # ── Knowledge ─────────────────────────────────
            case KnowledgeStart(pairs=n):
                self._phase = "knowledge"

            case KnowledgeDone():
                self._flush_tools()

            # ── System ────────────────────────────────────
            case ModelsUnloaded(stage=s):
                self._emit(f"  [dim]♻ unloaded {s}[/]")

            case SeriesProgress(done=d, failed=f, skipped=sk, total=t):
                self._project_done = d
                self._project_failed = f
                self._project_skipped = sk
                self._project_total = t

            case PipelineError(stage=s, error=e):
                self._set_ch_status(self._chapter, "failed")
                self._emit(f"  [red]✗[/] {s}: {e}")


# ── Load projects helper ─────────────────────────────────────────

async def load_projects() -> tuple[list[dict], dict[int, list[dict]]]:
    from .sqlite_store import SqliteStore
    from ..config import load_config

    _, paths = load_config()
    if not paths.db.exists():
        return [], {}

    store = await SqliteStore.open(paths.db)
    try:
        projects = await store.list_projects()
        chapters_map: dict[int, list[dict]] = {}
        for p in projects:
            chapters_map[p["id"]] = await store.get_chapters(p["id"])
        return projects, chapters_map
    finally:
        await store.close()
