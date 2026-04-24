"""CLI hook — fullscreen TUI with Rich Live + raw key input."""

from __future__ import annotations

import sys
import termios
import threading
import time
import tty
from collections import deque
from pathlib import Path

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from typoon.app.events import (
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

# ── Fullscreen TUI ───────────────────────────────────────────────


class RichHook(Hook):
    def __init__(self, log_file: Path | None = None) -> None:
        self._log: deque[Text] = deque(maxlen=50)
        self._full_log: list[str] = []
        self._log_file = log_file
        self._tool_queue: list[tuple[str, str]] = []
        self._t0 = time.monotonic()

        # Project-level
        self._project_done = 0
        self._project_failed = 0
        self._project_skipped = 0
        self._project_total = 0
        self._chapter_times: list[float] = []

        # Chapter-level
        self._chapter: float = 0
        self._phase = "init"
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

        # Streaming state
        self._streaming_agent: str = ""
        self._streaming_turn: int = 0
        self._thinking_buf: str = ""
        self._text_buf: str = ""
        self._stream_line: str = ""

        # Interactive state
        self.quit_requested = False
        self.paused = False

        self._live: Live | None = None
        self._key_thread: threading.Thread | None = None
        self._old_termios = None

    def start(self) -> None:
        self._live = Live(self._build(), console=console, screen=True, refresh_per_second=8)
        self._live.start()
        self._start_key_reader()

    def stop(self) -> None:
        self._stop_key_reader()
        if self._live:
            self._live.stop()
            self._live = None
        if self._log_file and self._full_log:
            self._log_file.parent.mkdir(parents=True, exist_ok=True)
            self._log_file.write_text("\n".join(self._full_log) + "\n")

    # ── Key reader ───────────────────────────────────────

    def _start_key_reader(self) -> None:
        fd = sys.stdin.fileno()
        self._old_termios = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        self._key_thread = threading.Thread(target=self._read_keys, daemon=True)
        self._key_thread.start()

    def _stop_key_reader(self) -> None:
        if self._old_termios is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_termios)
            self._old_termios = None

    def _read_keys(self) -> None:
        while not self.quit_requested:
            try:
                ch = sys.stdin.read(1)
            except (OSError, ValueError):
                break
            if ch in ("q", "Q"):
                self.quit_requested = True
                self._emit("[yellow]⏹ quitting after current chapter…[/]")
            elif ch in ("p", "P"):
                self.paused = not self.paused
                if self.paused:
                    self._emit("[yellow]⏸ paused — press p to resume[/]")
                else:
                    self._emit("[green]▶ resumed[/]")

    def log(self, msg: str) -> None:
        self._emit(msg)

    def _emit(self, line: str | Text) -> None:
        if isinstance(line, str):
            line = Text.from_markup(line)
        self._log.append(line)
        self._full_log.append(line.plain)
        self._refresh()

    def _refresh(self) -> None:
        if self._live:
            self._live.update(self._build())

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
        self._emit(f"  [dim]→ {parts}[/]")
        self._tool_queue.clear()

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

    # ── Layout ───────────────────────────────────────────

    def _build(self) -> Panel:
        parts: list[Text | str] = []

        # Header
        elapsed = time.monotonic() - self._t0
        mins, secs = divmod(int(elapsed), 60)
        t_str = f"{mins}m{secs:02d}s" if mins else f"{secs}s"
        header = f" [bold #58a6ff]typoon[/]  ch{self._ch_label(self._chapter)}"
        if self._project_total > 0:
            progress = self._project_done + self._project_failed + self._project_skipped
            header += f"  {progress}/{self._project_total}"
        eta = self._eta_str()
        if eta:
            header += f"  [cyan]{eta}[/]"
        header += f"  [dim]{t_str}[/]"
        parts.append(Text.from_markup(header))
        parts.append(Text(""))

        # 3-phase status
        parts.append(Text.from_markup(self._phase_line_preprocess()))
        parts.append(Text.from_markup(self._phase_line_translate()))
        parts.append(Text.from_markup(self._phase_line_render()))
        parts.append(Text(""))

        # LLM streaming
        if self._stream_line:
            parts.append(Text.from_markup(f" {self._stream_line}"))
        buf = self._thinking_buf or self._text_buf
        if buf:
            icon = "[dim italic]💭[/] " if self._thinking_buf else "   "
            stream_lines = buf.replace("\r", "").split("\n")
            for sl in stream_lines[-4:]:
                parts.append(Text.from_markup(f" {icon}[dim]{sl[:90]}[/]"))

        # Tool queue
        if self._tool_queue:
            groups: list[tuple[str, int]] = []
            for tool, _ in self._tool_queue:
                if groups and groups[-1][0] == tool:
                    groups[-1] = (tool, groups[-1][1] + 1)
                else:
                    groups.append((tool, 1))
            tparts = ", ".join(f"{t} ×{c}" if c > 1 else t for t, c in groups)
            parts.append(Text.from_markup(f" [dim]→ {tparts}[/]"))

        # Log
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

    # ── Event dispatch ───────────────────────────────────

    @staticmethod
    def _ch_label(ch: float) -> str:
        return str(int(ch)) if ch == int(ch) else str(ch)

    def on(self, event: Event) -> None:
        match event:
            case ChapterStart(chapter=ch, pages=n):
                self._reset_chapter()
                self._chapter = ch
                self._pages_total = n
                self._phase = "preprocess"
                self._stream_line = ""
                self._emit(f"\n [bold]ch{self._ch_label(ch)}[/]  {n} pages")

            case ChapterSkipped(chapter=ch, reason=r):
                self._emit(f"  [dim]ch{self._ch_label(ch)} skipped ({r})[/]")

            case ChapterDone(chapter=ch, pages=p, bubbles=b, elapsed=s):
                if ch == self._chapter:
                    self._phase = "done"
                    self._chapter_times.append(s)
                self._emit(f"  [green]✓[/] ch{self._ch_label(ch)}  {p}p {b}b  [dim]{s:.1f}s[/]")

            case PageScanned(page=p, total=t, bubbles=b, det_ms=d, ocr_ms=o):
                self._pages_total = t
                self._bubbles += b
                self._scan_ms += d + o

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
                self._refresh()

            case TranslateDone(translated=tr, total=n):
                self._flush_tools()
                self._translated = tr
                self._stream_line = ""

            case TranslationReady(translated=tr, total=n):
                self._phase = "render"
                self._render_total = self._pages_total
                self._emit(f"  [green]⚡ {tr}/{n} translated[/]")

            case PageErased(page=p, total=t, ms=ms):
                self._erase_ms += ms
                self._pages_rendered = p + 1
                self._render_total = t

            case PageRendered():
                pass

            case KnowledgeStart(pairs=n):
                self._phase = "knowledge"

            case KnowledgeDone():
                self._flush_tools()

            case ModelsUnloaded(stage=s):
                self._emit(f"  [dim]♻ unloaded {s}[/]")

            case SeriesProgress(done=d, failed=f, skipped=sk, total=t):
                self._project_done = d
                self._project_failed = f
                self._project_skipped = sk
                self._project_total = t
                self._refresh()

            case PipelineError(stage=s, error=e):
                self._emit(f"  [red]✗[/] {s}: {e}")
