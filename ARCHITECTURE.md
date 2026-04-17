# Typoon v2 — Architecture Design

## Overview

Manga/manhwa translation pipeline. Python core, 3 deployment modes.

```
                    ┌─────────────┐
                    │  Core       │  vision / translation / llm / render
                    │  Pipeline   │  (stateless compute)
                    └──────┬──────┘
                           │
                    ┌──────┴──────┐
                    │  AppService │  orchestration, workflows, state
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
         Local app    API Server    Platform
         CLI/Gradio   FastAPI       Crawler + auto-translate
         SQLite       Postgres      Postgres
         in-memory Q  Redis Q       Redis Q + scheduler
         local disk   S3/CDN        S3/CDN
```

---

## Principles

1. **AppService is the only entry point.** No UI calls Engine/Store/Scanner directly.
2. **Compute modules are stateless.** Det, erase, translate, render — pure functions.
3. **Scheduler owns concurrency.** Resource semaphores, not ad-hoc threading.
4. **Events are observation, not control.** Scheduler decides what runs. Events tell UI what happened.
5. **UI is replaceable.** Gradio now, native app later. Same AppService.

---

## Directory Structure

```
typoon/
├── app/
│   ├── service.py              # AppService — single entry point
│   ├── scheduler.py            # Resource-aware task scheduler
│   ├── events.py               # AppEvent, EventSink, CompositeSink
│   ├── workflows/
│   │   ├── scan.py             # scan_chapter workflow
│   │   ├── translate.py        # translate_chapter workflow
│   │   ├── render.py           # render_chapter workflow
│   │   └── project.py          # translate_project (multi-chapter)
│   └── state.py                # ChapterState enum, transitions
│
├── domain/
│   ├── project.py              # Project, Chapter dataclasses
│   └── bubble.py               # Bubble, Page dataclasses
│
├── vision/
│   ├── detect.py               # TextDetector (unchanged)
│   ├── scanner.py              # Scanner: det + OCR + merge (unchanged)
│   ├── erase.py                # Eraser: median + AOT (unchanged)
│   ├── merge.py                # group_lines (unchanged)
│   ├── chapter_images.py       # ChapterImages (unchanged)
│   ├── paginate.py             # smart_split (unchanged)
│   ├── tiling.py               # tile compute (unchanged)
│   ├── types.py                # TextMask, TextRegion (unchanged)
│   └── runtime/
│       ├── aot/                # AOT CoreML + ONNX
│       ├── ppocr_det/          # PP-OCR CoreML + MLX + ONNX
│       └── _pp_lcnet_v3.py     # backbone (MLX fallback)
│
├── translation/
│   ├── agent.py                # TranslatorAgent (unchanged)
│   ├── knowledge.py            # KnowledgeAgent (unchanged)
│   ├── context.py              # ContextAgent (unchanged)
│   ├── prompt.py               # prompt templates (unchanged)
│   ├── prompts/                # language policies (unchanged)
│   └── tools/                  # LLM tools (unchanged)
│
├── llm/
│   ├── ir.py                   # provider-agnostic IR (unchanged)
│   ├── agent.py                # generic agent loop (unchanged)
│   ├── openai.py               # (unchanged)
│   ├── anthropic.py            # (unchanged)
│   ├── gemini.py               # (unchanged)
│   └── tool_dec.py             # @tool decorator (unchanged)
│
├── storage/
│   └── sqlite.py               # SqliteStore (moved from adapters/)
│
├── interfaces/
│   ├── cli.py                  # thin CLI — calls AppService
│   ├── tui.py                  # TUI event sink (from adapters/tui.py)
│   ├── gradio_ui.py            # Gradio web UI
│   └── daemon.py               # stdio JSON-RPC daemon (phase 2)
│
├── config.py                   # Config, Paths (unchanged)
├── models.py                   # ModelHub (unchanged)
└── providers.py                # LLM provider factory (unchanged)
```

### What moves where

| Current | New | Notes |
|---|---|---|
| `cli.py` (476 LOC) | `interfaces/cli.py` (~50 LOC) + `app/workflows/` | Logic extracted to workflows |
| `runner.py` | **deleted** → `app/workflows/scan.py`, `translate.py`, `render.py` | Runner was artificial abstraction |
| `orchestrator.py` | **deleted** → `app/workflows/project.py` + `app/scheduler.py` | Split orchestration from scheduling |
| `engine.py` | **deleted** → `app/service.py` (wiring) + vision modules (compute) | Engine mixed lifecycle + compute + factory |
| `types.py` | `domain/project.py` + `domain/bubble.py` + `vision/types.py` | Split by concern |
| `ports.py` | **deleted** — protocols live with consumers | |
| `events.py` | `app/events.py` | AppEvent replaces Hook |
| `adapters/sqlite_store.py` | `storage/sqlite.py` | |
| `adapters/tui.py` | `interfaces/tui.py` | |
| `adapters/cli_hook.py` | `interfaces/tui.py` (merged) | |
| `adapters/comix.py` | `adapters/comix.py` (stays) | |
| `adapters/local_source.py` | `adapters/local_source.py` (stays) | |

### What stays unchanged

- `vision/` internals (detect, scanner, erase, merge, tiling, paginate)
- `translation/` (agents, tools, prompts)
- `llm/` (IR, providers, agent loop)
- `config.py`, `models.py`, `providers.py`
- `crates/render/` (Rust PyO3)
- All runtime backends (aot/, ppocr_det/)

---

## AppService

```python
class AppService:
    """Single entry point for all operations."""

    def __init__(self, config, paths, store, scheduler, models, sink, output):
        ...

    @classmethod
    async def create(cls, config_path=None) -> "AppService":
        """Bootstrap everything: config, DB, models, scheduler."""
        ...

    # ── Project management ──

    async def list_projects(self) -> list[dict]: ...
    async def get_project(self, project_id: int) -> dict: ...
    async def create_project(self, title, source_lang, target_lang, source_url=None) -> int: ...

    # ── Chapter operations ──

    async def scan_chapter(self, project_id, chapter_idx, source) -> ScanResult: ...
    async def translate_chapter(self, project_id, chapter_idx) -> TranslateResult: ...
    async def render_chapter(self, project_id, chapter_idx) -> RenderResult: ...

    # ── Batch operations ──

    async def translate_project(self, source, policy) -> ProjectResult: ...

    # ── Review/edit ──

    async def update_bubble(self, project_id, chapter_idx, bubble_id, text) -> None: ...
    async def rerender_page(self, project_id, chapter_idx, page_idx) -> None: ...

    # ── Events ──

    def subscribe(self, sink: EventSink) -> None: ...
```

---

## Scheduler

```python
# Resources
ACCELERATOR = "accelerator"   # CoreML det + erase (1 at a time)
LLM = "llm"                   # translation API calls
STORAGE = "storage"            # DB writes
CPU = "cpu"                    # OCR, image ops

# Limits
_LIMITS = {
    ACCELERATOR: 1,
    LLM: 2,
    STORAGE: 1,
    CPU: 4,
}

@dataclass
class TaskSpec:
    id: str
    requires: set[str]
    coro: Callable[[], Awaitable[Any]]

class Scheduler:
    def __init__(self):
        self._sems = {k: asyncio.Semaphore(v) for k, v in _LIMITS.items()}
        self._tasks: dict[str, asyncio.Task] = {}

    async def submit(self, spec: TaskSpec) -> str:
        """Submit task. Runs when required resources available."""
        ...

    async def cancel(self, task_id: str) -> None: ...
    def active_tasks(self) -> list[str]: ...
```

### What can run in parallel

```
scan_page (accelerator)  ║  translate_batch (llm)     → YES
erase_page (accelerator) ║  translate_batch (llm)     → YES
scan_page (accelerator)  ║  erase_page (accelerator)  → NO (same resource)
translate (llm)          ║  save_result (storage)      → YES
ocr_batch (cpu)          ║  translate (llm)            → YES
download (cpu)           ║  scan_page (accelerator)    → YES
```

---

## Events

```python
@dataclass(frozen=True)
class AppEvent:
    type: str
    job_id: str | None = None
    data: dict = field(default_factory=dict)

# Concrete event types (just strings, not classes — keep simple)
# "job_started", "page_scanned", "ocr_done", "translation_progress",
# "page_erased", "page_rendered", "chapter_done", "job_failed"

class EventSink(Protocol):
    def emit(self, event: AppEvent) -> None: ...

class CompositeSink:
    def __init__(self, *sinks: EventSink):
        self._sinks = sinks
    def emit(self, event: AppEvent) -> None:
        for s in self._sinks:
            s.emit(event)
```

### Sinks

```python
class LogSink:       # print to console
class TuiSink:       # Rich TUI rendering
class GradioSink:    # push to Gradio queue
class FileSink:      # append JSONL log
class DaemonSink:    # write to stdout JSON-RPC notification (phase 2)
```

---

## Workflows

Each workflow is a plain async function. No class hierarchy.

### scan.py

```python
async def scan_chapter(
    scanner, source, store, project_id, chapter_idx, emit,
) -> tuple[list[Page], ChapterImages]:
    await source.fetch()
    pages = []
    for i in range(source.page_count()):
        img = source.load_page(i)
        scanned = scanner.scan(img)
        page = Page(index=i, bubbles=...)
        pages.append(page)
        emit(AppEvent("page_scanned", data={"page": i, "total": source.page_count()}))
    ...
    return pages, images
```

### translate.py

```python
async def translate_chapter(
    pages, session, emit,
) -> tuple[int, Exception | None]:
    emit(AppEvent("translate_start", data={"total": total_bubbles}))
    turns, error = await translate_pages(pages, session)
    emit(AppEvent("translate_done", data={"translated": n, "total": total}))
    return turns, error
```

### render.py

```python
async def render_chapter(
    pages, images, eraser, emit,
) -> None:
    for page in pages:
        masks = [m for b in page.bubbles for m in b.masks]
        eraser.erase(canvas, masks)
        # render text via Rust
        ...
        emit(AppEvent("page_rendered", data={"page": page.index}))
```

### project.py

```python
async def translate_project(
    service, source, policy, emit,
) -> ProjectResult:
    # resolve source (URL or path)
    # for each chapter:
    #   scheduler.submit(scan, requires={"accelerator"})
    #   scheduler.submit(translate, requires={"llm"})
    #   scheduler.submit(render, requires={"accelerator"})
    ...
```

---

## Chapter State

```python
class ChapterState(StrEnum):
    NEW = "new"
    SCANNING = "scanning"
    SCANNED = "scanned"
    TRANSLATING = "translating"
    TRANSLATED = "translated"
    RENDERING = "rendering"
    RENDERED = "rendered"
    FAILED = "failed"
```

Transitions:
```
new → scanning → scanned → translating → translated → rendering → rendered
                                ↑                          ↑
                          (re-translate)              (re-render after edit)
any → failed
```

Stored in SQLite `chapters.status`.

---

## Interfaces

### CLI (interfaces/cli.py)

```python
@app.command()
def translate(source: str, ...):
    service = asyncio.run(AppService.create())
    service.subscribe(TuiSink())
    asyncio.run(service.translate_project(source, policy))
```

~50 LOC total for all commands.

### Gradio UI (interfaces/gradio_ui.py)

```python
def build_ui(service: AppService) -> gr.Blocks:
    with gr.Blocks() as demo:
        with gr.Tab("Translate"):
            source = gr.Textbox(label="URL or folder path")
            btn = gr.Button("Start")
            progress = gr.Textbox(label="Progress", interactive=False)
            gallery = gr.Gallery(label="Output")
            btn.click(fn=run_translate, inputs=[source], outputs=[progress, gallery])
        with gr.Tab("Projects"):
            ...
        with gr.Tab("Review"):
            ...
    return demo
```

### Daemon (interfaces/daemon.py) — phase 2

```python
async def main():
    service = await AppService.create()
    # Read JSON-RPC from stdin, dispatch to service, write results to stdout
    async for request in read_jsonrpc(sys.stdin):
        result = await dispatch(service, request)
        write_jsonrpc(sys.stdout, result)
```

---

## Runtime Stack (Mac)

```
Detection:   CoreML .mlpackage (RangeDim, ~15-58ms)
             Fallback: MLX compiled → ONNX CPU
Inpainting:  CoreML .mlpackage (EnumeratedShapes, 3-37ms)
             Fallback: ONNX CPU/CUDA
OCR:         Apple Vision (parallel ThreadPool)
             Fallback: Windows OCR → Tesseract
Rendering:   Rust PyO3 crate
LLM:         OpenAI / Anthropic / Gemini (streaming)
Storage:     SQLite (aiosqlite)
```

Models on HuggingFace `nghyane/typoon`:
```
aot-inpaint.mlpackage    11MB   (CoreML)
aot-inpaint.onnx         22MB   (cross-platform)
ppocr-det.mlpackage       7MB   (CoreML)
ppocr-det.onnx           16MB   (cross-platform)
ppocr-det-config.json
```

Total: ~56MB (was 260MB).

---

## Migration Plan

### Step 1 — Create app/ layer (2 days)

- `app/events.py` — AppEvent, EventSink, CompositeSink
- `app/scheduler.py` — resource semaphores
- `app/service.py` — AppService wrapping current logic
- `app/workflows/` — extract from runner.py + orchestrator.py
- Tests pass, CLI still works via AppService

### Step 2 — Thin interfaces (1 day)

- `interfaces/cli.py` — rewrite CLI to call AppService
- `interfaces/tui.py` — extract TUI sink from adapters/
- Delete old `cli.py`, `runner.py`, `orchestrator.py`, `engine.py`

### Step 3 — Gradio UI (1 day)

- `interfaces/gradio_ui.py`
- Translate tab, Projects tab, basic Review tab
- `typoon ui` command launches Gradio

### Step 4 — Domain cleanup (0.5 day)

- `domain/project.py`, `domain/bubble.py`
- Move types from `types.py`
- Delete `ports.py`, `adapters/` (move contents)

### Step 5 — Daemon (phase 2, when needed)

- `interfaces/daemon.py` — stdio JSON-RPC
- Native app shell (Swift/Tauri)

Total: ~4.5 days for steps 1-4.

---

## What We Don't Do Now

- No event bus (EventSink callback is enough)
- No microservices
- No plugin system
- No abstract factory
- No DDD aggregates

---

## Backend Protocols (swap per deployment mode)

### JobQueue

```python
class JobQueue(Protocol):
    async def submit(self, name: str, params: dict) -> str: ...
    async def status(self, job_id: str) -> JobStatus: ...
    async def cancel(self, job_id: str) -> None: ...
    async def subscribe(self, callback: Callable[[JobEvent], None]) -> None: ...
```

- `MemoryQueue` — local app (asyncio, no deps)
- `RedisQueue` — server / platform (arq, persistent)

### Store

```python
class Store(Protocol):
    async def get_project(self, project_id: int) -> dict | None: ...
    async def save_translations(self, ...) -> None: ...
    # ... same interface as current SqliteStore
```

- `SqliteStore` — local app
- `PostgresStore` — server / platform

### OutputWriter

```python
class OutputWriter(Protocol):
    async def save_pages(self, project_id: int, chapter: float, pages: list) -> str: ...
```

- `LocalWriter` — save to disk, return path
- `S3Writer` — upload S3, return CDN URL

### AppService receives all via constructor

```python
class AppService:
    def __init__(self, store: Store, queue: JobQueue, output: OutputWriter, ...):
        ...
```

Same AppService, different backends per deployment.

- No event bus (EventSink callback is enough)
- No UnitOfWork / Repository pattern (SqliteStore is fine)
- No daemon process yet (Gradio runs in-process)
- No microservices
- No plugin system
- No abstract factory
- No DDD aggregates

---

## Success Criteria

1. `typoon translate <path>` works via AppService (not direct engine calls)
2. `typoon ui` launches Gradio with translate + review
3. Scan + translate can overlap (different resources)
4. Adding new UI = implement EventSink + call AppService methods
5. `cli.py` < 60 LOC
6. No module imports `interfaces/`
