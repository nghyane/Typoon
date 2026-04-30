# Typoon v2 — Roadmap

Manga/manhwa translation pipeline. Python + Rust (PyO3 render).
Local-first cho CLI/Desktop, scalable cho Platform.

---

## Architecture Overview

```
Interface Layer          CLI / Web UI / Platform
                              │
                         ResumePolicy (per interface)
                              │
Orchestration            Orchestrator
                         status lifecycle, skip/resume, error isolation
                              │
Pipeline                 ChapterRunner
                         scan_and_translate / scan_and_load / render / consolidate
                              │
Core (stateless)         Engine (vision)  │  Translation (LLM agents)
                         scan/erase/render │  agent/knowledge/context
                              │                     │
                         Vision              LLM Providers
                         PP-OCR/LaMa/MLX     OpenAI/Anthropic/Gemini (streaming)
```

### Nguyên tắc

- **Hexagonal**: Core không biết interface. Adapters implement protocols.
- **Project-centric**: Project do user tạo, chapters từ bất kỳ nguồn nào.
- **Pipeline chỉ đọc local**: Connector download → local folder → Pipeline xử lý.
- **Acquisition tách Processing**: Download ≠ Translate. Hai concern riêng.
- **ResumePolicy**: Config flags, không phải strategy classes.

---

## Phase 1 — Core Pipeline ✅

Đã hoàn thành. 69 tests passing.

### Engine (vision compute)
- [x] PP-OCR detection (MLX + PyTorch backends)
- [x] PP-OCR recognition (batch)
- [x] Line → bubble merging (spatial proximity)
- [x] LaMa inpainting (MLX + PyTorch)
- [x] Text rendering (Rust PyO3 crate)
- [x] Model lifecycle (lazy load, unload, reload)

### LLM Infrastructure
- [x] Provider-agnostic IR (Message, ContentPart, ToolCallMsg, ToolDef)
- [x] Streaming architecture (StreamEvent, async generator)
- [x] 3 providers: OpenAI (streaming), Anthropic (streaming), Gemini (streaming)
- [x] Generic agent loop (stream-first, call() fallback)
- [x] Thinking token support (o-series, Claude thinking)

### Translation Agents
- [x] Translation agent (tool-calling: translate, view_page, search_glossary, get_context)
- [x] Knowledge agent (update_snapshot, add_note, upsert_glossary)
- [x] Context agent (search translations + notes from DB)
- [x] Prompt system with language-specific policies

### Events & Hooks
- [x] Typed events (PageScanned, LLMCall, LLMThinking, LLMText, ChapterDone, etc.)
- [x] Hook pattern (observer)
- [x] RichHook (fullscreen TUI with streaming LLM preview)
- [x] CompositeHook (fan-out to multiple hooks)

---

## Phase 2 — Orchestration ✅

### Runner + Orchestrator
- [x] ChapterRunner — stateless pipeline phases
- [x] Orchestrator — series-level lifecycle management
- [x] ResumePolicy (force, resume_translated, retry_failed, max_retries)
- [x] Chapter lifecycle: pending → translating → translated → rendering → done | failed
- [x] Review pause point at "translated" (desktop app preparation)
- [x] Error isolation (1 chapter fail → continue next)
- [x] retry_count tracking in DB

### Store
- [x] SqliteStore adapter
- [x] Series CRUD + glossary + knowledge snapshots + notes
- [x] FTS5 search (translations, glossary, notes)
- [x] Chapter status + retry_count
- [x] get_series_by_title, get_series_by_url

### CLI
- [x] `typoon translate <path>` — auto-detect series/chapter
- [x] --from/--to, --force, --source-lang, --target-lang
- [x] Series identity by folder name (auto-create)
- [x] Partial translation threshold (≥80% = done with warning)

---

## Phase 3 — Redesign: Project-Centric 🔲

### Tại sao redesign

Hiện tại series gắn chặt với folder name hoặc URL. Thực tế:
- User cần tạo project với tên chính xác, không auto-parse từ URL.
- Chapters có thể từ nhiều nguồn (connector, local, screenshot).
- Chapter index do project quyết định, không phụ thuộc source numbering.
- Download cần cơ chế phức tạp (cookies, browser, CF bypass) — không abstract được bằng `http_headers()`.

### 3.1 Data Model

```
Project (thay thế Series)
├── id, title, source_lang, target_lang
├── source_url (nullable — for auto-crawl lookup)
├── preferred_group (nullable — scanlation group preference)
└── created_at

Chapter (thuộc Project)
├── id, project_id
├── idx (REAL) — canonical ordering do project quyết định
├── local_path — folder chứa images trên disk
├── source_url (nullable — where it came from)
├── source_name (nullable — display name từ source)
├── status: pending → translating → translated → rendering → done | failed
├── retry_count
└── created_at
```

`idx` là thứ tự thật. Knowledge chain dùng `idx` để order.
`source_url` / `source_name` chỉ là metadata, không ảnh hưởng pipeline.

### 3.2 Connector Protocol

Connector chỉ làm 2 việc: **discover** và **get page URLs**.
Download là shared utility, không phải concern của connector.

```python
class Connector(Protocol):
    site_name: str
    def accepts(url: str) -> bool
    async def discover(url: str) -> SourceInfo
    async def get_page_urls(chapter, variant?) -> list[str]

# Auth/cookies/browser là internal concern
# Mỗi connector tự quản lý, không expose ra protocol

@dataclass
class SourceInfo:
    suggested_title: str
    suggested_lang: str
    cover_url: str | None
    chapters: list[DiscoveredChapter]

@dataclass
class DiscoveredChapter:
    number: float
    title: str | None
    variants: list[ChapterVariant]
```

### 3.3 Shared Downloader

```python
async def download_images(urls, dest, headers, concurrency=5, max_retries=3) -> Path
```

Connector cung cấp URLs + headers. Downloader handle retry, skip existing, concurrency.
Output = folder ảnh trên disk. Pipeline đọc bằng LocalSource.

### 3.4 CLI UX — 1 command

```bash
# URL → discover → create project → download → translate
typoon translate https://comix.to/title/z0yj-jujutsu-kaisen

# Local folder → create project → translate
typoon translate /path/to/jujutsu-kaisen/

# Resume (cùng lệnh, auto-detect existing project)
typoon translate https://comix.to/title/z0yj-jujutsu-kaisen
# → skip done, download new, translate new

# Power user: project management
typoon projects                        # list
typoon project info "JJK"              # status
typoon project add "JJK" /path/ch051/  # thêm chapter local
typoon project glossary "JJK"          # view/edit
```

Smart resolution:
- URL → connector discover → find/create project by URL
- Path → scan folder → find/create project by title (folder name)
- Lần đầu: hỏi confirm. Lần sau: auto-resume.

### 3.5 Chapter Index Mapping

```
Từ connector:  source ch1 → idx 1.0 (auto)
               source ch2.5 → idx 2.5 (auto)
Từ local:      folder "ch024.3" → idx 24.3 (parse)
Manual:        user chỉ định index
Conflict:      "Chapter 24 exists. Replace? [y/N]"
```

Auto-crawl: `idx = source_number` (không cần user confirm).
Manual: user map index khi add chapter.

### 3.6 Tasks

- [ ] Rename `series` → `project` trong DB schema, Store, types
- [ ] Thêm `idx`, `local_path` vào chapters table
- [ ] Update Store protocol (project-centric methods)
- [ ] Update SqliteStore adapter
- [ ] Refactor Connector protocol (discover + get_page_urls)
- [ ] Update ComixConnector
- [ ] Update CLI (smart URL/path resolution)
- [ ] Update Orchestrator (dùng project + chapter idx)
- [ ] Update Runner (dùng local_path)
- [ ] Update conftest.py (MockStore mới)
- [ ] Migration script cho existing DB

---

## Phase 4 — Web UI 🔲

### 4.1 API Server (FastAPI + SSE)

```
POST /translate          start translation job (URL or path)
GET  /events/{job_id}    SSE stream of pipeline events
GET  /projects           list projects
GET  /projects/{id}      project detail + chapters
PUT  /projects/{id}      update settings
GET  /projects/{id}/glossary
POST /projects/{id}/chapters  add chapter (local path or URL)
```

### 4.2 SSE Hook

```python
class SseHook(Hook):
    """Hook → Server-Sent Events for Web UI."""
    def on(self, event):
        for connection in self._connections:
            connection.send(event_to_json(event))
```

Map 1:1 với event system hiện có. Web UI subscribe SSE stream.

### 4.3 Frontend

- Paste URL hoặc chọn folder → preview → 1 click translate
- Chapter list với status (done/failed/pending)
- Realtime progress (SSE)
- Glossary viewer/editor
- Side-by-side original ↔ translated (future)

### 4.4 Background Jobs

Translation chạy trong `asyncio.create_task`. API trả `job_id` ngay.
Frontend subscribe `GET /events/{job_id}` cho progress.

### 4.5 Tasks

- [ ] FastAPI app + routes
- [ ] SseHook adapter
- [ ] Job manager (background tasks)
- [ ] Static file serving (rendered pages)
- [ ] Frontend (HTML/JS hoặc Svelte)

---

## Phase 5 — Platform Layer 🔲

Platform = content site tự động. Thêm layer **Library Management** phía trên Orchestrator.

### 5.1 Architecture

```
Platform Dashboard
    │
Library Manager              ← Phase 5 mới
    │ follow/unfollow manga
    │ monitor new chapters
    │ auto-trigger pipeline
    │
Orchestrator                 ← Phase 2 (đã có)
    │
ChapterRunner → Engine       ← Phase 1 (đã có)
```

### 5.2 Library Manager

```python
class LibraryManager:
    """Manages followed manga on the platform."""

    async def follow(self, url: str) -> Project:
        """Add manga to library. Discover + create project."""

    async def unfollow(self, project_id: int) -> None:
        """Remove from library."""

    async def check_updates(self) -> list[UpdateInfo]:
        """Poll all followed projects for new chapters."""

    async def process_updates(self, updates: list[UpdateInfo]) -> None:
        """Download + translate + publish new chapters."""
```

### 5.3 Scheduler

```python
# Chạy định kỳ (config: interval)
async def scheduler_tick(library: LibraryManager):
    updates = await library.check_updates()
    if updates:
        await library.process_updates(updates)
```

Config:
```toml
[platform]
check_interval = "1h"
policy = "platform"          # ResumePolicy(max_retries=3)

[platform.publish]
type = "s3"
bucket = "manga-translated"
```

### 5.4 Platform Dashboard

```
┌─ Sources ─────────────────────────────┐
│ comix.to     ● online                 │
│ mangadex     ● online                 │
└───────────────────────────────────────┘

┌─ Library ─────────────────────────────┐
│ Jujutsu Kaisen   107/107  ✓ up to date│
│ One Piece        1100/1100  ✓         │
│ Solo Leveling    180/183  ⟳ 3 new     │
│                                       │
│ Total: 3 series, 1390 chapters done   │
│ Queued: 3 chapters                    │
│ Failed: 0                             │
└───────────────────────────────────────┘
```

### 5.5 Source Selection Logic

Platform chọn manga, không auto-crawl mù:
1. Admin add URL → `library.follow(url)`
2. Library tạo project từ connector discover
3. Scheduler monitor chỉ các project đang follow
4. New chapters → auto: download → translate → publish
5. Admin có thể unfollow → stop monitoring

### 5.6 Publisher Adapters

```python
class Publisher(Protocol):
    async def publish(self, project_id, chapter_idx, pages) -> str:
        """Publish rendered pages. Returns public URL."""

class S3Publisher:        # upload to S3/R2
class LocalPublisher:     # save to disk (dev)
class CDNPublisher:       # upload + invalidate cache
```

`on_chapter` callback dùng Publisher:
```python
def on_chapter(ch, pages):
    publisher.publish(project_id, ch, pages)
```

### 5.7 Tasks

- [ ] LibraryManager
- [ ] Scheduler (interval-based polling)
- [ ] Publisher protocol + S3Publisher
- [ ] Platform API endpoints (follow/unfollow, dashboard)
- [ ] Webhook notifications (Discord on new chapter published)
- [ ] PostgresStore adapter (production DB)

---

## Phase 6 — Desktop App 🔲

### 6.1 Architecture

Desktop = Web UI chạy local. Cùng codebase, khác config:
- SQLite (không Postgres)
- LocalPublisher (save to disk)
- DesktopPolicy (pause at "translated" for review)
- User's own API keys (BYOK)

### 6.2 Review Flow

```
translate → save translations to DB → status "translated"
    ↓ (pause — user opens desktop app)
user reviews side-by-side: original ↔ translated
user edits individual bubbles in UI
    ↓ (user clicks "Render")
load translations from DB → render → status "done"
```

Đã chuẩn bị: `runner.scan_and_load()` + `ResumePolicy(resume_translated=False)`.

### 6.3 Tasks

- [ ] Desktop-specific config (BYOK, local paths)
- [ ] Review UI (side-by-side original ↔ translated per bubble)
- [ ] Bubble editor (edit translated_text in DB)
- [ ] Re-render single page (after edit)
- [ ] Package as standalone app (PyInstaller hoặc Python server + browser)

---

## Phase 7 — Quality & Polish 🔲

- [ ] Cost tracking (token count per chapter, budget limits)
- [ ] Rate limiting + backoff (LLM providers + manga sources)
- [ ] Export formats (CBZ, PDF)
- [ ] Glossary sharing across projects (same universe)
- [ ] SFX detection + handling (sound effects vs dialogue)
- [ ] Cache cleanup (eviction policy for downloaded images)
- [ ] PyTorch backends for Windows/Linux server (PP-OCR, LaMa)

---

## Current File Structure

```
v2/typoon/
├── engine.py          # Vision compute (scan, erase, render)
├── runner.py          # ChapterRunner (pipeline phases)
├── orchestrator.py    # Orchestrator + ResumePolicy
├── config.py          # Config data (no factory)
├── providers.py       # LLM provider factory
├── downloader.py      # Shared image downloader
├── types.py           # Value types (Bubble, Page, Session, SeriesInfo, ...)
├── ports.py           # Protocols (ChapterSource, Connector, Store)
├── events.py          # Typed events + Hook
├── models.py          # HuggingFace model hub
├── cli.py             # CLI entry point
├── adapters/
│   ├── local_source.py    # LocalSource (folder → images)
│   ├── sqlite_store.py    # SQLite adapter
│   ├── comix.py           # comix.to connector
│   ├── cli_hook.py        # Rich TUI hook
│   └── webhook_hook.py    # Discord webhook hook
├── llm/
│   ├── ir.py              # IR types + StreamEvent + Provider protocol
│   ├── agent.py           # Generic agent loop (streaming)
│   ├── openai.py          # OpenAI streaming provider
│   ├── anthropic.py       # Anthropic streaming provider
│   ├── gemini.py          # Gemini streaming provider
│   └── tool_dec.py        # @tool decorator
├── translation/
│   ├── agent.py           # Translation agent
│   ├── knowledge.py       # Knowledge consolidation agent
│   ├── context.py         # Context retrieval agent
│   ├── prompt.py          # Prompt templates
│   └── tools/             # Tool definitions
└── vision/
    ├── detect.py          # PP-OCR detection
    ├── ocr.py             # PP-OCR recognition
    ├── erase.py           # LaMa inpainting
    ├── merge.py           # Line → bubble merging
    └── runtime/           # MLX + PyTorch backends
```

## Test Structure

```
v2/tests/
├── conftest.py            # Shared fixtures (MockProvider, MockStore, make_session)
├── test_llm_ir.py         # IR types
├── test_llm_providers.py  # Provider serialization
├── test_llm_agent.py      # Agent loop
├── test_translation.py    # Translation agent
├── test_knowledge.py      # Knowledge agent
├── test_detect.py         # PP-OCR detection
├── test_ocr.py            # OCR recognition
├── test_erase.py          # Inpainting
└── test_merge.py          # Line merging
```

69 tests. Run: `v2/.venv/bin/python -m pytest v2/tests/ -q`
