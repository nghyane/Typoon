# RFC: Pipeline Queue & Web API

**Status**: Draft  
**Scope**: Storage redesign, worker queue, REST API  

---

## 1. Problem Statement

Pipeline hiện tại có ba vấn đề cốt lõi:

**1. Sequential execution.** Một chapter phải chạy xong toàn bộ prepare → scan →
translate → render trước khi chapter tiếp theo bắt đầu. GPU idle trong khi
translate. LLM idle trong khi scan. Lãng phí toàn bộ với full manga.

**2. Mixed concerns trong storage.** `chapters.status` vừa là domain state vừa
là process state. Filesystem JSON files vừa là artifact vừa là state marker. Hai
source of truth cho cùng một fact — inconsistent sau crash.

**3. Không thể tách API/worker.** `ProjectService` nhập nhằng giữa orchestration,
pipeline, và query. Không có ranh giới rõ để expose API mà không kéo toàn bộ
pipeline vào web process.

---

## 2. Principles

**P1. Một fact — một chỗ lưu.**  
Nếu fact có thể derive từ data đã có, không lưu lại.

**P2. Mỗi tầng chứa đúng loại data của nó.**

```
DB          → structured, queryable, relational, searchable
Filesystem  → binary, opaque, không query được (images, geometry, masks)
Derived     → computed on-the-fly từ hai tầng trên, không stored
```

**P3. DB là queue cho coordination, không phải cho domain state.**  
`tasks` table tồn tại để workers coordinate. Xóa khi xong.

**P4. Worker tách khỏi API.**  
API process chỉ đọc/ghi DB. Worker process chỉ poll DB và xử lý.
Không có shared in-memory state.

---

## 3. Storage Design

### 3.1 Database — structured + queryable data only

```sql
-- ── Identity ──────────────────────────────────────────────────────

CREATE TABLE projects (
    id           INTEGER PRIMARY KEY,
    slug         TEXT NOT NULL UNIQUE,
    title        TEXT NOT NULL,
    source_url   TEXT,
    source_lang  TEXT NOT NULL,
    target_lang  TEXT NOT NULL,
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE chapters (
    id           INTEGER PRIMARY KEY,
    project_id   INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    idx          REAL NOT NULL,
    source_url   TEXT,
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(project_id, idx)
);

-- ── Worker coordination ───────────────────────────────────────────
-- Transient: row tồn tại = work pending/running. DELETE khi xong.

CREATE TABLE tasks (
    chapter_id   INTEGER NOT NULL REFERENCES chapters(id) ON DELETE CASCADE,
    stage        TEXT NOT NULL CHECK(stage IN ('prepare','scan','translate','render')),
    claimed_by   TEXT,                      -- worker UUID, NULL = free
    claimed_at   TEXT,                      -- stale lock detection
    attempts     INTEGER NOT NULL DEFAULT 0,
    last_error   TEXT,
    PRIMARY KEY (chapter_id, stage)
);

CREATE INDEX idx_tasks_claim ON tasks(stage, claimed_by, claimed_at);

-- ── Knowledge — OCR text ──────────────────────────────────────────
-- Lưu text để search/FTS. Geometry lưu ở filesystem.

CREATE TABLE bubbles (
    chapter_id   INTEGER NOT NULL REFERENCES chapters(id) ON DELETE CASCADE,
    page_index   INTEGER NOT NULL,
    bubble_idx   INTEGER NOT NULL,
    source_text  TEXT NOT NULL,
    confidence   REAL NOT NULL,
    PRIMARY KEY (chapter_id, page_index, bubble_idx)
);

-- ── Knowledge — Translations ──────────────────────────────────────

CREATE TABLE translations (
    chapter_id      INTEGER NOT NULL REFERENCES chapters(id) ON DELETE CASCADE,
    page_index      INTEGER NOT NULL,
    bubble_idx      INTEGER NOT NULL,
    translated_text TEXT NOT NULL,
    kind            TEXT NOT NULL CHECK(kind IN ('dialogue','sfx','skip')),
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (chapter_id, page_index, bubble_idx),
    FOREIGN KEY (chapter_id, page_index, bubble_idx)
        REFERENCES bubbles(chapter_id, page_index, bubble_idx) ON DELETE CASCADE
);

-- ── Knowledge — Context ───────────────────────────────────────────

CREATE TABLE chapter_briefs (
    chapter_id   INTEGER PRIMARY KEY REFERENCES chapters(id) ON DELETE CASCADE,
    brief_json   TEXT NOT NULL,
    summary      TEXT,
    terms_text   TEXT,
    facts_text   TEXT,
    rules_text   TEXT,
    updated_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE glossary (
    id           INTEGER PRIMARY KEY,
    project_id   INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    source_term  TEXT NOT NULL,
    target_term  TEXT NOT NULL,
    notes        TEXT,
    UNIQUE(project_id, source_term)
);

-- ── FTS ───────────────────────────────────────────────────────────

CREATE VIRTUAL TABLE bubbles_fts USING fts5(
    source_text, content='bubbles', content_rowid='rowid', tokenize='unicode61'
);

CREATE VIRTUAL TABLE translations_fts USING fts5(
    source_text, translated_text,
    content='translations', content_rowid='rowid', tokenize='unicode61'
);

CREATE VIRTUAL TABLE briefs_fts USING fts5(
    summary, terms_text, facts_text, rules_text,
    content='chapter_briefs', content_rowid='rowid', tokenize='unicode61'
);

CREATE VIRTUAL TABLE glossary_fts USING fts5(
    source_term, content='glossary', content_rowid='id', tokenize='unicode61'
);
```

### 3.2 Filesystem — binary artifacts only

```
~/.typoon/projects/<slug>/ch<NNN>/
├── pages/
│   ├── 0001.png          prepared image (source pixel)
│   └── ...
├── scan.bin              bubble geometry: polygon, fit_box, erase_box, text_box
│                         format: msgpack. Không có text — text ở DB.
├── masks/
│   └── <page>_<bubble>.npz   erase + text masks (numpy arrays)
└── render/
    ├── 0001.png          rendered output
    └── ...
```

**Không có JSON stage files.** Không có `manifest.json`, `scan.json`,
`translate.json`.

**`scan.bin` format (msgpack):**

```python
{
    "version": 1,
    "pages": [
        {
            "index": 0,
            "width": 1200,
            "height": 1800,
            "bubbles": [
                {
                    "idx": 0,
                    "polygon": [[x, y], ...],   # list of [float, float]
                    "fit_box":   [x, y, w, h],  # int[4]
                    "erase_box": [x, y, w, h],
                    "text_box":  [x, y, w, h],
                }
            ]
        }
    ]
}
```

Geometry là float arrays — binary encoding nhỏ hơn JSON ~4x, không cần parse text.

### 3.3 Stage done detection

Stage completion được derive từ data presence — không stored:

```python
def is_prepared(cp: ChapterPaths) -> bool:
    return cp.pages.exists() and any(cp.pages.iterdir())

def is_scanned(cp: ChapterPaths) -> bool:
    return cp.scan_bin.exists()

async def is_translated(db, chapter_id: int) -> bool:
    return await db.has_translations(chapter_id)

def is_rendered(cp: ChapterPaths) -> bool:
    return cp.render.exists() and any(cp.render.iterdir())
```

Không có `chapters.status`. Không có flag. Mỗi câu hỏi được hỏi đúng source.

---

## 4. Worker Queue Design

### 4.1 Atomic claim

Worker claim task bằng atomic UPDATE. SQLite WAL đảm bảo serialization.

```sql
UPDATE tasks
SET    claimed_by = :worker_id,
       claimed_at = datetime('now')
WHERE  rowid = (
    SELECT rowid FROM tasks
    WHERE  stage = :stage
      AND  (claimed_by IS NULL
            OR claimed_at < datetime('now', '-10 minutes'))
    ORDER  BY chapter_id
    LIMIT  1
)
RETURNING chapter_id, stage
```

`claimed_at < -10 minutes` là stale lock recovery. Worker crash → lock expire tự
động → next worker re-claim.

### 4.2 Task lifecycle

```
Enqueue:   INSERT INTO tasks (chapter_id, stage)
Claim:     UPDATE SET claimed_by=?, claimed_at=now
Success:   DELETE FROM tasks WHERE chapter_id=? AND stage=?
           → Nếu có next stage: INSERT INTO tasks (chapter_id, next_stage)
Fail:      UPDATE SET claimed_by=NULL, attempts=attempts+1, last_error=?
           → Nếu attempts >= MAX_ATTEMPTS: không re-insert, worker log error
```

### 4.3 Worker processes

Ba worker loops chạy as separate processes hoặc asyncio tasks trong worker process.
Không chạy chung với API process.

```
scan_worker      (1 instance)   — exclusive VisionRuntime.scanner
translate_worker (N instances)  — concurrent LLM I/O bound
render_worker    (1 instance)   — exclusive VisionRuntime.eraser
```

`N` translate workers = `config.translate_concurrency` (default 3).

Worker là dumb poller — không biết gì về projects, không biết gì về API:

```python
async def scan_worker(db: Store, runtime: VisionRuntime) -> None:
    worker_id = str(uuid4())
    while True:
        chapter_id = await db.claim_task("scan", worker_id)
        if chapter_id is None:
            await asyncio.sleep(1)
            continue
        try:
            cp = await db.chapter_paths(chapter_id)
            prepared = load_prepared(cp)                          # đọc filesystem
            result = await asyncio.to_thread(scan_chapter, prepared, runtime)
            result.save_bin(cp)                                   # ghi scan.bin
            await db.save_bubbles(chapter_id, result.bubbles)    # text → DB
            await db.complete_task(chapter_id, "scan")           # DELETE + INSERT translate
        except Exception as e:
            await db.fail_task(chapter_id, "scan", str(e))
```

`asyncio.to_thread` cho phép scan ch2 overlap với translate ch1 trong cùng process
mà không block event loop. CoreML và numpy release GIL khi vào native code.

### 4.4 Overlap timeline

```
ch1:   [prepare][scan]─────────────────────────────[render]
ch2:            [prepare][scan]──────────────[render]
ch3:                     [prepare][scan]─[render]
         translate:      [ch1──────────────][ch2────][ch3──]

scan_worker:    [ch1][ch2][ch3]       ← serialize
translate:      ch1+ch2+ch3 overlap   ← N concurrent
render_worker:              [ch3][ch2][ch1]  ← serialize
```

---

## 5. API Design

### 5.1 Architecture

API process chỉ đọc/ghi DB. Không load vision models. Không chạy pipeline.
Worker process độc lập, poll DB.

```
Client → FastAPI (api/) → SQLite DB ← Worker (workers/)
                                    → Filesystem (read/write)
```

### 5.2 Endpoints

```
# Discovery — stateless, không lưu DB
POST   /discover
       body:  { url: string }
       return: { title, source_lang, chapters: [{number, title, variants}] }

# Projects
POST   /projects
       body:  { title, source_lang, target_lang, source_url? }
       return: { id, slug }

GET    /projects
       return: [{ id, slug, title, source_lang, target_lang, chapter_count, done_count }]

GET    /projects/{slug}
       return: { ...project, chapters: [{ idx, status, bubble_count, render_count }] }

DELETE /projects/{slug}

# Chapters — enqueue work
POST   /projects/{slug}/pull
       body:  { url: string, chapters: [number] }
       return: { enqueued: number }
       effect: download images → enqueue prepare tasks

POST   /projects/{slug}/import
       body:  { path: string }   # local folder
       return: { enqueued: number }

POST   /projects/{slug}/translate
       body:  { chapters?: [number], redo?: "scan"|"translate"|"render" }
       return: { enqueued: number }

GET    /projects/{slug}/chapters/{idx}
       return: { idx, status, stages: {...}, bubble_count, error? }

# Output — serve rendered files
GET    /projects/{slug}/chapters/{idx}/pages
       return: [{ page: number, url: string }]

GET    /projects/{slug}/chapters/{idx}/pages/{n}
       return: image/png (file serve)

# Real-time
GET    /projects/{slug}/chapters/{idx}/stream
       return: text/event-stream (SSE)
       events: StageStarted, StageDone, StageFailed, ChapterDone
```

### 5.3 Status computation

Status là derived — computed từ tasks + filesystem:

```python
async def compute_chapter_status(chapter_id: int, cp: ChapterPaths, db) -> dict:
    if is_rendered(cp):
        return {"status": "done"}

    tasks = await db.get_tasks(chapter_id)

    if running := [t for t in tasks if t.claimed_by]:
        return {"status": "running", "stage": running[0].stage}

    if failed := [t for t in tasks if t.last_error and t.attempts >= MAX_ATTEMPTS]:
        return {"status": "error", "stage": failed[0].stage, "error": failed[0].last_error}

    if pending := [t for t in tasks if not t.claimed_by]:
        return {"status": "pending", "stage": pending[0].stage}

    # Không có task và chưa rendered — idle hoặc partially done
    if is_translated(db, chapter_id): return {"status": "idle", "next": "render"}
    if is_scanned(cp):                return {"status": "idle", "next": "translate"}
    if is_prepared(cp):               return {"status": "idle", "next": "scan"}
    return {"status": "idle", "next": "prepare"}
```

### 5.4 SSE — real-time progress

Worker emit events vào `task_events` table (ring buffer, giữ N events per chapter).
SSE endpoint poll và stream ra client. Không cần WebSocket.

```sql
CREATE TABLE task_events (
    id          INTEGER PRIMARY KEY,
    chapter_id  INTEGER NOT NULL REFERENCES chapters(id) ON DELETE CASCADE,
    event_type  TEXT NOT NULL,
    payload     TEXT,
    ts          TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX idx_task_events_chapter ON task_events(chapter_id, id);
```

Client poll `GET /stream?after_id=<last_id>` hoặc dùng SSE long-poll.

---

## 6. Domain Type Changes

### 6.1 Load từ DB thay vì filesystem chain

Hiện tại domain types load theo chain:

```python
TranslatedChapter.load(cp) → ScannedChapter.load(cp) → PreparedChapter.load(root)
```

Sau RFC: load từ DB + scan.bin:

```python
# Scan stage input/output
ScanInput   = prepared pages directory (filesystem)
ScanOutput  = scan.bin (geometry) + bubbles rows (DB text)

# Translate stage input
TranslateInput = bubbles rows from DB + chapter_briefs

# Render stage input
RenderInput = scan.bin (geometry) + translations rows + masks (filesystem)
```

Domain types không còn nested chain. Mỗi stage load đúng thứ nó cần từ đúng source.

### 6.2 Bubble identity

Identity là `(chapter_id, page_index, bubble_idx)` — structural position, không
phải hash của content. Rescan không làm orphan translations cũ vì:

- Nếu bubble vẫn ở đúng vị trí → identity giữ nguyên → translation reuse
- Nếu bubble biến mất → CASCADE DELETE translations của bubble đó
- Translation key cho LLM vẫn được generate runtime trong `keys.py` nhưng không
  persist vào DB

### 6.3 `translate.json` không còn

`translate.json` là redundant với `translations` DB rows. Bỏ. Stage "translate done"
được detect bằng `has_translations(chapter_id)`.

---

## 7. Package Structure Changes

```
typoon/
  domain/       không đổi về types, load logic thay đổi
  stages/       scan/translate/render không đổi về business logic
                pipeline.py → bỏ, thay bằng:
  workers/      NEW
    loop.py     worker main loop, poll DB
    claim.py    atomic claim logic
    scan.py     scan worker
    translate.py translate worker
    render.py   render worker
  api/          NEW
    app.py      FastAPI app
    routes/
      projects.py
      chapters.py
      output.py
      stream.py
    deps.py     shared dependencies (db, paths)
  storage/
    sqlite.py   schema mới + claim query
    records.py  updated types
  adapters/
    project_service.py  → bỏ hoặc thu gọn thành thin enqueue layer
```

---

## 8. What Does Not Change

- `domain/` types: `PreparedPage`, `ScannedBubble`, `TranslatedBubble`, v.v.
- `stages/scan.py`, `stages/translate.py`, `stages/render.py`, `stages/prepare.py`
  — business logic giữ nguyên
- `vision/` — toàn bộ không đổi
- `adapters/vision_runtime.py` — không đổi
- `llm/`, `agents/` — không đổi
- `sources/` — không đổi
- `cli/` — giữ, gọi vào workers/api thay vì project_service

---

## 9. Migration Path

Đang trong giai đoạn dev — không cần migration script. Thứ tự implement:

1. **Storage** — viết schema mới, xóa schema cũ, update `SqliteStore`
2. **Domain load** — update domain types load từ DB + scan.bin thay vì JSON chain
3. **Workers** — viết worker loops với claim logic
4. **Stages wiring** — wire stages vào workers, bỏ `pipeline.py` monolith
5. **API** — FastAPI skeleton, routes, SSE
6. **CLI** — update CLI gọi vào worker/api thay vì ProjectService

Mỗi bước verify bằng E2E một chapter trước khi đi tiếp.

---

## 10. Open Questions

**Q1.** `task_events` table có cần không, hay SSE poll `tasks` table là đủ?  
Tasks table chỉ có current state, không có history. Nếu client miss event thì
không biết gì đã xảy ra. `task_events` giải quyết điều này.

**Q2.** Download là task riêng hay inline với prepare?  
Download là I/O bound, dễ fail, cần retry riêng. Nên là task riêng:
`download → prepare → scan → translate → render`.

**Q3.** `scan.bin` format: msgpack vs flatbuffers vs numpy npz?  
Msgpack: simple, nhiều library. Flatbuffers: zero-copy nhưng cần schema. Numpy npz:
đã dùng cho masks nhưng không phù hợp cho mixed types. Đề xuất msgpack.
