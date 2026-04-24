# RFC: Per-Project Workspace for Image Cache and Output Isolation

**Status:** Draft  
**Author:** Assistant  
**Date:** 2025-01-24  
**Scope:** `typoon/config.py`, `typoon/downloader.py`, `typoon/storage/`, CLI commands, pipeline  

---

## 1. Problem Statement

### 1.1 Current flat layout

```
~/.typoon/
├── config.toml
├── typoon.db
├── models/
├── cache/
└── output/          ← ALL projects dump here
    ├── p001.jpg     ← from project A, chapter 1
    ├── p001.jpg     ← from project B, chapter 1 (overwrite or stale)
    └── chapter.pdf
```

### 1.2 Issues

| Issue | Impact | Frequency |
|---|---|---|
| **No per-project isolation** | Project A output overwrites Project B | Every multi-project user |
| **No image cache** | Resume re-downloads all pages | Every crash/resume |
| **No on-disk project state** | Hard to debug without SQLite queries | Every bug report |
| **Flat `output/`** | Impossible to zip/share one chapter | Every share action |

### 1.3 Data flow today

```
URL → httpx.download → np.ndarray (memory) → scan → render → ~/.typoon/output/*.jpg
                              ↑
                              └─ discarded after render; resume = re-download
```

---

## 2. Goals

1. **Isolation**: Project A and Project B never share disk state.
2. **Resume**: Download once, cache forever (until user deletes).
3. **Transparency**: Project state visible on disk (`ls`, `cat`, `zip`).
4. **Backward compat**: Existing `typoon.db` schema stays; only paths change.

Non-goals:
- Change translation/glossary schema.
- Add cloud sync.
- Replace SQLite with JSON.

---

## 3. Proposed Design

### 3.1 Directory layout

```
~/.typoon/
├── config.toml              # global: API keys, default_target_lang
├── typoon.db                # index: projects list, chapter status, glossary
├── models/                  # shared weights
│
└── projects/                # PER-PROJECT workspace
    └── {slug}/              # slug = sanitized title or "unnamed-{id}"
        ├── project.toml     # project-level config
        ├── .state.json      # chapter status mirror (debug/inspect)
        ├── source/          # cached raw images (download once)
        │   ├── ch001/
        │   │   ├── p001.webp
        │   │   └── p002.webp
        │   └── ch002/
        │       └── ...
        └── output/          # rendered results per chapter
            ├── ch001/
            │   ├── p001.jpg
            │   ├── p002.jpg
            │   └── chapter.pdf
            └── ch002/
                └── ...
```

### 3.2 `project.toml` schema

```toml
[project]
source_url = "https://comix.to/manga/jujutsu-kaisen"
title = "Jujutsu Kaisen"
source_lang = "ja"
target_lang = "vi"
created_at = "2025-01-24T10:00:00Z"

[download]
# last known chapter range; used for resume
from_ch = 1.0
to_ch = 10.0
```

### 3.3 `.state.json` (mirror, read-only for humans)

```json
{
  "project_id": 42,
  "slug": "jujutsu-kaisen",
  "chapters": {
    "1.0": { "status": "done", "pages": 18, "bubbles": 47 },
    "2.0": { "status": "translating", "pages": 0, "bubbles": 0 },
    "3.0": { "status": "pending" }
  },
  "updated_at": "2025-01-24T10:05:00Z"
}
```

Written by pipeline after each chapter; **not** a replacement for SQLite.

---

## 4. Detailed Changes

### 4.1 `typoon/config.py` — new `ProjectPaths` class

```python
class ProjectPaths:
    """Paths for a single project workspace."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    @property
    def config(self) -> Path:      return self.root / "project.toml"
    @property
    def state(self) -> Path:      return self.root / ".state.json"
    @property
    def source(self) -> Path:     return self.root / "source"
    @property
    def output(self) -> Path:     return self.root / "output"

    def chapter_source(self, ch: float) -> Path:
        return self.source / f"ch{ch:06.1f}"

    def chapter_output(self, ch: float) -> Path:
        return self.output / f"ch{ch:06.1f}"

    def ensure(self) -> None:
        for d in (self.source, self.output):
            d.mkdir(parents=True, exist_ok=True)
```

`Paths` gains:
```python
@property
def projects_dir(self) -> Path: 
    return self.root / "projects"

def project(self, slug: str) -> ProjectPaths:
    return ProjectPaths(self.projects_dir / slug)
```

### 4.2 `typoon/downloader.py` — cache-aware download

**Change:** Accept `dest` as chapter-specific source dir.

```python
async def download_images(
    urls: list[str],
    dest: Path,              # = ProjectPaths.chapter_source(ch)
    headers: dict | None = None,
    max_retries: int = 3,
    concurrency: int = 5,
    skip_existing: bool = True,   # NEW
) -> Path:
```

**Resume logic in pipeline:**
```python
source_dir = project_paths.chapter_source(chapter)
if skip_existing and source_dir.exists():
    files = sorted(source_dir.iterdir())
    if len(files) == expected_pages:
        return source_dir   # skip download entirely
```

### 4.3 `typoon/interfaces/cli_output.py` — per-chapter output

```python
def save_pages(
    pages: list[Page],
    out_dir: Path,           # = ProjectPaths.chapter_output(ch)
) -> int:
    ...
```

No functional change; caller provides different `out_dir`.

### 4.4 Pipeline — `run_pipeline()` integration

**Before:**
```python
source = DirSource(path)   # or URL source
pages, images = engine.preprocess(source)
engine.erase_and_render(pages, images)
# saves to ~/.typoon/output/
```

**After:**
```python
slug = _slugify(project_title)
ppaths = paths.project(slug)
ppaths.ensure()

# 1. Download (or reuse cache)
if not ppaths.chapter_source(ch).exists():
    await download_images(urls, ppaths.chapter_source(ch))

source = CachedSource(ppaths.chapter_source(ch))

# 2. Scan + render
pages, images = engine.preprocess(source)
engine.erase_and_render(pages, images)

# 3. Save to isolated output
save_pages(pages, ppaths.chapter_output(ch))

# 4. Mirror state
_write_state_mirror(ppaths.state, project_id, store)
```

### 4.5 New adapter: `CachedSource`

```python
class CachedSource(ChapterSource):
    """Reads from local disk (download cache)."""

    def __init__(self, source_dir: Path) -> None:
        self._files = sorted(source_dir.iterdir())

    async def fetch(self) -> None:
        pass  # already local

    def page_count(self) -> int:
        return len(self._files)

    def load_page(self, index: int) -> np.ndarray:
        bgr = cv2.imread(str(self._files[index]))
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
```

---

## 5. Migration Path

### 5.1 Existing users

- `~/.typoon/output/` becomes legacy; new writes go to `~/.typoon/projects/{slug}/output/`.
- Old flat output ignored (user can `rm -rf` manually).
- SQLite schema **unchanged**; only `local_path` column in `chapters` table now points to `projects/{slug}/source/{ch}/`.

### 5.2 First-run behavior

```
$ typoon translate https://comix.to/manga/one-piece
→ fetch metadata → title = "One Piece" → slug = "one-piece"
→ create ~/.typoon/projects/one-piece/
→ download ch1 pages to source/ch001.0/
→ render to output/ch001.0/
```

---

## 6. Impact Matrix

| File | Change | Lines |
|---|---|---|
| `typoon/config.py` | Add `ProjectPaths`, update `Paths` | ~30 |
| `typoon/downloader.py` | Add `skip_existing`, accept `dest` | ~10 |
| `typoon/interfaces/cli_output.py` | None (caller changes) | 0 |
| `typoon/interfaces/cli_commands.py` | Resolve slug, create `ProjectPaths` | ~15 |
| `typoon/app/workflows/project/pipeline.py` | Use `CachedSource`, write state mirror | ~20 |
| `typoon/app/workflows/project/chapter.py` | Pass `out_dir` to `save_pages` | ~5 |
| `typoon/storage/sqlite.py` | `local_path` column update logic | ~5 |
| **New** `typoon/adapters/cached_source.py` | `CachedSource` class | ~25 |

Total: ~110 lines across 8 files.

---

## 7. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Long paths on Windows (>260 chars) | Use `pathlib.Path` (handles it); warn if slug > 50 chars |
| Slug collision ("One Piece" vs "one-piece") | Slugify with `{slug}-{hash(url)[0:6]}` |
| Disk space bloat from infinite cache | Add `typoon clean --older-than 30d` command |
| .state.json out of sync with SQLite | Always write after SQLite commit; .state.json is "best effort" |

---

## 8. Alternatives Considered

| Approach | Rejected because |
|---|---|
| Keep flat `output/`, prefix filenames | Still no image cache; filenames get unwieldy |
| Replace SQLite with JSON entirely | Lose ACID, concurrent access, existing queries |
| Cloud sync (iCloud/S3) as primary | Over-engineered; local cache is prerequisite anyway |
| Store images in SQLite BLOB | Slower, larger DB, harder to inspect |

---

## 9. Open Questions

1. Should `source/` store original format (`.webp`) or normalize to `.png`?
2. Should `output/` keep chapter PDF or also emit per-page PNG?
3. Should `.state.json` be human-editable (and read by CLI)?

---

## 10. Acceptance Criteria

- [ ] `~/.typoon/projects/` created on first translate.
- [ ] Re-running same project does not re-download existing chapters.
- [ ] Two different projects produce output in different folders.
- [ ] `ls ~/.typoon/projects/{slug}/output/ch001.0/` shows `p001.jpg` + `chapter.pdf`.
- [ ] `.state.json` reflects chapter statuses after pipeline exit.
- [ ] All existing tests pass (SQLite schema unchanged).
