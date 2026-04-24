# RFC: Complete Architecture Alignment

## Status
In Progress

## Context
Architecture spec (ARCHITECTURE.md) defines target structure. Codebase has completed Phase 1-6 of initial refactor but still has gaps vs spec.

## Decisions

### 1. Keep `ports.py` and `engine.py`
Spec says "delete" but both are actively used and have clear responsibility:
- `ports.py`: Protocols shared by `app/`, `adapters/`, `interfaces/`. Moving them "with consumers" causes import cycles and duplication.
- `engine.py`: Vision compute (preprocess, erase, render). Stateless, not mixed. Spec likely meant "don't call directly from UI" which `AppService` already enforces.

**Decision**: Keep both. Update ARCHITECTURE.md to match reality.

### 2. Create `storage/` package
Spec has `storage/sqlite.py`. Currently `adapters/sqlite_store.py`. Store is a port implementation, not an adapter for external service.

**Decision**: Move `adapters/sqlite_store.py` → `storage/sqlite.py`. Create `storage/__init__.py` exposing `SqliteStore`.

### 3. Merge TUI + cli_hook into `interfaces/`
Spec has `interfaces/tui.py`. Currently `adapters/tui.py` + `adapters/cli_hook.py`. Both are UI layer.

**Decision**: Move `adapters/tui.py` → `interfaces/tui.py`. Merge `cli_hook.py` into `interfaces/cli_output.py` (both handle output/sink). Delete `adapters/cli_hook.py`.

### 4. Merge `events.py` into `app/events.py`
Currently both `events.py` (root) and `app/events.py` exist. Root `events.py` has Hook + event dataclasses. `app/events.py` has EventSink + bridge.

**Decision**: Move root `events.py` content into `app/events.py`. Update all imports. Delete root `events.py`.

### 5. Create `app/state.py`
Small, pure data. ChapterState enum used by ResumePolicy/_decide.

**Decision**: Add `app/state.py` with ChapterState enum. Update `policy.py` to reference it (optional, not breaking).

### 6. Skip `app/scheduler.py` for now
Resource semaphore management. New feature, not refactor. Current ThreadPoolExecutor in pipeline.py works. Will create scheduler when needed for server/platform mode.

## Changes

| File | Action |
|---|---|
| `adapters/sqlite_store.py` | Move → `storage/sqlite.py` |
| `adapters/cli_hook.py` | Merge into `interfaces/cli_output.py` |
| `adapters/tui.py` | Move → `interfaces/tui.py` |
| `events.py` | Merge into `app/events.py`, then delete |
| `app/events.py` | Expand to include all event types |
| `app/state.py` | Create — ChapterState enum |
| `ARCHITECTURE.md` | Update to reflect kept files (`ports.py`, `engine.py`) |

## Import updates needed
- All `from typoon.adapters.sqlite_store import SqliteStore` → `from typoon.storage.sqlite import SqliteStore`
- All `from typoon.events import ...` → `from typoon.app.events import ...`
- `interfaces/cli_commands.py`: `from ..adapters.tui import TUI` → `from ..interfaces.tui import TUI`
- `interfaces/cli_pipeline.py`: `from ..adapters.cli_hook import RichHook` → `from ..interfaces.cli_output import RichHook`

## Risks
- Low. Mechanical moves, no logic changes. Tests verify imports.

## Verification
- `python -c "from typoon.storage.sqlite import SqliteStore"`
- `python -c "from typoon.app.events import Hook, EventSink"`
- `python -c "from typoon.interfaces.tui import TUI"`
- `pytest tests/test_llm_providers.py tests/test_knowledge.py`

## Timeline
Single commit. ~30 min.
