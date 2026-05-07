"""Worker stage helpers shared by `prepare_archive` and `render_archive`."""

from __future__ import annotations

import contextlib
import tempfile
from pathlib import Path
from typing import Iterator


@contextlib.contextmanager
def workdir(explicit: Path | None = None) -> Iterator[Path]:
    """Yield a workdir path. If `explicit` is None, use a fresh tmp dir
    that is removed on exit; otherwise use it without cleanup."""
    if explicit is None:
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)
    else:
        explicit.mkdir(parents=True, exist_ok=True)
        yield explicit
