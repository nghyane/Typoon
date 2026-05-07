"""Event bus — pipeline event transport.

Two implementations, chosen by deploy mode:

  InProcessEventBus  — single-process (SQLite). API + workers in same loop.
                       No persistence. asyncio.Queue fan-out.
  PostgresEventBus   — multi-host (Postgres). LISTEN/NOTIFY for wakeup,
                       `events` table for replay via Last-Event-ID.

SQLite mode does NOT support multi-host. Workers and API must share the
same Python process. Use Postgres for any scale-out deploy.

Selection: make_event_bus(config, store) — single source of truth.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from typing import AsyncIterator, Protocol, runtime_checkable

from typoon.config import Config
from typoon.runs.events import Event, Hook
from typoon.storage import Store

logger = logging.getLogger(__name__)


# ── Serialization ─────────────────────────────────────────────────────


def event_to_dict(event: Event) -> dict:
    d = dataclasses.asdict(event) if dataclasses.is_dataclass(event) else {}
    d["type"] = type(event).__name__
    d.pop("ts", None)
    for k, v in list(d.items()):
        if isinstance(v, Exception):
            d[k] = str(v)
    return d


# ── Interface ─────────────────────────────────────────────────────────


@runtime_checkable
class EventBus(Protocol):
    async def publish(self, event: Event) -> None: ...
    def subscribe(self, last_id: str = "0") -> AsyncIterator[tuple[str, dict]]: ...
    async def close(self) -> None: ...


# ── In-process (SQLite single-process mode) ───────────────────────────


class InProcessEventBus:
    """Single-process bus: asyncio.Queue fan-out. No persistence.

    Reconnect (Last-Event-ID) replays from in-memory ring buffer (last 256).
    On API restart the buffer is empty — tolerable in single-process mode
    because workers restart together with API.
    """

    _BUFFER_SIZE = 256

    def __init__(self) -> None:
        self._seq = 0
        self._buffer: list[tuple[int, dict]] = []
        self._subscribers: set[asyncio.Queue[tuple[int, dict]]] = set()
        self._lock = asyncio.Lock()

    async def publish(self, event: Event) -> None:
        async with self._lock:
            self._seq += 1
            data = event_to_dict(event)
            entry = (self._seq, data)
            self._buffer.append(entry)
            if len(self._buffer) > self._BUFFER_SIZE:
                self._buffer.pop(0)
            for q in list(self._subscribers):
                q.put_nowait(entry)

    async def subscribe(self, last_id: str = "0") -> AsyncIterator[tuple[str, dict]]:
        seq = int(last_id) if last_id.isdigit() else 0
        q: asyncio.Queue[tuple[int, dict]] = asyncio.Queue()

        async with self._lock:
            for buffered_seq, data in self._buffer:
                if buffered_seq > seq:
                    q.put_nowait((buffered_seq, data))
            self._subscribers.add(q)

        try:
            while True:
                buffered_seq, data = await q.get()
                yield str(buffered_seq), data
        finally:
            self._subscribers.discard(q)

    async def close(self) -> None:
        pass


# ── Postgres (multi-host mode) ────────────────────────────────────────


class PostgresEventBus:
    """Multi-host bus: events table + LISTEN/NOTIFY.

    publish    INSERT events RETURNING id; NOTIFY chan, str(id)
    subscribe  SELECT events WHERE id > last_id (replay), then LISTEN.
               On NOTIFY, SELECT new row by id.

    Pool reused across publishes to avoid connection storms.
    """

    _CHANNEL = "typoon_events"

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool = None  # asyncpg.Pool

    async def _ensure_pool(self):
        if self._pool is None:
            import asyncpg
            self._pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=4)
            async with self._pool.acquire() as conn:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS events (
                        id          BIGSERIAL PRIMARY KEY,
                        data        JSONB NOT NULL,
                        created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                """)
        return self._pool

    async def publish(self, event: Event) -> None:
        pool = await self._ensure_pool()
        payload = json.dumps(event_to_dict(event))
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO events (data) VALUES ($1::jsonb) RETURNING id", payload
            )
            await conn.execute(f"NOTIFY {self._CHANNEL}, '{row['id']}'")

    async def subscribe(self, last_id: str = "0") -> AsyncIterator[tuple[str, dict]]:
        import asyncpg
        seq = int(last_id) if last_id.isdigit() else 0

        # Dedicated connection for LISTEN (cannot share pool conn)
        conn = await asyncpg.connect(self._dsn)
        queue: asyncio.Queue[int] = asyncio.Queue()

        def _on_notify(_conn, _pid, _channel, payload: str) -> None:
            try:
                queue.put_nowait(int(payload))
            except (ValueError, asyncio.QueueFull):
                pass

        await conn.add_listener(self._CHANNEL, _on_notify)
        try:
            # Replay missed events
            rows = await conn.fetch(
                "SELECT id, data FROM events WHERE id > $1 ORDER BY id LIMIT 1000", seq
            )
            for row in rows:
                seq = row["id"]
                yield str(seq), _parse_json(row["data"])

            # Live events
            while True:
                new_id = await queue.get()
                if new_id <= seq:
                    continue
                row = await conn.fetchrow(
                    "SELECT id, data FROM events WHERE id = $1", new_id
                )
                if row is None:
                    continue
                seq = row["id"]
                yield str(seq), _parse_json(row["data"])
        finally:
            try:
                await conn.remove_listener(self._CHANNEL, _on_notify)
            finally:
                await conn.close()

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None


def _parse_json(value) -> dict:
    return value if isinstance(value, dict) else json.loads(value)


# ── Factory ───────────────────────────────────────────────────────────


def is_postgres(database_url: str) -> bool:
    return database_url.startswith(("postgresql://", "postgres://"))


def make_event_bus(config: Config, store: Store | None = None) -> EventBus:
    """Pick the right bus for this deploy mode.

    SQLite (or empty database_url) ⇒ InProcessEventBus (single-process only).
    Postgres                       ⇒ PostgresEventBus (multi-host capable).

    `store` is unused now but kept for future single-process modes that may
    persist events to SQLite for offline inspection.
    """
    if is_postgres(config.database_url):
        return PostgresEventBus(config.database_url)
    return InProcessEventBus()


# ── Hook bridge — thread-safe ─────────────────────────────────────────


class EventHook(Hook):
    """Bridge sync Hook.on(event) → async EventBus.publish(event).

    Thread-safe: stages run on worker threads (asyncio.to_thread). We
    capture the event loop at construction and schedule publishes onto it
    via run_coroutine_threadsafe.
    """

    def __init__(self, bus: EventBus, loop: asyncio.AbstractEventLoop) -> None:
        self._bus = bus
        self._loop = loop

    def on(self, event: Event) -> None:
        try:
            asyncio.run_coroutine_threadsafe(self._bus.publish(event), self._loop)
        except RuntimeError:
            # Loop is closed during shutdown — drop event silently.
            logger.debug("event bus closed; dropped %s", type(event).__name__)
