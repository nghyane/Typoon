"""Event bus — Postgres LISTEN/NOTIFY transport.

API holds ONE LISTEN connection per process and fans out incoming
NOTIFY payloads to in-process subscribers via local asyncio.Queue.
Replay-on-reconnect uses the regular pool, not the listener connection.

publish    INSERT events RETURNING id; pg_notify(chan, str(id))
subscribe  one-time replay (events.id > last_id), then live
           events from a local queue fed by the shared listener.

Why one listener: opening a dedicated asyncpg connection per SSE
subscriber blew up Postgres `max_connections` at modest user count.
LISTEN state is per-connection, but NOTIFY payloads are visible to
every LISTEN session, so a single connection is enough as long as we
broadcast in-process.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from typing import AsyncIterator, Protocol, runtime_checkable

from typoon.runs.events import Event, Hook

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
class EventBusProtocol(Protocol):
    async def publish(self, event: Event) -> None: ...
    def subscribe(self, last_id: str = "0") -> AsyncIterator[tuple[str, dict]]: ...
    async def close(self) -> None: ...


# ── Implementation ────────────────────────────────────────────────────


# Per-subscriber queue depth. Slow consumers drop oldest events when
# this fills; the missed-id gap is recovered next time the client
# reconnects with a Last-Event-ID. 256 is comfortable for a 100-page
# render's PageDone burst.
_SUBSCRIBER_QUEUE_SIZE = 256

# Replay cap on reconnect. If a client has been offline long enough to
# miss more than this, they get the tail; older events are considered
# historical and live in the events table for audit, not for catch-up.
_REPLAY_LIMIT = 100


class EventBus:
    """Postgres-backed bus. See module docstring."""

    _CHANNEL = "typoon_events"

    def __init__(self, dsn: str) -> None:
        if not dsn or not dsn.startswith(("postgresql://", "postgres://")):
            raise RuntimeError(
                f"EventBus requires a postgresql:// DSN, got: {dsn!r}"
            )
        self._dsn = dsn
        self._pool = None  # asyncpg.Pool — used by publish + replay
        self._listen_conn = None  # asyncpg.Connection — single LISTEN
        self._subscribers: set[asyncio.Queue[int]] = set()
        self._init_lock = asyncio.Lock()

    # ── lazy init ────────────────────────────────────────────────────

    async def _ensure_pool(self):
        # Double-checked locking — avoids creating two pools if two
        # coroutines hit a cold bus simultaneously.
        if self._pool is None:
            async with self._init_lock:
                if self._pool is None:
                    import asyncpg
                    self._pool = await asyncpg.create_pool(
                        self._dsn, min_size=1, max_size=4,
                    )
        return self._pool

    async def _ensure_listener(self):
        if self._listen_conn is not None:
            return
        async with self._init_lock:
            if self._listen_conn is not None:
                return
            import asyncpg
            conn = await asyncpg.connect(self._dsn)
            await conn.add_listener(self._CHANNEL, self._on_notify)
            self._listen_conn = conn

    # ── publish ──────────────────────────────────────────────────────

    async def publish(self, event: Event) -> None:
        pool = await self._ensure_pool()
        payload = json.dumps(event_to_dict(event))
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO events (data) VALUES ($1::jsonb) RETURNING id",
                payload,
            )
            # Parameterized — pg_notify avoids string interpolation
            # with the event id (hardening against any future code path
            # that lets non-numeric values reach this point).
            await conn.execute(
                "SELECT pg_notify($1, $2)",
                self._CHANNEL, str(row["id"]),
            )

    # ── listener side ────────────────────────────────────────────────

    def _on_notify(self, _conn, _pid, _channel, payload: str) -> None:
        try:
            event_id = int(payload)
        except ValueError:
            return
        # Snapshot the set so a subscriber unregistering mid-iteration
        # doesn't blow up.
        for queue in tuple(self._subscribers):
            try:
                queue.put_nowait(event_id)
            except asyncio.QueueFull:
                # Slow consumer — drop. They'll catch up via replay on
                # their next reconnect (Last-Event-ID).
                pass

    # ── subscribe ────────────────────────────────────────────────────

    async def subscribe(self, last_id: str = "0") -> AsyncIterator[tuple[str, dict]]:
        await self._ensure_listener()
        pool = await self._ensure_pool()
        seq = int(last_id) if last_id.isdigit() else 0

        queue: asyncio.Queue[int] = asyncio.Queue(maxsize=_SUBSCRIBER_QUEUE_SIZE)
        self._subscribers.add(queue)
        try:
            # Replay: capped — older history stays in the table for audit.
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT id, data FROM events "
                    "WHERE id > $1 "
                    "ORDER BY id "
                    "LIMIT $2",
                    seq, _REPLAY_LIMIT,
                )
            for row in rows:
                seq = row["id"]
                yield str(seq), _parse_json(row["data"])

            # Live tail: queue is fed by the shared listener.
            while True:
                new_id = await queue.get()
                if new_id <= seq:
                    continue
                async with pool.acquire() as conn:
                    row = await conn.fetchrow(
                        "SELECT id, data FROM events WHERE id = $1", new_id,
                    )
                if row is None:
                    continue
                seq = row["id"]
                yield str(seq), _parse_json(row["data"])
        finally:
            self._subscribers.discard(queue)

    # ── lifecycle ────────────────────────────────────────────────────

    async def close(self) -> None:
        if self._listen_conn is not None:
            try:
                await self._listen_conn.remove_listener(
                    self._CHANNEL, self._on_notify,
                )
            finally:
                await self._listen_conn.close()
            self._listen_conn = None
        if self._pool is not None:
            await self._pool.close()
            self._pool = None


def _parse_json(value) -> dict:
    return value if isinstance(value, dict) else json.loads(value)


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
