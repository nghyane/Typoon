"""Event bus — Postgres LISTEN/NOTIFY transport.

Single backend (RFC-005). API and workers connect to the same Postgres
DSN; the API holds a LISTEN connection per SSE client and replays
missed events from the `events` table on reconnect.

publish    INSERT events RETURNING id; NOTIFY chan, str(id)
subscribe  SELECT events WHERE id > last_id (replay), then LISTEN.
           On NOTIFY, SELECT new row by id.

Pool reused across publishes to avoid connection storms. A dedicated
asyncpg connection per subscriber holds the LISTEN — Postgres LISTEN
state is per-connection, so the pool's connection-recycling would lose
notifications.
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


class EventBus:
    """Postgres-backed bus. See module docstring."""

    _CHANNEL = "typoon_events"

    def __init__(self, dsn: str) -> None:
        if not dsn or not dsn.startswith(("postgresql://", "postgres://")):
            raise RuntimeError(
                f"EventBus requires a postgresql:// DSN, got: {dsn!r}"
            )
        self._dsn = dsn
        self._pool = None  # asyncpg.Pool

    async def _ensure_pool(self):
        if self._pool is None:
            import asyncpg
            self._pool = await asyncpg.create_pool(
                self._dsn, min_size=1, max_size=4,
            )
        return self._pool

    async def publish(self, event: Event) -> None:
        pool = await self._ensure_pool()
        payload = json.dumps(event_to_dict(event))
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO events (data) VALUES ($1::jsonb) RETURNING id",
                payload,
            )
            await conn.execute(f"NOTIFY {self._CHANNEL}, '{row['id']}'")

    async def subscribe(self, last_id: str = "0") -> AsyncIterator[tuple[str, dict]]:
        import asyncpg
        seq = int(last_id) if last_id.isdigit() else 0

        # Dedicated connection — LISTEN state is per-connection.
        conn = await asyncpg.connect(self._dsn)
        queue: asyncio.Queue[int] = asyncio.Queue()

        def _on_notify(_conn, _pid, _channel, payload: str) -> None:
            try:
                queue.put_nowait(int(payload))
            except (ValueError, asyncio.QueueFull):
                pass

        await conn.add_listener(self._CHANNEL, _on_notify)
        try:
            # Replay missed events.
            rows = await conn.fetch(
                "SELECT id, data FROM events WHERE id > $1 ORDER BY id LIMIT 1000",
                seq,
            )
            for row in rows:
                seq = row["id"]
                yield str(seq), _parse_json(row["data"])

            # Live events.
            while True:
                new_id = await queue.get()
                if new_id <= seq:
                    continue
                row = await conn.fetchrow(
                    "SELECT id, data FROM events WHERE id = $1", new_id,
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
