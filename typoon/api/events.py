"""EventBus — pipeline event streaming.

LocalEventBus    — SQLite polling, zero extra deps, single-VPS.
PostgresEventBus — LISTEN/NOTIFY, multi-VPS, ~10-50ms latency.

Selection is done once in deps.py based on database_url.
Worker and API are fully decoupled — both only talk to DB.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from typing import AsyncIterator, Protocol, runtime_checkable

from typoon.runs.events import Event, Hook


# ── Serialization ─────────────────────────────────────────────────────


def _event_to_dict(event: Event) -> dict:
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
    async def subscribe(self, last_id: str = "0") -> AsyncIterator[tuple[str, dict]]: ...


# ── Local — SQLite polling ────────────────────────────────────────────


class LocalEventBus:
    """Single-VPS. Polls events table every 2s. No extra dependencies."""

    def __init__(self, db) -> None:
        self._db = db

    async def publish(self, event: Event) -> None:
        await self._db.append_event(_event_to_dict(event))

    async def subscribe(self, last_id: str = "0") -> AsyncIterator[tuple[str, dict]]:
        seq = int(last_id) if last_id.isdigit() else 0
        while True:
            rows = await self._db.get_events_after(seq)
            for row in rows:
                seq = row["id"]
                yield str(seq), {k: v for k, v in row.items() if k != "id"}
            if not rows:
                await asyncio.sleep(2)


# ── Postgres — LISTEN/NOTIFY ──────────────────────────────────────────


class PostgresEventBus:
    """Multi-VPS. Uses LISTEN/NOTIFY — ~10-50ms latency, no polling."""

    _CHANNEL = "typoon_events"

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    async def publish(self, event: Event) -> None:
        import asyncpg
        conn = await asyncpg.connect(self._dsn)
        try:
            await conn.execute(
                f"NOTIFY {self._CHANNEL}, $1",
                json.dumps(_event_to_dict(event)),
            )
        finally:
            await conn.close()

    async def subscribe(self, last_id: str = "0") -> AsyncIterator[tuple[str, dict]]:
        import asyncpg
        seq   = int(last_id) if last_id.isdigit() else 0
        conn  = await asyncpg.connect(self._dsn)
        queue: asyncio.Queue[str] = asyncio.Queue()

        def _on_notify(conn, pid, channel, payload):
            queue.put_nowait(payload)

        await conn.add_listener(self._CHANNEL, _on_notify)
        try:
            while True:
                payload = await queue.get()
                seq += 1
                yield str(seq), json.loads(payload)
        finally:
            await conn.remove_listener(self._CHANNEL, _on_notify)
            await conn.close()


# ── Hook bridge ───────────────────────────────────────────────────────


class EventHook(Hook):
    """Bridge Hook.on() → EventBus.publish(). Works with both implementations."""

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus

    def on(self, event: Event) -> None:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self._bus.publish(event))
        except RuntimeError:
            pass
