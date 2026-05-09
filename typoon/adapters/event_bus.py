"""Event bus — Postgres LISTEN/NOTIFY with in-process fan-out.

Two independent paths:

  Live path: a background asyncio task holds one LISTEN connection,
             decodes the NOTIFY payload (event id + data inline), and
             fans out to per-subscriber asyncio.Queue. Reconnects on
             its own with exponential backoff if the connection dies.

  Replay:    one-shot SELECT at subscribe time using the regular pool,
             ordered, capped. Last-Event-ID resumption.

NOTIFY payload format:

  {"id": <BIGINT>, "data": <event_dict>}

We ship the data inline so subscribers never re-query the events table
on the hot path. Postgres NOTIFY caps payload size at 8000 bytes; on
the rare event that exceeds the budget we send only the id and the
listener fetches the row (degraded but rare).

Subscribers can opt into a project filter at `subscribe()` time. The
filter is applied at fan-out (a few hundred subscribers all watching
distinct projects each see only their slice), keeping outbound work
proportional to the number of *interested* clients per event rather
than the total connected count.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from typing import AsyncIterator, Iterable, Protocol, runtime_checkable

from typoon.runs.events import Event, Hook

logger = logging.getLogger(__name__)


# ── Tunables ──────────────────────────────────────────────────────────


_CHANNEL = "typoon_events"

# Per-subscriber queue depth. Slow consumers drop oldest events when
# this fills; the missed-id gap is recovered next time the client
# reconnects with a Last-Event-ID. 256 is comfortable for a 100-page
# render's PageDone burst.
_SUBSCRIBER_QUEUE_SIZE = 256

# Replay cap on reconnect. Older history stays in the table for audit
# but is not delivered as catch-up.
_REPLAY_LIMIT = 100

# Postgres NOTIFY hard limit is 8000 bytes; leave headroom for json
# framing overhead and channel name.
_NOTIFY_PAYLOAD_BUDGET = 7800

# Exponential backoff bounds for the listener reconnect loop.
_LISTENER_BACKOFF_INITIAL = 1.0
_LISTENER_BACKOFF_MAX = 30.0


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
    async def start(self) -> None: ...
    async def close(self) -> None: ...


# ── Subscriber record ─────────────────────────────────────────────────


@dataclasses.dataclass(eq=False)
class _Subscriber:
    """One in-process consumer of the bus.

    eq=False so each instance is hashable by identity — we store
    subscribers in a set and never compare them by content.
    """
    queue: asyncio.Queue[tuple[int, dict]]
    # None = firehose (every event); set = filter by project_id.
    project_ids: frozenset[int] | None

    def matches(self, data: dict) -> bool:
        if self.project_ids is None:
            return True
        pid = data.get("project_id")
        return pid in self.project_ids


# ── Implementation ────────────────────────────────────────────────────


class EventBus:
    """Postgres-backed bus. See module docstring."""

    def __init__(self, dsn: str) -> None:
        if not dsn or not dsn.startswith(("postgresql://", "postgres://")):
            raise RuntimeError(
                f"EventBus requires a postgresql:// DSN, got: {dsn!r}"
            )
        self._dsn = dsn
        self._pool = None  # asyncpg.Pool — publish + replay
        self._pool_lock = asyncio.Lock()
        self._subscribers: set[_Subscriber] = set()
        self._listener_task: asyncio.Task | None = None
        self._closed = asyncio.Event()

    # ── pool ─────────────────────────────────────────────────────────

    async def _ensure_pool(self):
        if self._pool is None:
            async with self._pool_lock:
                if self._pool is None:
                    import asyncpg
                    # max_size=16 absorbs reconnect storms (e.g. an API
                    # restart with hundreds of SSE clients all replaying
                    # in parallel) without queuing on the pool.
                    self._pool = await asyncpg.create_pool(
                        self._dsn, min_size=2, max_size=16,
                    )
        return self._pool

    # ── publish ──────────────────────────────────────────────────────

    async def publish(self, event: Event) -> None:
        pool = await self._ensure_pool()
        data = event_to_dict(event)
        data_json = json.dumps(data)
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO events (data) VALUES ($1::jsonb) RETURNING id",
                data_json,
            )
            event_id = int(row["id"])
            notify = json.dumps({"id": event_id, "data": data})
            if len(notify) > _NOTIFY_PAYLOAD_BUDGET:
                # Oversized — fall back to id-only; subscribers will
                # fetch the row from the table. Logged once because if
                # this trips often, an event type is too fat.
                logger.warning(
                    "event %s payload too large for NOTIFY (%d bytes); "
                    "shipping id only", data.get("type"), len(notify),
                )
                notify = json.dumps({"id": event_id})
            await conn.execute(
                "SELECT pg_notify($1, $2)", _CHANNEL, notify,
            )

    # ── listener task ────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the background listener. Idempotent.

        Call once at app startup (FastAPI lifespan). Workers publishing
        events do not need to call this — only readers (the SSE host)
        need a listener.
        """
        if self._listener_task is not None and not self._listener_task.done():
            return
        self._closed.clear()
        self._listener_task = asyncio.create_task(
            self._listener_loop(), name="event-bus-listener",
        )

    async def _listener_loop(self) -> None:
        backoff = _LISTENER_BACKOFF_INITIAL
        while not self._closed.is_set():
            try:
                await self._run_listener_once()
                # Clean exit (close requested) — break out.
                return
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception(
                    "event listener crashed; reconnecting in %.1fs", backoff,
                )
                # Wait either for the backoff window or close, whichever
                # comes first, so shutdown isn't blocked by a long sleep.
                try:
                    await asyncio.wait_for(self._closed.wait(), timeout=backoff)
                    return
                except asyncio.TimeoutError:
                    pass
                backoff = min(backoff * 2, _LISTENER_BACKOFF_MAX)
            else:
                backoff = _LISTENER_BACKOFF_INITIAL

    async def _run_listener_once(self) -> None:
        import asyncpg
        conn = await asyncpg.connect(self._dsn)
        terminated = asyncio.Event()

        def _on_terminate(_conn) -> None:
            terminated.set()

        try:
            conn.add_termination_listener(_on_terminate)
            await conn.add_listener(_CHANNEL, self._on_notify)
            # Wait for either an orderly close or the connection
            # dropping out from under us. asyncpg drives the listener
            # via its read loop — termination_listener fires when that
            # loop sees EOF.
            close_task = asyncio.create_task(self._closed.wait())
            term_task = asyncio.create_task(terminated.wait())
            done, pending = await asyncio.wait(
                {close_task, term_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
            if term_task in done and not self._closed.is_set():
                # Connection died — let the outer loop reconnect.
                raise ConnectionError("Postgres LISTEN connection terminated")
        finally:
            try:
                await conn.remove_listener(_CHANNEL, self._on_notify)
            except Exception:
                pass
            try:
                await conn.close(timeout=2.0)
            except Exception:
                pass

    def _on_notify(
        self, _conn, _pid: int, _channel: str, payload: str,
    ) -> None:
        try:
            msg = json.loads(payload)
        except (ValueError, TypeError):
            logger.warning("dropping malformed NOTIFY payload")
            return
        event_id = msg.get("id")
        if not isinstance(event_id, int):
            return
        data = msg.get("data")
        if data is None:
            # Degraded oversized path: NOTIFY only had the id, fetch
            # the row asynchronously. Schedule on the loop so we don't
            # block the listener.
            asyncio.create_task(self._fetch_and_fanout(event_id))
            return
        self._fanout(event_id, data)

    async def _fetch_and_fanout(self, event_id: int) -> None:
        try:
            pool = await self._ensure_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT data FROM events WHERE id = $1", event_id,
                )
            if row is None:
                return
            self._fanout(event_id, _parse_json(row["data"]))
        except Exception:
            logger.exception("oversized event fetch failed for id=%d", event_id)

    def _fanout(self, event_id: int, data: dict) -> None:
        item = (event_id, data)
        # Snapshot the set so a subscriber unregistering mid-iteration
        # doesn't blow up. Filter by project so users tailing one set
        # of projects don't see traffic from others.
        for sub in tuple(self._subscribers):
            if not sub.matches(data):
                continue
            try:
                sub.queue.put_nowait(item)
            except asyncio.QueueFull:
                # Slow consumer — drop. They'll catch up via replay on
                # their next reconnect (Last-Event-ID).
                pass

    # ── subscribe ────────────────────────────────────────────────────

    async def subscribe(
        self,
        last_id: str = "0",
        *,
        project_ids: Iterable[int] | None = None,
    ) -> AsyncIterator[tuple[str, dict]]:
        """Live tail + optional one-shot replay from `last_id`.

        `project_ids=None` means firehose (every event); pass a set of
        ids to filter by `data["project_id"]` on the hot path. Replay
        is filtered the same way using a SQL WHERE clause so it stays
        cheap regardless of total event volume.
        """
        pool = await self._ensure_pool()
        seq = int(last_id) if last_id.isdigit() else 0
        filt = frozenset(project_ids) if project_ids is not None else None

        # Register the live queue BEFORE replay so events that arrive
        # during replay are not dropped. The seq filter below dedups
        # any overlap between replay and live.
        sub = _Subscriber(
            queue=asyncio.Queue(maxsize=_SUBSCRIBER_QUEUE_SIZE),
            project_ids=filt,
        )
        self._subscribers.add(sub)
        try:
            # Cold path: replay. SQL-side project filter so the API
            # process never sees the full table on a narrow tail.
            async with pool.acquire() as conn:
                if filt is None:
                    rows = await conn.fetch(
                        "SELECT id, data FROM events "
                        "WHERE id > $1 "
                        "ORDER BY id "
                        "LIMIT $2",
                        seq, _REPLAY_LIMIT,
                    )
                else:
                    rows = await conn.fetch(
                        "SELECT id, data FROM events "
                        "WHERE id > $1 "
                        "  AND (data->>'project_id')::int = ANY($2::int[]) "
                        "ORDER BY id "
                        "LIMIT $3",
                        seq, list(filt), _REPLAY_LIMIT,
                    )
            for row in rows:
                seq = row["id"]
                yield str(seq), _parse_json(row["data"])

            # Hot path: live tail. The listener fans out (id, data)
            # tuples — no per-event SELECT.
            while True:
                event_id, data = await sub.queue.get()
                if event_id <= seq:
                    continue
                seq = event_id
                yield str(seq), data
        finally:
            self._subscribers.discard(sub)

    # ── lifecycle ────────────────────────────────────────────────────

    async def close(self) -> None:
        self._closed.set()
        if self._listener_task is not None:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except (asyncio.CancelledError, Exception):
                pass
            self._listener_task = None
        if self._pool is not None:
            await self._pool.close()
            self._pool = None


def _parse_json(value) -> dict:
    return value if isinstance(value, dict) else json.loads(value)


# ── Hook bridge — thread-safe ─────────────────────────────────────────


class EventHook(Hook):
    """Bridge sync Hook.on(event) → async EventBus.publish(event).

    Thread-safe: stages run on worker threads (asyncio.to_thread). We
    capture the event loop at construction and schedule publishes onto
    it via run_coroutine_threadsafe.
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
