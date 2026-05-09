"""ChannelBus — Postgres LISTEN/NOTIFY scoped per channel.

Subscribers register interest in a specific channel name (typically
'typoon:project:<id>'). The bus opens an underlying Postgres LISTEN
on that channel only when at least one in-process subscriber is
interested, and drops the LISTEN when the last subscriber leaves.

Why channels not filters: Postgres pg_notify routes payloads to
LISTEN sessions matching the channel name. Pushing the scope down to
the database means:

  - the API process never sees events for projects no one is viewing
  - multi-tenant fan-out scales with active scope, not total users
  - no ad-hoc app-side filter code, no events table, no replay buffer

Lifecycle:

  bus.start()                    open the listener connection
  async with bus.subscribe(ch):  refcount up; LISTEN if first
      ...                        consume queue
                                 refcount down; UNLISTEN if last

Reconnect: if the listener connection drops, the bus reconnects with
exponential backoff and re-LISTENs every channel currently held by
subscribers, so transient outages are invisible to consumers.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import AsyncIterator

logger = logging.getLogger(__name__)


# ── Tunables ──────────────────────────────────────────────────────────


# Per-subscriber queue depth. Slow consumers drop oldest events when
# this fills; lifecycle handler on the client reloads view state on
# reconnect anyway, so we don't try to buffer past disconnect windows.
_SUBSCRIBER_QUEUE_SIZE = 256

# Exponential backoff bounds for the listener reconnect loop.
_BACKOFF_INITIAL = 1.0
_BACKOFF_MAX = 30.0


# ── Implementation ────────────────────────────────────────────────────


class ChannelBus:
    """Postgres-backed channel pub/sub. See module docstring."""

    def __init__(self, dsn: str) -> None:
        if not dsn or not dsn.startswith(("postgresql://", "postgres://")):
            raise RuntimeError(
                f"ChannelBus requires a postgresql:// DSN, got: {dsn!r}"
            )
        self._dsn = dsn
        self._pool = None
        self._listen_conn = None
        self._channels: dict[str, set[asyncio.Queue[dict]]] = {}
        self._lock = asyncio.Lock()
        self._listener_task: asyncio.Task | None = None
        self._closed = asyncio.Event()
        self._connection_ready = asyncio.Event()

    # ── lifecycle ────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the background listener task. Idempotent."""
        if self._listener_task is not None and not self._listener_task.done():
            return
        self._closed.clear()
        self._listener_task = asyncio.create_task(
            self._listener_loop(), name="channel-bus-listener",
        )

    async def close(self) -> None:
        self._closed.set()
        if self._listener_task is not None:
            self._listener_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._listener_task
            self._listener_task = None
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    # ── publish ──────────────────────────────────────────────────────

    async def publish(self, channel: str, data: dict) -> None:
        pool = await self._ensure_pool()
        payload = json.dumps(data)
        async with pool.acquire() as conn:
            await conn.execute(
                "SELECT pg_notify($1, $2)", channel, payload,
            )

    # ── subscribe ────────────────────────────────────────────────────

    @contextlib.asynccontextmanager
    async def subscribe(
        self, channel: str,
    ) -> AsyncIterator[asyncio.Queue[dict]]:
        """Register interest in `channel` for the duration of the
        async-with block.

        First subscriber on a channel triggers LISTEN; last subscriber
        leaving triggers UNLISTEN. Concurrency-safe.
        """
        # Wait for the listener connection so the LISTEN command has
        # somewhere to go. start() is called once at app startup; this
        # await is essentially instant after that.
        await self._connection_ready.wait()

        queue: asyncio.Queue[dict] = asyncio.Queue(
            maxsize=_SUBSCRIBER_QUEUE_SIZE,
        )

        async with self._lock:
            first = channel not in self._channels
            if first:
                self._channels[channel] = set()
                assert self._listen_conn is not None
                await self._listen_conn.add_listener(channel, self._on_notify)
            self._channels[channel].add(queue)
        try:
            yield queue
        finally:
            async with self._lock:
                bucket = self._channels.get(channel)
                if bucket is None:
                    return
                bucket.discard(queue)
                if not bucket:
                    del self._channels[channel]
                    if self._listen_conn is not None:
                        with contextlib.suppress(Exception):
                            await self._listen_conn.remove_listener(
                                channel, self._on_notify,
                            )

    # ── listener task ────────────────────────────────────────────────

    async def _listener_loop(self) -> None:
        """Run a listener connection, reconnect with backoff on drop."""
        backoff = _BACKOFF_INITIAL
        while not self._closed.is_set():
            try:
                await self._run_once()
                return  # closed cleanly
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception(
                    "channel bus listener crashed; reconnecting in %.1fs",
                    backoff,
                )
                self._connection_ready.clear()
                try:
                    await asyncio.wait_for(self._closed.wait(), timeout=backoff)
                    return
                except asyncio.TimeoutError:
                    pass
                backoff = min(backoff * 2, _BACKOFF_MAX)
            else:
                backoff = _BACKOFF_INITIAL

    async def _run_once(self) -> None:
        import asyncpg

        conn = await asyncpg.connect(self._dsn)
        terminated = asyncio.Event()
        conn.add_termination_listener(lambda _c: terminated.set())

        async with self._lock:
            self._listen_conn = conn
            # Re-attach LISTEN for every channel the app currently
            # tracks. After a reconnect this restores the prior set
            # without subscribers having to do anything.
            for channel in list(self._channels):
                await conn.add_listener(channel, self._on_notify)
            self._connection_ready.set()

        try:
            close_task = asyncio.create_task(self._closed.wait())
            term_task = asyncio.create_task(terminated.wait())
            done, pending = await asyncio.wait(
                {close_task, term_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
            if term_task in done and not self._closed.is_set():
                raise ConnectionError(
                    "Postgres LISTEN connection terminated",
                )
        finally:
            self._connection_ready.clear()
            async with self._lock:
                self._listen_conn = None
            with contextlib.suppress(Exception):
                await conn.close(timeout=2.0)

    def _on_notify(
        self, _conn, _pid: int, channel: str, payload: str,
    ) -> None:
        try:
            data = json.loads(payload)
        except (ValueError, TypeError):
            logger.warning("dropping malformed NOTIFY payload on %s", channel)
            return
        # Snapshot the set; mutation during iteration is otherwise
        # possible if a subscriber unregisters mid-fanout.
        bucket = self._channels.get(channel)
        if not bucket:
            return
        for queue in tuple(bucket):
            try:
                queue.put_nowait(data)
            except asyncio.QueueFull:
                # Slow consumer — drop. The client reloads its view
                # state on reconnect, so missed transient updates
                # heal naturally.
                pass

    # ── pool ─────────────────────────────────────────────────────────

    async def _ensure_pool(self):
        if self._pool is None:
            async with self._lock:
                if self._pool is None:
                    import asyncpg
                    self._pool = await asyncpg.create_pool(
                        self._dsn, min_size=2, max_size=8,
                    )
        return self._pool


# ── Channel naming ────────────────────────────────────────────────────


def project_channel(project_id: int) -> str:
    """Channel name for project-scoped events."""
    return f"typoon:project:{project_id}"


# ── Hook bridge — thread-safe ─────────────────────────────────────────


import dataclasses

from typoon.runs.events import Event, Hook


def event_to_dict(event: Event) -> dict:
    """Serialize an Event dataclass to a dict for the channel bus.

    We avoid `dataclasses.asdict`: it recursively deepcopies every
    field, and several Event types carry an `error: Exception` whose
    third-party subclasses (e.g. `openai.APIStatusError`) cannot be
    deepcopied because their `__init__` requires kwargs that aren't
    captured by the default reduce protocol. A shallow walk over the
    declared fields is sufficient here — events are flat scalar
    records by design.
    """
    if not dataclasses.is_dataclass(event):
        return {"type": type(event).__name__}
    d: dict = {}
    for f in dataclasses.fields(event):
        if f.name == "ts":
            continue
        v = getattr(event, f.name)
        if isinstance(v, BaseException):
            v = str(v) or type(v).__name__
        d[f.name] = v
    d["type"] = type(event).__name__
    return d


class ChannelHook(Hook):
    """Bridge sync Hook.on(event) → async ChannelBus.publish(channel, dict).

    Routes events to per-project channels by reading event.project_id.
    Events without a project_id are silently dropped (e.g. ToolCallStart
    from inside an LLM agent that isn't bound to any project context).

    Thread-safe: stages may emit from worker threads (asyncio.to_thread).
    We capture the loop at construction and schedule publishes onto it
    via run_coroutine_threadsafe.
    """

    def __init__(self, bus: ChannelBus, loop: asyncio.AbstractEventLoop) -> None:
        self._bus = bus
        self._loop = loop

    def on(self, event: Event) -> None:
        project_id = getattr(event, "project_id", None)
        if not isinstance(project_id, int) or project_id <= 0:
            return
        channel = project_channel(project_id)
        data = event_to_dict(event)
        try:
            asyncio.run_coroutine_threadsafe(
                self._bus.publish(channel, data), self._loop,
            )
        except RuntimeError:
            # Loop closed during shutdown.
            logger.debug("channel bus closed; dropped %s", data.get("type"))
