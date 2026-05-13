"""CLI — operator commands.

After the material refactor, ingestion is HTTP-only:
  - SPA + extension upload chapters via /api/material/.../upload-init|finalize
  - There is no CLI ingest path.

Operator-facing commands that survived the refactor:
  - `typoon api`     — run the FastAPI HTTP service
  - `typoon work`    — run the pipeline workers (prepare/scan/translate/render)
  - `typoon version` — print git SHA + schema version

Project-bound commands (redo, export, status, prune, debug-scan,
retry-failed) are removed for this slice. They'll return as
material-aware variants once the rest of the pipeline lands.
"""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console

app     = typer.Typer(name="typoon", help="Manga translation pipeline.")
console = Console()


# ── api ───────────────────────────────────────────────────────────────


@app.command()
def api(
    host:   str  = typer.Option(None, "--host", help="Bind host (default from [server].host)"),
    port:   int  = typer.Option(None, "--port", "-p", help="Bind port (default from [server].port)"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes (dev)"),
):
    """Run the HTTP API (FastAPI on uvicorn).

    DATABASE_URL must point at a Postgres instance. Workers can run in
    the same process via `typoon work --role full` or on a separate
    host (vision/llm) sharing the same DB.
    """
    import logging
    import uvicorn
    from ..config import load_config
    cfg, _ = load_config()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    uvicorn.run(
        "typoon.api.app:app",
        host=host or cfg.server.host,
        port=port or cfg.server.port,
        reload=reload,
        # Long-lived SSE streams keep connections open from uvicorn's
        # perspective; without a graceful timeout Ctrl+C hangs.
        timeout_graceful_shutdown=5,
    )


# ── work ──────────────────────────────────────────────────────────────


@app.command()
def work(
    role:        str = typer.Option("full", "--role", "-r",
                                    help="vision | llm | api | storage | full"),
    concurrency: int = typer.Option(3, "--concurrency", "-c",
                                    help="Translate workers (only used when role=llm/full)"),
):
    """Start pipeline workers for a deployment role.

    full     everything in-process (default; dev on Mac)
    vision   prepare + scan + render (GPU node)
    llm      prepare + translate (LLM I/O node)
    api      no worker loops (API server only)
    storage  no worker loops; storage role lives in the API process
    """
    asyncio.run(_work(role, concurrency))


async def _work(role: str, concurrency: int) -> None:
    import logging
    from ..workers.loop import Role, run_workers
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        role_enum = Role(role)
    except ValueError:
        console.print(
            f"[red]invalid role: {role}[/] — use vision|llm|api|storage|full",
        )
        raise typer.Exit(1)
    await run_workers(role_enum, translate_concurrency=concurrency)


# ── version ───────────────────────────────────────────────────────────


@app.command()
def version():
    """Print SCHEMA_VERSION + git SHA for diagnostic logs."""
    import subprocess
    from ..storage.postgres import SCHEMA_VERSION
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
        ).strip()
    except Exception:
        sha = "unknown"
    console.print(f"typoon schema={SCHEMA_VERSION} commit={sha}")


# ── stage (operator pause / resume) ───────────────────────────────────
#
# Pipeline stages auto-pause when a worker raises
# `OperatorActionRequired` — model not found, credential revoked,
# storage misconfigured. While paused, `claim_task` skips every task
# under that stage. These commands let the operator see WHO paused
# WHAT and resume once the underlying config is fixed.

stage = typer.Typer(help="Inspect and control paused pipeline stages.")
app.add_typer(stage, name="stage")


_STAGES = ("prepare", "scan", "translate", "render")


def _validate_stage(name: str) -> str:
    if name not in _STAGES:
        console.print(
            f"[red]invalid stage: {name}[/] — use one of "
            f"{', '.join(_STAGES)}"
        )
        raise typer.Exit(1)
    return name


@stage.command("list")
def stage_list():
    """Show every stage the workers have paused, with reason and
    timestamp. Empty output = nothing paused, workers are healthy."""
    asyncio.run(_stage_list())


async def _stage_list() -> None:
    from ..storage import PostgresStore
    from ..config import load_config
    cfg, _ = load_config()
    db = await PostgresStore.open(cfg.database_url)
    try:
        rows = await db.list_paused_stages()
    finally:
        await db.close()
    if not rows:
        console.print("[green]no stages paused[/]")
        return
    for r in rows:
        console.print(
            f"[yellow]{r['stage']:<10}[/] "
            f"paused_at={r['paused_at']:%Y-%m-%d %H:%M:%S} "
            f"by={r['paused_by'] or '-'}"
        )
        console.print(f"  reason: {r['reason']}")


@stage.command("pause")
def stage_pause(
    name:   str = typer.Argument(..., help="prepare | scan | translate | render"),
    reason: str = typer.Option(..., "--reason", "-r",
                                help="Why workers should stop claiming this stage"),
):
    """Manually pause a stage. Workers finish in-flight tasks then
    stop claiming new ones until `stage resume`."""
    _validate_stage(name)
    asyncio.run(_stage_pause(name, reason))


async def _stage_pause(name: str, reason: str) -> None:
    import getpass
    from ..storage import PostgresStore
    from ..config import load_config
    cfg, _ = load_config()
    paused_by = f"cli:{getpass.getuser()}"
    db = await PostgresStore.open(cfg.database_url)
    try:
        inserted = await db.pause_stage(name, reason=reason, paused_by=paused_by)
    finally:
        await db.close()
    if inserted:
        console.print(f"[yellow]paused[/] stage={name} by={paused_by}")
    else:
        console.print(f"stage={name} was already paused (no change)")


@stage.command("resume")
def stage_resume(
    name: str = typer.Argument(..., help="prepare | scan | translate | render"),
):
    """Lift a pause. Workers wake up on the resume NOTIFY and start
    claiming again immediately — no restart required."""
    _validate_stage(name)
    asyncio.run(_stage_resume(name))


async def _stage_resume(name: str) -> None:
    from ..storage import PostgresStore
    from ..config import load_config
    cfg, _ = load_config()
    db = await PostgresStore.open(cfg.database_url)
    try:
        deleted = await db.resume_stage(name)
    finally:
        await db.close()
    if deleted:
        console.print(f"[green]resumed[/] stage={name}")
    else:
        console.print(f"stage={name} was not paused (no change)")
