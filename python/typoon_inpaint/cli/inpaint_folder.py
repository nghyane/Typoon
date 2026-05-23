"""CLI: inpaint a folder of images (parallel)."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import click

from typoon_inpaint.artifact_sink import FileArtifactSink  # type: ignore[attr-defined]

log = logging.getLogger("typoon.inpaint-folder")


@click.command("inpaint-folder")
@click.argument("folder", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--model", "model_path",
              type=click.Path(path_type=Path),
              envvar="TYPOON_MODEL", required=True)
@click.option("--out", type=click.Path(path_type=Path), default=None,
              help="Output directory. Default: <folder>/../inpaint-out/")
@click.option("--concurrency", default=4, show_default=True)
@click.option("--run-id", default="local-batch", show_default=True)
@click.option("--lang", default="ja", show_default=True)
def main(
    folder:      Path,
    model_path:  Path,
    out:         Path | None,
    concurrency: int,
    run_id:      str,
    lang:        str,
) -> None:
    """Inpaint every image in a folder."""
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname).1s %(name)s: %(message)s")
    asyncio.run(_run(folder, model_path, out, concurrency, run_id, lang))


async def _run(
    folder:      Path,
    model_path:  Path,
    out_dir:     Path | None,
    concurrency: int,
    run_id:      str,
    lang:        str,
) -> None:
    from typoon_inpaint import InpaintRuntime  # type: ignore[import]
    from typoon_inpaint.scan import build_plan_for_image

    exts   = {".jpg", ".jpeg", ".png", ".webp"}
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)
    if not images:
        raise click.ClickException(f"No images found in {folder}")

    out_dir = out_dir or (folder.parent / "inpaint-out")
    out_dir.mkdir(parents=True, exist_ok=True)

    sink = FileArtifactSink(Path("debug-runs") / run_id)
    rt   = InpaintRuntime(str(model_path))   # warm once
    sem  = asyncio.Semaphore(concurrency)

    async def one(i: int, path: Path) -> None:
        async with sem:
            page_sink  = sink.subdir(f"page_{i:04d}")
            plan_bytes = await build_plan_for_image(path, lang=lang,
                                                     sink=page_sink.subdir("scan"))
            jpeg = path.read_bytes()
            png: bytes = await rt.inpaint_page_async(
                jpeg, plan_bytes,
                debug_dir=str(page_sink.path),
            )
            (out_dir / f"{i:04d}.png").write_bytes(bytes(png))
            log.info("page %04d done", i)

    await asyncio.gather(*[one(i, p) for i, p in enumerate(images)])
    log.info("done. output: %s", out_dir)


if __name__ == "__main__":
    main()
