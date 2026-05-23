"""CLI: inpaint a single image."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import click

from typoon_inpaint_py.artifact_sink import FileArtifactSink  # type: ignore[attr-defined]
from typoon_inpaint_py.storage import LocalFsStorage


log = logging.getLogger("typoon.inpaint-one")


@click.command("inpaint-one")
@click.argument("image", type=click.Path(exists=True, path_type=Path))
@click.option("--plan", "scan_path",
              type=click.Path(exists=True, path_type=Path),
              help="Pre-computed scan msgpack (with embedded InpaintPlan). Skip scan when provided.")
@click.option("--model", "model_path",
              type=click.Path(path_type=Path),
              envvar="TYPOON_MODEL",
              default=None,
              help="Path to model.safetensors. Omit to use stub (mask only).")
@click.option("--run-id", default="local-001", show_default=True)
@click.option("--out",    type=click.Path(path_type=Path), default=None)
@click.option("--lang",   default="ja", show_default=True)
@click.option("--debug-dir", "debug_dir",
              type=click.Path(path_type=Path), default=None,
              help="Override artifact output directory.")
def main(
    image:      Path,
    scan_path:  Path | None,
    model_path: Path | None,
    run_id:     str,
    out:        Path | None,
    lang:       str,
    debug_dir:  Path | None,
) -> None:
    """Inpaint a single image locally."""
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname).1s %(name)s: %(message)s")
    asyncio.run(_run(image, scan_path, model_path, run_id, out, lang, debug_dir))


async def _run(
    image:      Path,
    scan_path:  Path | None,
    model_path: Path | None,
    run_id:     str,
    out:        Path | None,
    lang:       str,
    debug_dir:  Path | None,
) -> None:
    from typoon_inpaint import InpaintRuntime  # type: ignore[import]

    sink = FileArtifactSink(
        debug_dir or (Path("debug-runs") / run_id / "05_inpaint")
    )

    if scan_path is None:
        from typoon_inpaint_py.scan import build_plan_for_image
        log.info("scanning %s …", image.name)
        scan_bytes = await build_plan_for_image(image, lang=lang, sink=sink.subdir("scan"))
    else:
        scan_bytes = scan_path.read_bytes()
        log.info("using pre-computed scan: %s", scan_path)

    if model_path is None:
        raise click.ClickException("--model / TYPOON_MODEL required")

    rt   = InpaintRuntime(str(model_path))
    jpeg = image.read_bytes()
    log.info("inpainting …")
    png: bytes = await rt.inpaint_page_async(
        jpeg, scan_bytes,
        debug_dir=str(sink.path),
    )

    out_path = out or image.with_suffix(".inpaint.png")
    out_path.write_bytes(bytes(png))
    log.info("OK → %s", out_path)
    log.info("artifacts: %s", sink.path)


if __name__ == "__main__":
    main()
