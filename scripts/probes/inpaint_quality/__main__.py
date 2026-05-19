"""inpaint_quality — batch visual quality eval.

Runs full_page probe on each fixture, collects inpainted + final output,
assembles HTML report at debug-runs/inpaint_quality/report.html.

Usage:
    python -m scripts.probes.inpaint_quality [--images dir] [--out dir]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path
from textwrap import dedent

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for pkg in ("typoon-core", "typoon-vision", "typoon-stages", "typoon-app", "typoon-llm",
            "typoon-translate", "typoon-storage"):
    p = ROOT / "packages" / pkg
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

logging.basicConfig(level=logging.WARNING, format="%(levelname).1s %(name)s: %(message)s")
log = logging.getLogger("inpaint_quality")


# ─── image fixtures ───────────────────────────────────────────────────────────

def collect_images(images_dir: Path) -> list[tuple[str, Path]]:
    """Return (lang_hint, path) pairs."""
    fixtures = []
    # ja fixtures
    for p in sorted((ROOT / "tests/fixtures/sample_chapters/ch001").glob("*.webp")):
        fixtures.append(("ja", p))
    # mangadex eval dir
    if images_dir.exists():
        for p in sorted(images_dir.glob("*.*")):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".avif"):
                lang = "ja" if "ja" in p.stem else "zh" if "zh" in p.stem else "ja"
                fixtures.append((lang, p))
    return fixtures


# ─── run one page ─────────────────────────────────────────────────────────────


async def run_page(lang: str, image_path: Path, out_dir: Path) -> dict:
    """Run full pipeline (no-translate for speed). Return result dict."""
    from dotenv import load_dotenv
    load_dotenv(override=False)

    from typoon.config import load_config
    from typoon.vision.pipeline import PRESETS
    from typoon.vision.runtime import build_vision_runtime
    from typoon.domain import translate as t_dom
    from typoon.domain.brief import ChapterBrief
    from typoon.stages.keys import assign_keys
    from typoon.stages.noise import is_auto_skip
    from typoon.stages.scan import scan_chapter
    from typoon.stages.render import render_chapter
    from typoon.stages.translate import _normalize_for_render
    from typoon.runs.artifacts import FileArtifactSink

    sys.path.insert(0, str(ROOT / "scripts/probes/full_page"))
    from reader import SinglePageReader

    config, paths = load_config()
    models_dir = Path(config.models_dir).resolve()
    run_name = f"{image_path.stem}_{lang}"

    sink = FileArtifactSink(out_dir, run_name, clean=True)
    reader = SinglePageReader.from_path(image_path)
    runtime = build_vision_runtime(
        PRESETS["lens"], models_dir=models_dir,
        source_lang=lang,
        lens_endpoint=config.lens_endpoint or None,
    )

    t0 = time.monotonic()
    prepared = reader.chapter(source=str(image_path))
    scanned = await scan_chapter(prepared, reader, runtime,
                                 source_lang=lang, chapter_id=0, artifacts=sink)
    scan_t = time.monotonic() - t0
    n_bubbles = sum(len(p.bubbles) for p in scanned.chapter.pages)

    # stub translate
    keyed = assign_keys(scanned.chapter.all_bubbles, chapter_id=0)
    key_at = {(bk.bubble.page_index, bk.bubble.idx): bk.key for bk in keyed}
    pages = []
    for sp in scanned.chapter.pages:
        bubbles = []
        for sb in sp.bubbles:
            key = key_at.get((sb.page_index, sb.idx), f"b{sb.idx}")
            kind = "skip" if is_auto_skip(sb.source_text) else "dialogue"
            text = "" if kind == "skip" else _normalize_for_render(f"[vi] {sb.source_text}")
            bubbles.append(t_dom.Bubble(source=sb, translation_key=key,
                                        translated_text=text, kind=kind))
        pages.append(t_dom.Page(source=sp, bubbles=tuple(bubbles)))
    translated = t_dom.Chapter(scan=scanned, pages=tuple(pages))
    brief = ChapterBrief(brief_prose="", noise_keys=set())

    t1 = time.monotonic()
    final_dir = sink.root / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    page_geoms = {pg.page_index: pg for pg in scanned.geometry}
    await render_chapter(translated, final_dir, reader, runtime,
                         page_geoms, scanned.masks, chapter_id=0,
                         target_kind="draft", target_id=0, artifacts=sink)
    render_t = time.monotonic() - t1

    inpainted = sink.root / "07_render" / "0000_inpainted.png"
    final_jpg  = final_dir / "0000.jpg"

    return {
        "name":       run_name,
        "lang":       lang,
        "image":      str(image_path),
        "bubbles":    n_bubbles,
        "scan_t":     f"{scan_t:.1f}s",
        "render_t":   f"{render_t:.2f}s",
        "inpainted":  str(inpainted) if inpainted.exists() else "",
        "final":      str(final_jpg) if final_jpg.exists() else "",
        "run_dir":    str(sink.root),
    }


# ─── HTML report ──────────────────────────────────────────────────────────────


def img_b64(path: str) -> str:
    if not path or not Path(path).exists():
        return ""
    with open(path, "rb") as f:
        import base64
        data = base64.b64encode(f.read()).decode()
    ext = Path(path).suffix.lstrip(".")
    mime = "jpeg" if ext in ("jpg", "jpeg") else "png"
    return f"data:image/{mime};base64,{data}"


def write_report(results: list[dict], out_dir: Path) -> Path:
    rows = ""
    for r in results:
        inp_b64 = img_b64(r["inpainted"])
        fin_b64 = img_b64(r["final"])
        inp_tag = f'<img src="{inp_b64}" style="max-height:400px">' if inp_b64 else "missing"
        fin_tag = f'<img src="{fin_b64}" style="max-height:400px">' if fin_b64 else "missing"
        rows += f"""
        <tr>
          <td><b>{r['name']}</b><br>{r['lang']} · {r['bubbles']} bubbles<br>
              scan {r['scan_t']} · render {r['render_t']}</td>
          <td>{inp_tag}</td>
          <td>{fin_tag}</td>
        </tr>"""

    html = dedent(f"""
    <!DOCTYPE html><html><head>
    <meta charset=utf-8>
    <title>Inpaint Quality Report</title>
    <style>
      body {{ font-family: sans-serif; font-size: 13px; background: #111; color: #eee; }}
      table {{ border-collapse: collapse; width: 100%; }}
      td {{ border: 1px solid #333; padding: 8px; vertical-align: top; }}
      td:first-child {{ width: 200px; }}
      img {{ display: block; border: 1px solid #555; }}
      h1 {{ color: #aef; }}
    </style></head><body>
    <h1>Inpaint Quality — {len(results)} pages</h1>
    <table>
      <thead><tr><th>Page</th><th>Inpainted (before glyph)</th><th>Final (with VI text)</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
    </body></html>
    """).strip()

    report = out_dir / "report.html"
    report.write_text(html)
    return report


# ─── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path,
                    default=ROOT / "tests/fixtures/inpaint_eval")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "debug-runs/inpaint_quality")
    ap.add_argument("--limit", type=int, default=8,
                    help="max pages to process (default 8)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    fixtures = collect_images(args.images)[:args.limit]
    print(f"Processing {len(fixtures)} pages...")

    results = []
    for i, (lang, path) in enumerate(fixtures):
        print(f"\n[{i+1}/{len(fixtures)}] {path.name} ({lang})")
        try:
            r = asyncio.run(run_page(lang, path, args.out))
            results.append(r)
            print(f"  bubbles={r['bubbles']} scan={r['scan_t']} render={r['render_t']}")
        except Exception as e:
            log.error("failed %s: %s", path.name, e, exc_info=True)
            results.append({"name": path.stem, "lang": lang, "image": str(path),
                            "bubbles": 0, "scan_t": "err", "render_t": "err",
                            "inpainted": "", "final": "", "run_dir": ""})

    report = write_report(results, args.out)
    print(f"\nreport → {report}")


if __name__ == "__main__":
    main()
