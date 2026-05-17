"""Single-page full-pipeline probe — detect → group → translate → erase → render.

Run on a single PNG/JPG/WEBP file without DB or chapter context. Wires
the production stage functions directly (`scan_chapter`, `translate_chapter`,
`render_chapter`) against an in-memory `PreparedReader` shim, so the
output mirrors what the worker produces for a one-page chapter.

Run:
    python -m scripts.probes.full_page <image> [--out debug-runs/<name>]
                                       [--source-lang ja] [--target-lang vi]

Outputs under <out>/ (FileArtifactSink layout):
    01_prepare/source.png
    02_detect/page_0000_detect.png + state.json
    03_group/page_0000_groups.png
    04_ocr/ocr_all_pages.json
    05_brief/*.txt + parsed.json
    06_translate/w00_*.txt + parsed.json
    07_render/0000_inpainted.png  + 0000_rendered.png
    final/0000.jpg
"""
