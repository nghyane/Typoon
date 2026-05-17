"""Lens grouping probe — single overview image per page.

Run:
    python -m scripts.probes.lens_group <image> [--out debug-runs/<name>]

Outputs under <out>/:
    source.png         — input
    raw.json           — Lens blocks + comic_detr regions
    overview.png       — 2×2 grid:
        TL: Lens raw (word ⊂ line ⊂ paragraph + rotation)
        TR: Comic-DETR regions (3 classes)
        BL: final containers per group (what render receives as polygon)
        BR: erase mask overlay (what AOT inpaint sees)
"""
