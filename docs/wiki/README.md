# Typoon v2 Wiki

Read this before coding in a new thread.

## Pages

- [Architecture](architecture.md) — package layout, dependency rules, pipeline
- [Material architecture (planned)](material-architecture.md) — Phase B re-architect target; not yet implemented
- [Lens-native grouping](lens-native-grouping.md) — the `lens` preset detector + grouper; two-phase OCR, container vs mask geometry, rotation handling
- [Render archive storage & CDN](render-archive-storage.md) — multi-backend archive serving, bunle CDN, HF dataset
- [Cloudflare edge pipeline (feasibility)](cloudflare-edge-pipeline.md) — empirical findings on running the full pipeline on Workers + R2; reproducible probes under `spike/`
- [Browse mode](browse-mode.md) — source manifests, shelves, internal vs external, design rules
- [Reverse-engineering manga sources](reverse-engineering-manga-sources.md) — 30–60 min playbook to add a new browse-mode source manifest
- [Hard rules](hard-rules.md) — what is forbidden and why

## RFCs

- [`material-architecture`](../rfc/material-architecture.md) — Material + Translation entities; replaces Project

## Active handoff

- [Auto-merge multi-source chapter list](handoff-auto-merge-chapters.md) — drop `activeMaterial`/`?src=`, union manifest fetches across every installed source on a Work
- [Scan stage](handoff-scan-stage.md) — next task: `stages/scan.py`, RFC-003
