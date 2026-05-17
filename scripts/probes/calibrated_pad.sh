#!/usr/bin/env bash
# Run calibrated-pad probe across all current samples.
# Probe-only — no production logic touched.
#
# Usage:
#   scripts/probes/calibrated_pad.sh                 # all samples
#   scripts/probes/calibrated_pad.sh path/to/img.png # one image

set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ $# -gt 0 ]]; then
  SOURCES=("$@")
else
  SOURCES=()
  for s in \
    debug-runs/sample/happymh.png \
    debug-runs/sample/happymh2.png \
    debug-runs/sample/happymh3.png \
    debug-runs/sample/wowpic1.png \
    debug-runs/mangabuzz_374_1_9/source.webp \
  ; do
    [[ -f "$s" ]] && SOURCES+=("$s")
  done
fi

for src in "${SOURCES[@]}"; do
  name=$(basename "$src" | sed 's/\.[^.]*$//')
  parent=$(basename "$(dirname "$src")")
  if [[ "$parent" = "sample" ]]; then
    tag="$name"
  else
    tag="${parent}_${name}"
  fi
  out="debug-runs/calib_${tag}"
  echo "── $src → $out ───────────────────────────"
  python -m scripts.probes.calibrated_pad "$src" --out "$out" 2>&1 \
    | grep -vE 'coreml|onnxruntime|chrome_lens|httpx|bubble_pass|^I |^W |INFO|WARN|empty bubbles' \
    || true
  echo
done
