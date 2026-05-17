#!/usr/bin/env bash
# Probe a single image URL (or local path) through the full pipeline and
# dump per-group fit traces + crops into debug-runs/<name>/full_pipeline/.
#
# Usage:
#   scripts/probe_image.sh <url-or-path> [fixture-name]
#   scripts/probe_image.sh https://.../01.webp           # → lens_bubble_probe3
#   scripts/probe_image.sh /tmp/foo.png my_fixture       # → my_fixture
#
# Sets up debug-runs/<fixture-name>/source.png from the URL/path, then
# runs scripts/poc_full_pipeline.py for that fixture. Outputs:
#   debug-runs/<name>/full_pipeline/06_fit_debug.png   page overlay
#   debug-runs/<name>/full_pipeline/groups/NN_path.png per-group crops
#   stdout: per-group trace (Lens members, Comic-DETR overlaps, path)

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <url-or-path> [fixture-name]" >&2
  exit 1
fi

SRC="$1"
NAME="${2:-lens_bubble_probe3}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/debug-runs/$NAME"
mkdir -p "$OUT"

if [[ "$SRC" =~ ^https?:// ]]; then
  EXT="${SRC##*.}"
  TMP="$(mktemp -t probe_img.XXXXXX).${EXT:-png}"
  echo ">> downloading $SRC"
  curl -fsSL "$SRC" -o "$TMP"
else
  TMP="$SRC"
fi

echo ">> converting → $OUT/source.png"
"$ROOT/.venv/bin/python" - "$TMP" "$OUT/source.png" <<'PY'
import sys
from PIL import Image
src, dst = sys.argv[1], sys.argv[2]
img = Image.open(src).convert("RGB")
img.save(dst)
print(f"   size={img.size}")
PY

echo ">> running poc_full_pipeline $NAME"
cd "$ROOT"
"$ROOT/.venv/bin/python" scripts/poc_full_pipeline.py "$NAME"
