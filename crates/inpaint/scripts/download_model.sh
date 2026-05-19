# SPDX-License-Identifier: GPL-3.0-or-later
#! /usr/bin/env bash
# Download model.safetensors from HuggingFace for local dev / CI.
# Usage: scripts/download_model.sh [output_dir]
set -euo pipefail
OUT="${1:-$(dirname "$0")}"
URL="https://huggingface.co/mayocream/aot-inpainting/resolve/main/model.safetensors"
DEST="$OUT/model.safetensors"
if [ -f "$DEST" ]; then
  echo "already exists: $DEST"
  exit 0
fi
echo "downloading $URL → $DEST"
curl -L --progress-bar -o "$DEST" "$URL"
echo "done ($(du -sh "$DEST" | cut -f1))"
