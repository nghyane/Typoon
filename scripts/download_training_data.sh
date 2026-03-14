#!/bin/bash
# Download training data for MI-GAN distillation.
# Diverse mix of manga (B/W) + manhwa (color) for inpainting model.
#
# Usage:
#   bash scripts/download_training_data.sh          # download all
#   bash scripts/download_training_data.sh manga    # manga only
#   bash scripts/download_training_data.sh manhwa   # manhwa only
#
# Prerequisites:
#   - Valid cookies in cache/comix_cookies.txt
#   - pip install httpx websockets

set -e
cd "$(dirname "$0")/.."

FILTER="${1:-all}"
CONCURRENCY=10

echo "=== MI-GAN Training Data Download ==="
echo "Filter: $FILTER"
echo ""

# ── Manga (B/W Japanese) ──────────────────────────────────────────
download_manga() {
    local series=(
        "7nzg-jujutsu-kaisen"        # 271ch, action, heavy SFX
        "69l57-chainsaw-man"         # 231ch, action, gore, SFX
        "5vvzz-blue-lock"            # 339ch, sports, dynamic panels
        "pvry-one-piece"             # 1176ch, classic manga, all styles
        "n9z0-gachiakuta"            # 162ch, gritty, varied backgrounds
    )

    for slug in "${series[@]}"; do
        name="${slug#*-}"  # strip prefix
        echo "── Manga: $name ──"
        python3 scripts/comix_download.py "$slug" \
            -o "data/training/manga/$name" \
            -c "$CONCURRENCY" -t 2 || echo "  ⚠ Failed: $slug"
        echo ""
    done
}

# ── Manhwa (Color Korean) ─────────────────────────────────────────
download_manhwa() {
    local series=(
        "w2399-omniscient-readers-viewpoint"  # 304ch, action, diverse panels
        "emqg8-solo-leveling"                 # 200ch, action, SFX heavy
        "xlyyj-eleceed"                       # 392ch, action, clean art
        "grmvg-nano-machine"                  # 303ch, martial arts
        "0j5d-star-embracing-swordmaster"     # 113ch, action, detailed art
    )

    for slug in "${series[@]}"; do
        name="${slug#*-}"
        echo "── Manhwa: $name ──"
        python3 scripts/comix_download.py "$slug" \
            -o "data/training/manhwa/$name" \
            -c "$CONCURRENCY" -t 2 || echo "  ⚠ Failed: $slug"
        echo ""
    done
}

case "$FILTER" in
    manga)  download_manga ;;
    manhwa) download_manhwa ;;
    all)    download_manga; download_manhwa ;;
    *)      echo "Usage: $0 [all|manga|manhwa]"; exit 1 ;;
esac

echo "=== Download complete ==="
echo "Run: python3 scripts/data_stats.py data/training"
