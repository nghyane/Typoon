#!/bin/bash
# Setup and train small manga inpainting model on RunPod
# Usage: bash setup_and_train_small.sh

set -e

echo "=== Installing dependencies ==="
pip install torch torchvision --quiet 2>/dev/null || true
pip install Pillow numpy tqdm tensorboard onnx onnxruntime --quiet 2>/dev/null || true

echo "=== Checking data ==="
DATA_DIR="${DATA_DIR:-/root/data/training/manga}"
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data dir not found: $DATA_DIR"
    echo "Upload manga images first, or set DATA_DIR env var"
    exit 1
fi
NUM_IMAGES=$(find "$DATA_DIR" -name "*.webp" -o -name "*.png" -o -name "*.jpg" | wc -l)
echo "Found $NUM_IMAGES images in $DATA_DIR"

echo "=== Starting training ==="
cd /root

python train_small.py \
    --data_dir "$DATA_DIR" \
    --base_ch 48 \
    --batch_size 16 \
    --lr 1e-3 \
    --total_kimg 2000 \
    --l1_weight 10.0 \
    --output_dir runs/manga_inpaint \
    --save_every 2000 \
    --num_workers 4

echo "=== Done! ==="
echo "Checkpoint: runs/manga_inpaint/checkpoints/best.pt"
echo "ONNX model: runs/manga_inpaint/manga_inpaint.onnx"
