#!/bin/bash
set -e

# ============================================================
# MI-GAN Distillation — Setup & Train (Ubuntu + CUDA)
# ============================================================
# Usage:
#   ./setup_and_train.sh                          # defaults
#   ./setup_and_train.sh --data_dir /path/to/imgs # custom data
#   ./setup_and_train.sh --batch_size 32          # override any train.py arg
#
# Prerequisites: Python 3.10+, NVIDIA GPU with CUDA 12.x drivers
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/.venv"
DATA_DIR="${DATA_DIR:-../../data/training}"
LAMA_MODEL="${LAMA_MODEL:-../../models/lama_fp32.onnx}"
OUTPUT_DIR="${OUTPUT_DIR:-../../runs/migan_distill}"

# ---- Colors ----
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# ---- Check prerequisites ----
info "Checking prerequisites..."

command -v python3 >/dev/null 2>&1 || error "python3 not found. Install: sudo apt install python3 python3-venv"
nvidia-smi >/dev/null 2>&1 || error "nvidia-smi not found. Install NVIDIA drivers first."

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
info "GPU: $GPU_NAME (${GPU_MEM}MB VRAM)"

# ---- Auto batch size based on VRAM ----
if [ -z "$BATCH_SIZE_OVERRIDE" ]; then
    if [ "$GPU_MEM" -ge 20000 ]; then
        AUTO_BATCH=16
    elif [ "$GPU_MEM" -ge 10000 ]; then
        AUTO_BATCH=8
    elif [ "$GPU_MEM" -ge 6000 ]; then
        AUTO_BATCH=4
    else
        AUTO_BATCH=2
    fi
    info "Auto batch size: $AUTO_BATCH (based on ${GPU_MEM}MB VRAM)"
fi

# ---- Setup venv ----
if [ ! -d "$VENV_DIR" ]; then
    info "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# ---- Install dependencies ----
if ! python -c "import torch" 2>/dev/null; then
    info "Installing PyTorch with CUDA..."
    pip install --upgrade pip
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt
    info "Dependencies installed."
else
    TORCH_CUDA=$(python -c "import torch; print(torch.cuda.is_available())")
    if [ "$TORCH_CUDA" = "False" ]; then
        warn "PyTorch installed but CUDA not available. Reinstalling..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
    fi
    info "Dependencies already installed."
fi

# ---- Verify CUDA ----
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')
" || error "CUDA verification failed"

# ---- Check data ----
if [ ! -d "$DATA_DIR" ]; then
    error "Data directory not found: $DATA_DIR\nPut training images there or set DATA_DIR=/path/to/images"
fi

IMG_COUNT=$(find "$DATA_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.webp" \) | wc -l)
info "Training images: $IMG_COUNT in $DATA_DIR"

if [ "$IMG_COUNT" -eq 0 ]; then
    error "No images found in $DATA_DIR"
fi

# ---- Check LaMa model ----
if [ ! -f "$LAMA_MODEL" ]; then
    error "LaMa model not found: $LAMA_MODEL\nDownload or set LAMA_MODEL=/path/to/lama_fp32.onnx"
fi

LAMA_SIZE=$(du -h "$LAMA_MODEL" | cut -f1)
info "LaMa teacher: $LAMA_MODEL ($LAMA_SIZE)"

# ---- Train ----
info "Starting training..."
echo "============================================"

BATCH_SIZE="${BATCH_SIZE_OVERRIDE:-$AUTO_BATCH}"

python train.py \
    --data_dir "$DATA_DIR" \
    --lama_model "$LAMA_MODEL" \
    --batch_size "$BATCH_SIZE" \
    --num_workers 4 \
    --use_amp \
    --total_kimg 500 \
    --lr 1e-3 \
    --kd_weight 2.0 \
    --r1_gamma 10.0 \
    --log_every 50 \
    --vis_every 200 \
    --save_every 2000 \
    --output_dir "$OUTPUT_DIR" \
    "$@"
