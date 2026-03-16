#!/bin/bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════
# SVTRv2 Japanese manga OCR — finetune from Chinese pretrained
# Run LOCALLY. Connects to RunPod pod via SSH.
#
# Usage:
#   bash runpod_openocr.sh <SSH_HOST> <SSH_PORT> [SSH_KEY]
#   e.g.: bash runpod_openocr.sh root@213.173.99.4 19118 ~/.ssh/id_ed25519
# ═══════════════════════════════════════════════════════════════

if [ $# -lt 2 ]; then
    echo "Usage: bash runpod_openocr.sh <SSH_HOST> <SSH_PORT> [SSH_KEY]"
    exit 1
fi

SSH_HOST="$1"
SSH_PORT="$2"
SSH_KEY="${3:-$HOME/.ssh/id_ed25519}"
SSH_CMD="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p $SSH_PORT -i $SSH_KEY $SSH_HOST"
SCP_CMD="scp -o StrictHostKeyChecking=no -P $SSH_PORT -i $SSH_KEY"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "==> [1/5] Uploading data + config..."
# Recompress data if needed (only crops + labels + dict, no tar inside tar)
cd "$SCRIPT_DIR"
if [ ! -f data_upload.tar.gz ] || [ data/train.txt -nt data_upload.tar.gz ]; then
    echo "  Compressing data/..."
    tar czf data_upload.tar.gz -C data crops train.txt val.txt ja_dict.txt
fi
DATA_SIZE=$(du -sh data_upload.tar.gz | cut -f1)
echo "  Uploading data ($DATA_SIZE)..."
$SCP_CMD data_upload.tar.gz "$SSH_HOST:/workspace/"
$SCP_CMD openocr_finetune_ja.yml "$SSH_HOST:/workspace/"

echo "==> [2/5] Setting up environment..."
$SSH_CMD << 'SETUP_EOF'
set -euo pipefail
cd /workspace

# Extract data
echo "  Extracting data..."
mkdir -p data
tar xzf data_upload.tar.gz -C data/
rm data_upload.tar.gz

# Verify data
TRAIN_LINES=$(wc -l < data/train.txt)
VAL_LINES=$(wc -l < data/val.txt)
DICT_LINES=$(wc -l < data/ja_dict.txt)
echo "  Data: train=${TRAIN_LINES} val=${VAL_LINES} dict=${DICT_LINES}"
BAD=$(grep -cv $'\t' data/train.txt || true)
if [ "$BAD" -gt 0 ]; then
    echo "  ERROR: ${BAD} lines without tab separator!"
    exit 1
fi

# Fix numpy for imgaug compatibility
pip install 'numpy<2' -q 2>/dev/null

# Clone OpenOCR
if [ ! -d OpenOCR ]; then
    echo "  Cloning OpenOCR..."
    git clone --depth 1 https://github.com/Topdu/OpenOCR.git
fi

# Download pretrained Chinese model
if [ ! -f openocr_repsvtr_ch.pth ] || [ $(stat -c%s openocr_repsvtr_ch.pth 2>/dev/null || echo 0) -lt 1000000 ]; then
    echo "  Downloading pretrained Chinese SVTRv2..."
    wget -q 'https://github.com/Topdu/OpenOCR/releases/download/develop0.0.1/openocr_repsvtr_ch.pth' -O openocr_repsvtr_ch.pth
fi
CKPT_SIZE=$(du -sh openocr_repsvtr_ch.pth | cut -f1)
echo "  Pretrained: ${CKPT_SIZE}"

echo "  Setup complete!"
SETUP_EOF

echo "==> [3/5] Dry-run: verify model + config on CPU..."
$SSH_CMD << 'DRYRUN_EOF'
set -euo pipefail
cd /workspace/OpenOCR

python3 -c "
import sys, yaml, torch

cfg = yaml.safe_load(open('/workspace/openocr_finetune_ja.yml'))
print('[OK] Config loaded')

# Load pretrained
ckpt = torch.load('/workspace/openocr_repsvtr_ch.pth', map_location='cpu', weights_only=True)
state = ckpt
print(f'[OK] Pretrained: {len(state)} keys')

# Dict size
ja_dict = open('/workspace/data/ja_dict.txt').read().strip().split('\n')
ja_classes = len(ja_dict) + 2  # +blank +space
print(f'[OK] JA dict: {len(ja_dict)} chars -> {ja_classes} classes')

# Build model
sys.path.insert(0, '.')
from openrec.modeling.architectures.rec_model import RecModel
cfg['Architecture']['Decoder']['out_channels'] = ja_classes
model = RecModel(cfg['Architecture'])
model_state = model.state_dict()

# Check weight compatibility
matched = mismatched = missing = 0
for k in model_state:
    if k in state:
        if model_state[k].shape == state[k].shape:
            matched += 1
        else:
            mismatched += 1
            print(f'  RESET: {k} model={list(model_state[k].shape)} ckpt={list(state[k].shape)}')
    else:
        missing += 1
        if 'num_batches' not in k:
            print(f'  NEW: {k}')

total = len(model_state)
print(f'[OK] Weight compatibility: {matched}/{total} matched, {mismatched} reset, {missing} new')

if matched < total * 0.3:
    print('[FAIL] Too few matched weights!')
    sys.exit(1)

# Test partial load (like trainer does)
loaded = {k: v for k, v in state.items() if k in model_state and model_state[k].shape == v.shape}
model.load_state_dict(loaded, strict=False)
print(f'[OK] Partial load: {len(loaded)} weights transferred')

# Forward pass with dummy data
model.eval()
dummy = torch.randn(1, 3, 48, 320)
with torch.no_grad():
    out = model(dummy)
print(f'[OK] Forward pass: output keys={list(out.keys()) if isinstance(out, dict) else type(out)}')

# Verify loss
from openrec.losses import build_rec_loss
loss_cls = build_rec_loss(cfg['Loss'])
print(f'[OK] Loss: {type(loss_cls).__name__}')

print()
print('=== DRY-RUN PASSED ===')
"
DRYRUN_EOF

if [ $? -ne 0 ]; then
    echo "DRY-RUN FAILED! Fix config before training."
    exit 1
fi

echo "==> [4/5] Training (finetune, ~1h on RTX GPU)..."
$SSH_CMD << 'TRAIN_EOF'
set -euo pipefail
cd /workspace/OpenOCR

# Run in tmux so training survives SSH disconnect
tmux new-session -d -s train "python tools/train_rec.py -c /workspace/openocr_finetune_ja.yml 2>&1 | tee /workspace/train.log" || {
    tmux send-keys -t train "python tools/train_rec.py -c /workspace/openocr_finetune_ja.yml 2>&1 | tee /workspace/train.log" Enter
}

echo "  Training started in tmux session 'train'"
echo "  Monitor: ssh -p SSH_PORT SSH_HOST 'tmux attach -t train'"

# Wait for first eval to confirm training works
echo "  Waiting for first log output..."
for i in $(seq 1 30); do
    sleep 10
    if [ -f /workspace/train.log ] && grep -q "global_step:" /workspace/train.log 2>/dev/null; then
        tail -5 /workspace/train.log
        echo "  Training confirmed running!"
        exit 0
    fi
    if [ -f /workspace/train.log ] && grep -q "Error\|Traceback" /workspace/train.log 2>/dev/null; then
        echo "  TRAINING FAILED:"
        tail -20 /workspace/train.log
        exit 1
    fi
done
echo "  Timeout waiting for training output. Check manually."
TRAIN_EOF

echo ""
echo "════════════════════════════════════════"
echo "  Training running on pod."
echo "  Monitor: $SSH_CMD 'tmux attach -t train'"
echo ""
echo "  When done, export ONNX:"
echo "    cd /workspace/OpenOCR"
echo "    python tools/toonnx.py -c /workspace/openocr_finetune_ja.yml \\"
echo "      -o Global.device=cpu Global.pretrained_model=/workspace/output/svtrv2_ja_ft/best.pth"
echo ""
echo "  Download:"
echo "    $SCP_CMD $SSH_HOST:/workspace/OpenOCR/output/rec/*/export_rec/rec_model.onnx $PROJECT_DIR/models/svtrv2_rec_ja.onnx"
echo "════════════════════════════════════════"
