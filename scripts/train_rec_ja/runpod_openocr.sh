#!/bin/bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════
# SVTRv2 Japanese manga OCR — finetune from Chinese pretrained
# Run LOCALLY. Connects to RunPod pod via SSH.
#
# Prerequisites on pod: PyTorch 2.x, Python 3.11, CUDA
# Everything else (data, deps, pretrained) is fetched automatically.
#
# Usage:
#   bash runpod_openocr.sh <SSH_HOST> <SSH_PORT> [SSH_KEY]
#   e.g.: bash runpod_openocr.sh root@82.221.170.242 24274 ~/.ssh/id_ed25519
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

echo "==> [1/4] Setting up environment on pod..."
$SSH_CMD << 'SETUP_EOF'
set -euo pipefail
cd /workspace

# ── Install dependencies ──
echo "  Installing deps..."
pip install -q 'numpy<2' --force-reinstall 2>/dev/null
pip install -q opencv-python-headless imgaug rapidfuzz 'huggingface_hub>=0.20' 2>/dev/null

# Verify numpy <2 (imgaug requires it)
python3 -c "import numpy; v=numpy.__version__; assert v.startswith('1.'), f'numpy {v} >= 2!'; print(f'  numpy {v} OK')"

# ── Clone repos ──
echo "  Cloning repos..."
[ -d Typoon ] || git clone --depth 1 https://github.com/nghyane/Typoon.git &
[ -d OpenOCR ] || git clone --depth 1 https://github.com/Topdu/OpenOCR.git &
wait

# ── Download pretrained Chinese SVTRv2 ──
if [ ! -f openocr_repsvtr_ch.pth ] || [ "$(stat -c%s openocr_repsvtr_ch.pth 2>/dev/null || echo 0)" -lt 1000000 ]; then
    echo "  Downloading pretrained Chinese SVTRv2..."
    curl -sL 'https://github.com/Topdu/OpenOCR/releases/download/develop0.0.1/openocr_repsvtr_ch.pth' -o openocr_repsvtr_ch.pth
fi
echo "  Pretrained: $(du -sh openocr_repsvtr_ch.pth | cut -f1)"

# ── Download dataset from HuggingFace ──
echo "  Downloading dataset..."
mkdir -p data && cd data
python3 -c "
from huggingface_hub import hf_hub_download
for f in ['data.tar.gz', 'train.txt', 'val.txt', 'ja_dict.txt']:
    hf_hub_download('nghyane/manga109-crops', f, repo_type='dataset', local_dir='.')
    print(f'    {f} OK')
"

# Extract crops (tar has data/ prefix from macOS)
echo "  Extracting crops..."
tar xzf data.tar.gz 2>/dev/null || true
[ -d data/crops ] && mv data/crops . 2>/dev/null || true
rm -rf data/ .huggingface/ 2>/dev/null || true

# ── Verify data ──
TRAIN_LINES=$(wc -l < train.txt)
VAL_LINES=$(wc -l < val.txt)
DICT_LINES=$(wc -l < ja_dict.txt)
CROP_COUNT=$(ls crops/ | wc -l)
BAD=$(grep -cv $'\t' train.txt || echo 0)
echo "  Data: train=${TRAIN_LINES} val=${VAL_LINES} dict=${DICT_LINES} crops=${CROP_COUNT} bad=${BAD}"
if [ "$BAD" -gt 0 ]; then
    echo "  ERROR: ${BAD} lines without tab separator!"
    exit 1
fi
if [ "$CROP_COUNT" -lt 1000 ]; then
    echo "  ERROR: Only ${CROP_COUNT} crops extracted!"
    exit 1
fi

# ── Patch OpenOCR: skip shape-mismatched keys when loading pretrained ──
# OpenOCR's load_pretrained_params uses strict=False but PyTorch still
# raises RuntimeError on shape mismatches. We filter them out first.
cd /workspace/OpenOCR
python3 -c "
text = open('tools/utils/ckpt.py').read()
old = '    model.load_state_dict(state_dict, strict=False)'
new = '''    # Filter out keys with shape mismatch (e.g. CTC head when finetuning with different dict)
    model_state = model.state_dict()
    filtered = {k: v for k, v in state_dict.items() if k in model_state and model_state[k].shape == v.shape}
    skipped = [k for k in state_dict if k not in filtered]
    if skipped:
        logger.info(f'Skipped {len(skipped)} pretrained keys (shape mismatch): {skipped}')
    logger.info(f'Loaded {len(filtered)}/{len(model_state)} pretrained weights')
    model.load_state_dict(filtered, strict=False)'''
if old in text:
    text = text.replace(old, new)
    open('tools/utils/ckpt.py', 'w').write(text)
    print('  Patched ckpt.py (shape-mismatch filter)')
else:
    print('  ckpt.py already patched')
"

echo "  Setup complete!"
SETUP_EOF

echo "==> [2/4] Dry-run: verify model + config on CPU..."
$SSH_CMD << 'DRYRUN_EOF'
set -euo pipefail
cd /workspace/OpenOCR

python3 -c "
import sys, yaml, torch
sys.path.insert(0, '.')

cfg = yaml.safe_load(open('/workspace/Typoon/scripts/train_rec_ja/openocr_finetune_ja.yml'))
print('[OK] Config parsed')

# Load & unwrap pretrained
ckpt = torch.load('/workspace/openocr_repsvtr_ch.pth', map_location='cpu', weights_only=False)
state = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
print(f'[OK] Pretrained: {len(state)} keys')

# Dict
ja_dict = open('/workspace/data/ja_dict.txt').read().strip().split('\n')
ja_classes = len(ja_dict) + 2
print(f'[OK] JA dict: {len(ja_dict)} chars -> {ja_classes} classes')

# Build model
from openrec.modeling import build_model
arch = cfg['Architecture'].copy()
arch['Decoder']['out_channels'] = ja_classes
model = build_model(arch)
ms = model.state_dict()

# Weight compatibility
loaded = {k: v for k, v in state.items() if k in ms and ms[k].shape == v.shape}
skipped = [k for k in state if k in ms and ms[k].shape != state[k].shape]
model.load_state_dict(loaded, strict=False)
print(f'[OK] Weights: {len(loaded)}/{len(ms)} transferred, {len(skipped)} reset (shape mismatch)')
if skipped:
    print(f'     Reset keys: {skipped}')

# Forward pass
model.eval()
with torch.no_grad():
    out = model(torch.randn(1, 3, 48, 320))
print(f'[OK] Forward pass OK')

# Loss
from openrec.losses import build_loss
build_loss(cfg['Loss'])
print(f'[OK] Loss OK')

print()
print('=== DRY-RUN PASSED ===')
"
DRYRUN_EOF

if [ $? -ne 0 ]; then
    echo "DRY-RUN FAILED! Fix issues before training."
    exit 1
fi

echo "==> [3/4] Starting training (finetune, ~1h on RTX 4090)..."
$SSH_CMD << 'TRAIN_EOF'
set -euo pipefail
cd /workspace/OpenOCR

CONFIG="/workspace/Typoon/scripts/train_rec_ja/openocr_finetune_ja.yml"

# Use nohup so training survives SSH disconnect
nohup python tools/train_rec.py -c "$CONFIG" > /workspace/train.log 2>&1 &
TRAIN_PID=$!
echo "  Training PID: $TRAIN_PID"

# Wait for first log output to confirm it's running
for i in $(seq 1 40); do
    sleep 5
    if grep -q "global_step:" /workspace/train.log 2>/dev/null; then
        tail -3 /workspace/train.log
        echo "  Training confirmed running!"
        exit 0
    fi
    if grep -q "Traceback" /workspace/train.log 2>/dev/null; then
        echo "  TRAINING FAILED:"
        tail -20 /workspace/train.log
        kill $TRAIN_PID 2>/dev/null || true
        exit 1
    fi
done
echo "  Timeout. Check: tail -f /workspace/train.log"
TRAIN_EOF

echo ""
echo "════════════════════════════════════════"
echo "  Training running on pod."
echo ""
echo "  Monitor:"
echo "    $SSH_CMD 'tail -f /workspace/train.log'"
echo ""
echo "  When training completes, export + download ONNX:"
echo "    $SSH_CMD 'cd /workspace/OpenOCR && python tools/toonnx.py -c /workspace/Typoon/scripts/train_rec_ja/openocr_finetune_ja.yml -o Global.device=cpu Global.pretrained_model=/workspace/output/svtrv2_ja_ft/best.pth'"
echo "    $SCP_CMD $SSH_HOST:/workspace/OpenOCR/output/rec/svtrv2_ja_ft/export_rec/rec_model.onnx $PROJECT_DIR/models/svtrv2_rec_ja.onnx"
echo ""
echo "  Upload to HF backup:"
echo "    huggingface-cli upload nghyane/manga109-crops models/svtrv2_rec_ja.onnx --repo-type dataset"
echo "════════════════════════════════════════"
