# MI-GAN Distillation from LaMa Teacher

Distills a manga-finetuned LaMa inpainting model into a lightweight MI-GAN student
(~3-10MB ONNX) that uses only standard conv ops — no FFT, no transformer — suitable
for CoreML EP on Apple Silicon.

## Setup

```bash
cd scripts/distill
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data Preparation

Organize manga/manhwa images in a directory (nested subdirs OK):

```
/path/to/manga_images/
  series_a/
    001.jpg
    002.png
  series_b/
    ...
```

## Training

```bash
# Basic training (LaMa teacher runs online during training)
python train.py --data_dir /path/to/manga_images

# Optional: pre-generate teacher outputs to speed up training ~2x
python generate_teacher_cache.py \
  --data_dir /path/to/manga_images \
  --output_dir /path/to/teacher_cache \
  --lama_model ../../models/lama_fp32.onnx

# Train with teacher cache
python train.py --data_dir /path/to/manga_images --teacher_cache /path/to/teacher_cache
```

## Export to ONNX

```bash
python export_onnx.py --checkpoint checkpoints/best.pt --output migan_inpaint.onnx
```

The exported model accepts:
- `image`: `[1, 3, 512, 512]` float32, range `[0, 1]`
- `mask`: `[1, 1, 512, 512]` float32, `{0, 1}` where `1 = inpaint region`

Output: `[1, 3, 512, 512]` float32, range `[0, 1]`

## Architecture

MI-GAN (ICCV 2023) UNet with depthwise-separable convolutions and re-parameterizable
blocks. During training, RepVGG-style multi-branch blocks improve capacity; at export,
they fuse into single convolutions for efficient inference.
