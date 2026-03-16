# PP-OCR v5 Japanese Manga Recognition Training

Fine-tune PP-OCR v5 server rec model on Manga109-s dataset for Japanese manga
text recognition. Replaces the autoregressive manga-ocr with a CTC-based model
that supports batch inference.

## Step 1: Prepare Data

```bash
python prepare_manga109.py \
    --zip /path/to/Manga109s_released_2023_12_07.zip \
    --output ./data
```

Output:
- `data/crops/` — 111K+ cropped text regions (vertical text rotated to horizontal)
- `data/train.txt` — ~100K training labels
- `data/val.txt` — ~11K validation labels
- `data/ja_dict.txt` — 3191 Japanese characters

## Step 2: Setup PaddleOCR

```bash
# Install PaddlePaddle (GPU)
pip install paddlepaddle-gpu

# Clone PaddleOCR
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR && pip install -r requirements.txt

# Download pretrained weights
wget -P pretrained/ \
    https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_rec_pretrained.pdparams
```

## Step 3: Train

```bash
bash train.sh
```

Training takes ~8h on RTX 3060, ~3h on RTX 4090.

## Step 4: Export to ONNX

```bash
cd PaddleOCR

# Export Paddle inference model
python tools/export_model.py \
    -c /path/to/ppocr_rec_ja.yml \
    -o Global.pretrained_model=output/rec_ja/best_accuracy \
       Global.save_inference_dir=output/rec_ja/inference

# Convert to ONNX
pip install paddle2onnx
paddle2onnx \
    --model_dir output/rec_ja/inference \
    --model_filename inference.pdmodel \
    --params_filename inference.pdiparams \
    --save_file ppocr_rec_ja.onnx \
    --opset_version 14
```

## Step 5: Deploy in ComicScan

Copy `ppocr_rec_ja.onnx` and `data/ja_dict.txt` to `models/`:
```bash
cp ppocr_rec_ja.onnx ../../models/
cp data/ja_dict.txt ../../models/
```

Then update ComicScan to use PP-OCR rec for Japanese (removes manga-ocr dependency):
- Load `ppocr_rec_ja.onnx` as the rec model when `source_lang == "ja"`
- Use PP-OCR det (or comic-text-detector) for detection
- Batch inference + pipeline overlap works for all languages
