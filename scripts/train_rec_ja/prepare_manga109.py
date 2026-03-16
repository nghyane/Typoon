#!/usr/bin/env python3
"""
Prepare Manga109-s dataset for PP-OCR v5 recognition fine-tuning.

Reads the ZIP directly (no extraction needed), crops text regions from page
images using XML annotations, and outputs PaddleOCR training format:
    image_path\tlabel

Vertical text crops are rotated 90° CW so the model sees horizontal lines
(PP-OCR rec expects horizontal input). A padding margin is added around each
crop for better recognition at bubble edges.

Usage:
    python prepare_manga109.py \
        --zip /path/to/Manga109s_released_2023_12_07.zip \
        --output ./data \
        --val-ratio 0.1
"""

import argparse
import io
import random
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from PIL import Image

# Padding around crop as fraction of region size
CROP_PAD_RATIO = 0.08
# Minimum crop dimensions (skip tiny noise)
MIN_CROP_W = 8
MIN_CROP_H = 8
# Skip labels with only whitespace or single char (likely noise)
MIN_LABEL_LEN = 2

ZIP_PREFIX = "Manga109s_released_2023_12_07"


def parse_args():
    p = argparse.ArgumentParser(description="Prepare Manga109-s for PP-OCR rec training")
    p.add_argument("--zip", required=True, help="Path to Manga109s ZIP")
    p.add_argument("--output", default="./data", help="Output directory")
    p.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--annotations-dir", default="annotations",
                    help="Annotations subdirectory inside ZIP (default: latest)")
    return p.parse_args()


def crop_text_region(page_img: Image.Image, xmin: int, ymin: int, xmax: int, ymax: int):
    """Crop text region with padding, rotate vertical text to horizontal."""
    w, h = xmax - xmin, ymax - ymin
    if w < MIN_CROP_W or h < MIN_CROP_H:
        return None

    # Add padding
    pad_x = int(w * CROP_PAD_RATIO)
    pad_y = int(h * CROP_PAD_RATIO)
    pw, ph = page_img.size
    x1 = max(0, xmin - pad_x)
    y1 = max(0, ymin - pad_y)
    x2 = min(pw, xmax + pad_x)
    y2 = min(ph, ymax + pad_y)

    crop = page_img.crop((x1, y1, x2, y2))

    # Vertical text: height >> width → rotate 90° CW to make horizontal
    cw, ch = crop.size
    if ch > cw * 1.5:
        crop = crop.transpose(Image.Transpose.ROTATE_270)

    return crop


def process_volume(zf: zipfile.ZipFile, book_title: str, annot_dir: str,
                   output_dir: Path, crops_dir: Path):
    """Process one manga volume: parse XML, crop text regions, return labels."""
    xml_path = f"{ZIP_PREFIX}/{annot_dir}/{book_title}.xml"
    try:
        xml_bytes = zf.read(xml_path)
    except KeyError:
        print(f"  SKIP {book_title}: annotation not found at {xml_path}")
        return []

    root = ET.fromstring(xml_bytes)
    labels = []

    for page in root.iter("page"):
        page_idx = page.get("index")
        img_path = f"{ZIP_PREFIX}/images/{book_title}/{page_idx.zfill(3)}.jpg"

        try:
            img_bytes = zf.read(img_path)
        except KeyError:
            continue

        page_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        for text_el in page.findall("text"):
            label = text_el.text
            if not label or len(label.strip()) < MIN_LABEL_LEN:
                continue

            # Collapse multi-line labels into single line (manga bubbles can have newlines)
            label = label.strip().replace("\n", "").replace("\r", "")

            xmin = int(text_el.get("xmin"))
            ymin = int(text_el.get("ymin"))
            xmax = int(text_el.get("xmax"))
            ymax = int(text_el.get("ymax"))

            crop = crop_text_region(page_img, xmin, ymin, xmax, ymax)
            if crop is None:
                continue

            # Save crop
            crop_name = f"{book_title}_{page_idx}_{text_el.get('id')}.jpg"
            crop_path = crops_dir / crop_name
            crop.save(crop_path, "JPEG", quality=95)

            # PaddleOCR format: relative path from data dir
            rel_path = f"crops/{crop_name}"
            labels.append(f"{rel_path}\t{label}")

    return labels


def main():
    args = parse_args()
    random.seed(args.seed)

    output_dir = Path(args.output)
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading ZIP: {args.zip}")
    zf = zipfile.ZipFile(args.zip, "r")

    # Find all book titles from images/ directory
    image_dirs = set()
    for name in zf.namelist():
        parts = name.split("/")
        if len(parts) >= 3 and parts[1] == "images" and parts[2]:
            image_dirs.add(parts[2])

    books = sorted(image_dirs)
    print(f"Found {len(books)} volumes")

    all_labels = []
    for i, book in enumerate(books):
        labels = process_volume(zf, book, args.annotations_dir, output_dir, crops_dir)
        all_labels.extend(labels)
        if (i + 1) % 10 == 0 or i == len(books) - 1:
            print(f"  [{i+1}/{len(books)}] {book}: {len(labels)} crops (total: {len(all_labels)})")

    zf.close()

    if not all_labels:
        print("ERROR: No crops extracted!")
        return

    # Shuffle and split
    random.shuffle(all_labels)
    val_count = int(len(all_labels) * args.val_ratio)
    val_labels = all_labels[:val_count]
    train_labels = all_labels[val_count:]

    # Write label files
    train_file = output_dir / "train.txt"
    val_file = output_dir / "val.txt"

    train_file.write_text("\n".join(train_labels) + "\n", encoding="utf-8")
    val_file.write_text("\n".join(val_labels) + "\n", encoding="utf-8")

    # Build character dictionary from all labels
    chars = set()
    for line in all_labels:
        _, label = line.split("\t", 1)
        chars.update(label)

    dict_file = output_dir / "ja_dict.txt"
    sorted_chars = sorted(chars)
    dict_file.write_text("\n".join(sorted_chars) + "\n", encoding="utf-8")

    print(f"\nDone!")
    print(f"  Train: {len(train_labels)} samples → {train_file}")
    print(f"  Val:   {len(val_labels)} samples → {val_file}")
    print(f"  Dict:  {len(sorted_chars)} characters → {dict_file}")
    print(f"  Crops: {crops_dir}")


if __name__ == "__main__":
    main()
