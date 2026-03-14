"""Pre-generate LaMa teacher outputs for faster distillation training.

Runs LaMa on all training images with multiple random masks per image,
saving (image, mask, teacher_output) triplets as .npz files.

This avoids running LaMa inference during training (~50% speedup).

Usage:
    python generate_teacher_cache.py \
        --data_dir /path/to/manga_images \
        --output_dir /path/to/teacher_cache \
        --lama_model ../../models/lama_fp32.onnx \
        --masks_per_image 5
"""

import argparse
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as TF

from dataset import find_images, generate_freeform_mask


def prepare_image(img: Image.Image, size: int = 512) -> np.ndarray:
    """Resize and center-crop image to size, return [3,H,W] float32 in [0,1]."""
    w, h = img.size

    # Resize so shorter side = size
    if min(w, h) < size:
        scale = size / min(w, h)
        img = img.resize((int(w * scale) + 1, int(h * scale) + 1), Image.LANCZOS)
        w, h = img.size

    # Center crop
    x = (w - size) // 2
    y = (h - size) // 2
    img = img.crop((x, y, x + size, y + size))

    arr = np.array(img, dtype=np.float32) / 255.0  # [H, W, 3] in [0, 1]
    arr = arr.transpose(2, 0, 1)  # [3, H, W]
    return arr


def random_crop_image(img: Image.Image, size: int = 512) -> np.ndarray:
    """Random crop image to size, return [3,H,W] float32 in [0,1]."""
    import random
    w, h = img.size

    if min(w, h) < size:
        scale = size / min(w, h)
        img = img.resize((int(w * scale) + 1, int(h * scale) + 1), Image.LANCZOS)
        w, h = img.size

    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    img = img.crop((x, y, x + size, y + size))

    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    return arr


def main():
    p = argparse.ArgumentParser(description="Generate LaMa teacher cache")
    p.add_argument("--data_dir", type=str, required=True,
                   help="Root directory of training images")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory to save cache .npz files")
    p.add_argument("--lama_model", type=str,
                   default="../../models/lama_fp32.onnx",
                   help="Path to LaMa ONNX model")
    p.add_argument("--masks_per_image", type=int, default=5,
                   help="Number of random masks to generate per image")
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=4,
                   help="Batch size for LaMa inference")
    p.add_argument("--random_crop", action="store_true",
                   help="Use random crops instead of center crops")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load LaMa
    available = ort.get_available_providers()
    providers = []
    if "CUDAExecutionProvider" in available:
        providers.append(("CUDAExecutionProvider", {"device_id": 0}))
    providers.append("CPUExecutionProvider")
    session = ort.InferenceSession(args.lama_model, providers=providers)
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]
    print(f"LaMa loaded: inputs={input_names}, outputs={output_names}")

    # Find images
    image_paths = find_images(args.data_dir)
    print(f"Found {len(image_paths)} images")

    total = len(image_paths) * args.masks_per_image
    sample_idx = 0

    pbar = tqdm(total=total, desc="Generating cache")

    # Process images in batches for efficiency
    batch_images = []
    batch_masks = []
    batch_meta = []  # (sample_idx,) for naming

    for img_idx, img_path in enumerate(image_paths):
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Skipping {img_path}: {e}")
            pbar.update(args.masks_per_image)
            continue

        for mask_idx in range(args.masks_per_image):
            # Prepare image
            if args.random_crop:
                img_arr = random_crop_image(img, args.image_size)
            else:
                img_arr = prepare_image(img, args.image_size)

            # Generate mask (MI-GAN convention: 1=known, 0=masked)
            mask_arr = generate_freeform_mask(args.image_size, args.image_size)
            mask_arr = mask_arr[np.newaxis, ...]  # [1, H, W]

            # Skip if already cached
            out_path = os.path.join(args.output_dir, f"sample_{sample_idx:07d}.npz")
            if os.path.exists(out_path):
                sample_idx += 1
                pbar.update(1)
                continue

            batch_images.append(img_arr)
            batch_masks.append(mask_arr)
            batch_meta.append(sample_idx)
            sample_idx += 1

            # Process batch
            if len(batch_images) >= args.batch_size:
                _process_batch(session, input_names, output_names,
                               batch_images, batch_masks, batch_meta,
                               args.output_dir)
                pbar.update(len(batch_images))
                batch_images.clear()
                batch_masks.clear()
                batch_meta.clear()

    # Process remaining
    if batch_images:
        _process_batch(session, input_names, output_names,
                       batch_images, batch_masks, batch_meta,
                       args.output_dir)
        pbar.update(len(batch_images))

    pbar.close()
    print(f"Generated {sample_idx} cache samples in {args.output_dir}")


def _process_batch(session, input_names, output_names,
                   images, masks, meta, output_dir):
    """Run LaMa on a batch and save results."""
    img_batch = np.stack(images, axis=0)     # [B, 3, H, W] in [0, 1]
    mask_batch = np.stack(masks, axis=0)     # [B, 1, H, W] in {0, 1}

    # LaMa mask convention: 1=inpaint, 0=keep (opposite of MI-GAN)
    lama_mask = 1.0 - mask_batch

    # Masked image for LaMa
    masked_img = img_batch * mask_batch  # keep known, zero masked

    # Run LaMa
    feed = {input_names[0]: masked_img.astype(np.float32),
            input_names[1]: lama_mask.astype(np.float32)}
    teacher_out = session.run(output_names, feed)[0]  # [B, 3, H, W] in [0, 1]
    teacher_out = np.clip(teacher_out, 0, 1)

    # Save each sample
    for i, idx in enumerate(meta):
        # Convert to [-1, 1] for training (MI-GAN convention)
        image_norm = img_batch[i] * 2.0 - 1.0      # [3, H, W] in [-1, 1]
        teacher_norm = teacher_out[i] * 2.0 - 1.0   # [3, H, W] in [-1, 1]

        out_path = os.path.join(output_dir, f"sample_{idx:07d}.npz")
        np.savez_compressed(
            out_path,
            image=image_norm.astype(np.float32),
            mask=mask_batch[i].astype(np.float32),
            teacher=teacher_norm.astype(np.float32),
        )


if __name__ == "__main__":
    main()
