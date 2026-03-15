"""Quick overfit test to validate model capacity.

Loads a few manga images, applies random masks, and tries to overfit
the model on these samples. If the model can't memorize them, the
architecture lacks capacity for manga inpainting.

Usage:
    python overfit_test.py --data_dir ../../data/training/manga
    python overfit_test.py --data_dir ../../data/training/manga --num_images 10 --steps 500

Success criteria:
    - PSNR > 35 dB after overfitting → architecture is sufficient
    - PSNR < 25 dB → model too small, needs more capacity
    - Visual: screentone patterns reconstructed cleanly
"""

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path

from small_model import SmallInpaintModel, count_parameters
from dataset import generate_freeform_mask, find_images


def load_and_crop(path: str, size: int = 512) -> np.ndarray:
    """Load image, resize shorter side to size, center crop."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if min(w, h) < size:
        scale = size / min(w, h)
        img = img.resize((int(w * scale) + 1, int(h * scale) + 1), Image.LANCZOS)
        w, h = img.size
    x = (w - size) // 2
    y = (h - size) // 2
    img = img.crop((x, y, x + size, y + size))
    return np.array(img, dtype=np.float32) / 255.0  # [H,W,3] in [0,1]


def psnr(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    """Compute PSNR in masked region only."""
    diff = (pred - target) ** 2
    mse = (diff * mask).sum() / mask.sum().clamp(min=1) / 3  # per-channel
    if mse < 1e-10:
        return 50.0
    return -10 * torch.log10(mse).item()


def main():
    p = argparse.ArgumentParser(description="Overfit test for manga inpainting model")
    p.add_argument("--data_dir", type=str, default="../../data/training/manga",
                   help="Directory with manga images")
    p.add_argument("--num_images", type=int, default=8,
                   help="Number of images to overfit on")
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--steps", type=int, default=500,
                   help="Number of optimization steps")
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--base_ch", type=int, default=32,
                   help="Base channels for model")
    p.add_argument("--output_dir", type=str, default="overfit_output",
                   help="Directory to save comparison images")
    args = p.parse_args()

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load images
    all_paths = find_images(args.data_dir)
    if len(all_paths) == 0:
        print(f"No images found in {args.data_dir}")
        return

    random.seed(42)
    selected = random.sample(all_paths, min(args.num_images, len(all_paths)))
    print(f"Selected {len(selected)} images for overfit test")

    # Prepare fixed dataset (images + masks)
    images = []
    masks = []
    for path in selected:
        img = load_and_crop(path, args.image_size)
        img_t = torch.from_numpy(img.transpose(2, 0, 1))  # [3,H,W]

        # Generate mask (1=inpaint, 0=keep — opposite of dataset.py convention)
        mask_arr = generate_freeform_mask(args.image_size, args.image_size)
        mask_t = torch.from_numpy(1.0 - mask_arr).unsqueeze(0)  # flip: 1=inpaint

        images.append(img_t)
        masks.append(mask_t)

    images = torch.stack(images).to(device)  # [N, 3, H, W]
    masks = torch.stack(masks).to(device)    # [N, 1, H, W]

    masked_pct = masks.mean().item() * 100
    print(f"Average masked area: {masked_pct:.1f}%")

    # Build model
    model = SmallInpaintModel(base_ch=args.base_ch).to(device)
    params = count_parameters(model)
    param_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
    print(f"Model: {params:,} params ({param_mb:.2f} MB)")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps)

    # Overfit loop
    print(f"\nOverfitting for {args.steps} steps...")
    model.train()
    t0 = time.time()

    for step in range(1, args.steps + 1):
        output = model(images, masks)

        # L1 loss in masked region
        loss_masked = (output - images).abs() * masks
        loss_masked = loss_masked.sum() / masks.sum().clamp(min=1)

        # Small L1 on unmasked region too (should be near-zero due to compositing)
        loss_known = (output - images).abs() * (1 - masks)
        loss_known = loss_known.sum() / (1 - masks).sum().clamp(min=1)

        loss = loss_masked + loss_known * 0.1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 50 == 0 or step == 1:
            with torch.no_grad():
                p_val = psnr(output, images, masks.expand_as(images))
            elapsed = time.time() - t0
            print(f"  Step {step:4d}/{args.steps} | Loss: {loss.item():.5f} | "
                  f"PSNR(masked): {p_val:.1f} dB | Time: {elapsed:.1f}s")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        final_output = model(images, masks)
        final_psnr = psnr(final_output, images, masks.expand_as(images))

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Final PSNR (masked region): {final_psnr:.2f} dB")
    print(f"Total time: {elapsed:.1f}s")
    print(f"{'='*60}")

    if final_psnr > 35:
        print("✅ PASS — Model can fit manga patterns. Proceed to full training.")
    elif final_psnr > 28:
        print("⚠️  MARGINAL — Model partially fits. Consider increasing base_ch.")
    else:
        print("❌ FAIL — Model too small. Increase base_ch or add more layers.")

    # Save comparison images
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nSaving comparisons to {args.output_dir}/")

    for i in range(len(selected)):
        orig = (images[i].cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        masked = ((images[i] * (1 - masks[i])).cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        result = (final_output[i].cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)

        # Side-by-side: original | masked | result
        comparison = np.concatenate([orig, masked, result], axis=1)
        out_path = os.path.join(args.output_dir, f"compare_{i:02d}.png")
        Image.fromarray(comparison).save(out_path)

    print(f"Saved {len(selected)} comparison images (original | masked | result)")


if __name__ == "__main__":
    main()
