"""Dataset for manga/manhwa inpainting training.

Loads images from a directory (supports nested subdirs), generates random
free-form brush stroke masks, and applies augmentation.

Mask convention: 1 = known pixel, 0 = missing (to be inpainted).
Images normalized to [-1, 1].
"""

import os
import random
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import torchvision.transforms.functional as TF

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def find_images(root: str) -> list[str]:
    """Recursively find all image files under root."""
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if Path(f).suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(os.path.join(dirpath, f))
    paths.sort()
    return paths


# ---------------------------------------------------------------------------
# Free-form mask generation (thick brush strokes)
# ---------------------------------------------------------------------------

def generate_freeform_mask(h: int, w: int,
                           num_strokes: int = 0,
                           min_width: int = 15,
                           max_width: int = 60,
                           min_length: int = 50,
                           max_length: int = 200) -> np.ndarray:
    """Generate a random free-form mask with thick brush strokes.

    Returns a float32 array [H, W] where 1=known, 0=masked.
    Aims for ~15-40% masked area (typical for inpainting training).
    """
    mask = Image.new("L", (w, h), 255)  # start all known
    draw = ImageDraw.Draw(mask)

    if num_strokes == 0:
        num_strokes = random.randint(3, 8)

    for _ in range(num_strokes):
        num_vertices = random.randint(4, 12)
        width = random.randint(min_width, max_width)
        points = []
        x = random.randint(0, w)
        y = random.randint(0, h)
        for _ in range(num_vertices):
            angle = random.uniform(0, 2 * math.pi)
            length = random.randint(min_length, max_length)
            x = int(np.clip(x + length * math.cos(angle), 0, w))
            y = int(np.clip(y + length * math.sin(angle), 0, h))
            points.append((x, y))

        # Draw thick polyline
        for i in range(len(points) - 1):
            draw.line([points[i], points[i + 1]], fill=0, width=width)

        # Draw circles at vertices for smooth joints
        for px, py in points:
            r = width // 2
            draw.ellipse([px - r, py - r, px + r, py + r], fill=0)

    arr = np.array(mask, dtype=np.float32) / 255.0  # 1=known, 0=masked
    return arr


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MangaInpaintDataset(Dataset):
    """Manga/manhwa inpainting dataset.

    Args:
        image_dir: Root directory of images (nested subdirs OK).
        mask_dir: Optional directory of pre-computed masks (grayscale,
                  255=known, 0=masked). If None, random masks are generated.
        image_size: Output crop size (square).
        augment: Enable random augmentation (flip, crop).
    """

    def __init__(self, image_dir: str, mask_dir: Optional[str] = None,
                 image_size: int = 512, augment: bool = True):
        self.image_paths = find_images(image_dir)
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_dir}")

        self.mask_paths: Optional[list[str]] = None
        if mask_dir is not None:
            self.mask_paths = find_images(mask_dir)
            if len(self.mask_paths) == 0:
                raise ValueError(f"No masks found in {mask_dir}")

        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Load image
        img = Image.open(self.image_paths[idx]).convert("RGB")

        # Random crop / resize to target size
        img = self._prepare_image(img)

        # Random horizontal flip
        if self.augment and random.random() > 0.5:
            img = TF.hflip(img)

        # Convert to tensor [-1, 1]
        img_tensor = TF.to_tensor(img) * 2.0 - 1.0  # [3, H, W] in [-1, 1]

        # Generate or load mask
        if self.mask_paths is not None:
            mask_idx = random.randint(0, len(self.mask_paths) - 1)
            mask_img = Image.open(self.mask_paths[mask_idx]).convert("L")
            mask_img = mask_img.resize((self.image_size, self.image_size),
                                       Image.NEAREST)
            mask = np.array(mask_img, dtype=np.float32) / 255.0
        else:
            mask = generate_freeform_mask(self.image_size, self.image_size)

        mask_tensor = torch.from_numpy(mask).unsqueeze(0)  # [1, H, W], 1=known, 0=masked

        return img_tensor, mask_tensor

    def _prepare_image(self, img: Image.Image) -> Image.Image:
        """Resize and crop image to self.image_size."""
        w, h = img.size
        s = self.image_size

        # If image is smaller than target, resize up
        if min(w, h) < s:
            scale = s / min(w, h)
            img = img.resize((int(w * scale) + 1, int(h * scale) + 1),
                             Image.LANCZOS)
            w, h = img.size

        # Random crop
        if self.augment:
            x = random.randint(0, w - s)
            y = random.randint(0, h - s)
        else:
            x = (w - s) // 2
            y = (h - s) // 2

        img = img.crop((x, y, x + s, y + s))
        return img


# ---------------------------------------------------------------------------
# Teacher cache dataset
# ---------------------------------------------------------------------------

class TeacherCacheDataset(Dataset):
    """Load pre-generated (image, mask, teacher_output) triplets from disk.

    Each sample is stored as a .npz file containing:
        image: float32 [3, H, W] in [-1, 1]
        mask:  float32 [1, H, W] in {0, 1}
        teacher: float32 [3, H, W] in [-1, 1]
    """

    def __init__(self, cache_dir: str, augment: bool = True):
        self.files = sorted(
            str(p) for p in Path(cache_dir).rglob("*.npz")
        )
        if len(self.files) == 0:
            raise ValueError(f"No .npz files found in {cache_dir}")
        self.augment = augment

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data = np.load(self.files[idx])
        image = torch.from_numpy(data["image"])    # [3, H, W]
        mask = torch.from_numpy(data["mask"])      # [1, H, W]
        teacher = torch.from_numpy(data["teacher"])  # [3, H, W]

        # Random horizontal flip (apply consistently to all)
        if self.augment and random.random() > 0.5:
            image = torch.flip(image, [-1])
            mask = torch.flip(mask, [-1])
            teacher = torch.flip(teacher, [-1])

        return image, mask, teacher


if __name__ == "__main__":
    # Quick test: generate a mask and display stats
    mask = generate_freeform_mask(512, 512)
    masked_pct = (1 - mask.mean()) * 100
    print(f"Mask stats: shape={mask.shape}, masked={masked_pct:.1f}%")
