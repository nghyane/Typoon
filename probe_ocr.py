#!/usr/bin/env python3
"""Probe OCR quality: raw vs adaptive binarization on manhwa/manhua source images."""

import cv2
import numpy as np
from pathlib import Path

SRC = Path("/Users/nghiahoang/.typoon/projects/system-universe-3cdbf8/source")
CHAPTERS = ["ch001", "ch002", "ch003"]

def measure_text_density(gray: np.ndarray) -> float:
    """Estimate how much text content remains after binarization."""
    # Low mean = more dark pixels = text preserved
    # High mean = mostly white = text lost
    return gray.mean()

def probe_adaptive_params(gray: np.ndarray):
    """Test different adaptive threshold params."""
    results = []
    for block in [11, 21, 31, 51]:
        for c in [-10, -5, -2, 0, 2, 5, 10]:
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, c)
            density = measure_text_density(binary)
            results.append((block, c, density))
    return results

def analyze_page(img_path: Path):
    img = cv2.imread(str(img_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Check if image is primarily grayscale (manga) vs color (manhwa/manhua)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation_mean = hsv[:, :, 1].mean()
    
    print(f"\n{'='*60}")
    print(f"Analyzing: {img_path.name}")
    print(f"  Resolution: {img.shape}")
    print(f"  Saturation (colorfulness): {saturation_mean:.1f}")
    print(f"  Type: {'Manga (B&W/low color)' if saturation_mean < 30 else 'Manhwa/Manhua (color)'}")
    
    # Sample regions where text typically appears
    h, w = img.shape[:2]
    regions = [
        ("top", slice(0, h//3), slice(w//4, 3*w//4)),
        ("middle", slice(h//3, 2*h//3), slice(w//4, 3*w//4)),
        ("bottom", slice(2*h//3, h), slice(w//4, 3*w//4)),
    ]
    
    for name, rs, cs in regions:
        region_gray = gray[rs, cs]
        print(f"\n  Region [{name}]:")
        print(f"    Original gray: mean={region_gray.mean():.1f}, std={region_gray.std():.1f}")
        
        # Test adaptive params
        results = probe_adaptive_params(region_gray)
        best = min(results, key=lambda x: abs(x[2] - 127.5))  # Most balanced
        print(f"    Best adaptive: block={best[0]}, C={best[1]}, mean={best[2]:.1f}")
        
        # Current implementation params
        current = cv2.adaptiveThreshold(region_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        print(f"    Current (11,2): mean={current.mean():.1f}")
        
        # Warning if current implementation would lose too much text
        if current.mean() > 200:
            print(f"    ⚠️  WARN: Current adaptive binarization may lose text (mean > 200)")

def main():
    print("OCR Quality Probe - Adaptive Binarization Check")
    print("="*60)
    
    for ch in CHAPTERS:
        ch_dir = SRC / ch
        if not ch_dir.exists():
            print(f"Skipping {ch_dir} - not found")
            continue
            
        files = sorted(ch_dir.glob("*.webp"))
        if not files:
            print(f"No webp files in {ch_dir}")
            continue
            
        # Test first page only for quick probe
        analyze_page(files[0])
        
        # Save debug images
        debug_dir = Path("debug-bin")
        debug_dir.mkdir(exist_ok=True)
        
        img = cv2.imread(str(files[0]))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Current implementation
        current = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        cv2.imwrite(str(debug_dir / f"{ch}_current_adaptive.png"), current)
        
        # Optimal
        results = probe_adaptive_params(gray)
        best = min(results, key=lambda x: abs(x[2] - 127.5))
        optimal = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, best[0], best[1])
        cv2.imwrite(str(debug_dir / f"{ch}_optimal_adaptive.png"), optimal)
        
        # Otsu
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite(str(debug_dir / f"{ch}_otsu.png"), otsu)
        
        print(f"\nDebug images saved to {debug_dir}/")

if __name__ == "__main__":
    main()
