"""Export trained MI-GAN student to ONNX for inference.

Fuses re-parameterizable blocks, exports with opset 17, and verifies
the exported model matches PyTorch output.

Output model convention (matches LaMa for drop-in replacement):
    image: [1, 3, 512, 512] float32 [0, 1]
    mask:  [1, 1, 512, 512] float32 {0, 1} where 1=inpaint, 0=keep
    output: [1, 3, 512, 512] float32 [0, 1]

Note: internally MI-GAN uses 1=known, 0=masked and [-1,1] range.
The wrapper handles conversion so the ONNX model has the same I/O as LaMa.
"""

import argparse
import os

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from model import MIGANGenerator


class MIGANExportWrapper(nn.Module):
    """Wraps MI-GAN generator for ONNX export with LaMa-compatible I/O.

    Converts:
        Input:  image [0,1] + mask (1=inpaint) → MI-GAN input [-1,1] + mask (1=known)
        Output: MI-GAN output [-1,1] → composited output [0,1]
    """

    def __init__(self, generator: MIGANGenerator):
        super().__init__()
        self.gen = generator

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # image: [B,3,H,W] in [0,1]
        # mask:  [B,1,H,W] in {0,1}, 1=inpaint, 0=keep (LaMa convention)

        # Convert to MI-GAN convention: 1=known, 0=masked
        migan_mask = 1.0 - mask

        # Convert image to [-1,1]
        image_norm = image * 2.0 - 1.0

        # Masked image: zero out the inpaint region
        masked_rgb = image_norm * migan_mask

        # MI-GAN input: [mask-0.5, masked_rgb]
        mask_ch = migan_mask - 0.5
        gen_input = torch.cat([mask_ch, masked_rgb], dim=1)

        # Generate
        fake = self.gen(gen_input)

        # Composite: fill masked region with generated, keep known from original
        composite = fake * mask + image_norm * migan_mask

        # Convert back to [0,1]
        output = (composite + 1.0) * 0.5
        return output.clamp(0, 1)


def export(args):
    device = torch.device("cpu")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Prefer EMA weights
    if "ema" in ckpt:
        state_dict = ckpt["ema"]
        print("Using EMA weights")
    elif "gen" in ckpt:
        state_dict = ckpt["gen"]
        print("Using generator weights")
    else:
        state_dict = ckpt
        print("Using raw state dict")

    # Build model
    gen = MIGANGenerator(in_ch=4, base_ch=48)
    gen.load_state_dict(state_dict)
    gen.eval()

    # Fuse re-parameterizable blocks
    print("Fusing re-parameterizable blocks...")
    gen.fuse_reparam()

    # Verify fusion didn't break anything
    test_input = torch.randn(1, 4, 512, 512)
    with torch.no_grad():
        test_out = gen(test_input)
    print(f"Post-fusion output shape: {test_out.shape}")

    # Wrap for LaMa-compatible I/O
    wrapper = MIGANExportWrapper(gen)
    wrapper.eval()

    # Dummy inputs
    dummy_image = torch.rand(1, 3, 512, 512)
    dummy_mask = (torch.rand(1, 1, 512, 512) > 0.5).float()

    with torch.no_grad():
        pt_output = wrapper(dummy_image, dummy_mask)
    print(f"PyTorch output: shape={pt_output.shape}, range=[{pt_output.min():.3f}, {pt_output.max():.3f}]")

    # Export to ONNX
    output_path = args.output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print(f"Exporting to ONNX: {output_path}")
    torch.onnx.export(
        wrapper,
        (dummy_image, dummy_mask),
        output_path,
        opset_version=17,
        input_names=["image", "mask"],
        output_names=["output"],
        dynamic_axes=None,  # Fixed 512x512
        do_constant_folding=True,
    )

    # Verify ONNX model
    print("Verifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    # Run ONNX inference and compare
    session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    ort_output = session.run(
        ["output"],
        {"image": dummy_image.numpy(), "mask": dummy_mask.numpy()},
    )[0]

    ort_output_t = torch.from_numpy(ort_output)
    max_diff = (pt_output - ort_output_t).abs().max().item()
    mean_diff = (pt_output - ort_output_t).abs().mean().item()

    print(f"Verification: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")
    if max_diff > 1e-4:
        print("WARNING: Large difference between PyTorch and ONNX outputs!")
    else:
        print("OK: ONNX output matches PyTorch")

    # Print model size
    file_size = os.path.getsize(output_path)
    print(f"\nModel size: {file_size / 1024 / 1024:.2f} MB")
    print(f"Saved to: {output_path}")


def main():
    p = argparse.ArgumentParser(description="Export MI-GAN to ONNX")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to training checkpoint (.pt)")
    p.add_argument("--output", type=str, default="migan_inpaint.onnx",
                   help="Output ONNX path")
    args = p.parse_args()
    export(args)


if __name__ == "__main__":
    main()
