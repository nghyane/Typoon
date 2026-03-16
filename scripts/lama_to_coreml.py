#!/usr/bin/env python3
"""Load LaMa from safetensors → PyTorch (native FFT) → CoreML .mlpackage.

Native torch.fft.rfft2/irfft2 produce far fewer ops than the decomposed
FourierUnitJIT used in ONNX export (~2 ops vs ~40 ops per FFC block).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
import coremltools as ct
import numpy as np
import time, os, warnings
warnings.filterwarnings("ignore")


# ─── Model Architecture (matches koharu/mayocream weight keys) ───

class FourierUnit(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv_layer = nn.Conv2d(in_ch * 2, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.out_ch = out_ch // 2

    def forward(self, x):
        W = x.shape[3]
        spec = torch.fft.rfft2(x, norm="backward")              # complex [B,C,H,W//2+1]
        # stack real/imag → [B, C*2, H, W//2+1]
        stacked = torch.cat([spec.real, spec.imag], dim=1)
        y = self.conv_layer(stacked)
        y = F.relu(self.bn(y))
        # split back to complex
        half = self.out_ch
        y_complex = torch.complex(y[:, :half], y[:, half:])
        return torch.fft.irfft2(y_complex, s=(x.shape[2], W), norm="backward")


class SpectralTransform(nn.Module):
    def __init__(self, stride, in_ch, out_ch):
        super().__init__()
        self.downsample = stride == 2
        half = out_ch // 2
        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, half, 1, bias=False), nn.BatchNorm2d(half))
        self.fu = FourierUnit(half, out_ch)
        self.conv2 = nn.Conv2d(half, out_ch, 1, bias=False)

    def forward(self, x):
        if self.downsample:
            x = F.avg_pool2d(x, 2, 2)
        y = F.relu(self.conv1(x))
        fu = self.fu(y)
        return self.conv2(y + fu)


class FFC(nn.Module):
    def __init__(self, in_l, in_g, out_l, out_g, ks, stride, pad, dilation):
        super().__init__()
        self.convl2l = nn.Conv2d(in_l, out_l, ks, stride, 0, dilation, bias=False) if out_l > 0 and in_l > 0 else None
        self.convl2g = nn.Conv2d(in_l, out_g, ks, stride, 0, dilation, bias=False) if out_g > 0 and in_l > 0 else None
        self.convg2l = nn.Conv2d(in_g, out_l, ks, stride, 0, dilation, bias=False) if in_g > 0 and out_l > 0 else None
        self.convg2g = SpectralTransform(stride, in_g, out_g) if in_g > 0 and out_g > 0 else None
        self.pad = pad

    def forward(self, x_l, x_g):
        def pad_and_conv(conv, x):
            if conv is None:
                return None
            x = F.pad(x, [self.pad]*4, mode='reflect')
            return conv(x)

        out_l = pad_and_conv(self.convl2l, x_l)
        if self.convg2l is not None and x_g is not None:
            g2l = pad_and_conv(self.convg2l, x_g)
            out_l = out_l + g2l if out_l is not None else g2l

        out_g = pad_and_conv(self.convl2g, x_l)
        if self.convg2g is not None and x_g is not None:
            g2g = self.convg2g(x_g)
            out_g = out_g + g2g if out_g is not None else g2g

        return out_l, out_g


class FFCBnAct(nn.Module):
    def __init__(self, in_l, in_g, out_l, out_g, ks, stride, pad, dilation):
        super().__init__()
        self.ffc = FFC(in_l, in_g, out_l, out_g, ks, stride, pad, dilation)
        self.bn_l = nn.BatchNorm2d(out_l) if out_l > 0 else None
        self.bn_g = nn.BatchNorm2d(out_g) if out_g > 0 else None

    def forward(self, x_l, x_g):
        l, g = self.ffc(x_l, x_g)
        if self.bn_l is not None and l is not None:
            l = F.relu(self.bn_l(l))
        if self.bn_g is not None and g is not None:
            g = F.relu(self.bn_g(g))
        return l, g


class FFCResBlock(nn.Module):
    def __init__(self, ch_l, ch_g):
        super().__init__()
        self.conv1 = FFCBnAct(ch_l, ch_g, ch_l, ch_g, 3, 1, 1, 1)
        self.conv2 = FFCBnAct(ch_l, ch_g, ch_l, ch_g, 3, 1, 1, 1)

    def forward(self, x_l, x_g):
        y_l, y_g = self.conv1(x_l, x_g)
        y_l, y_g = self.conv2(y_l, y_g)
        out_l = y_l + x_l
        out_g = y_g + x_g if y_g is not None and x_g is not None else y_g
        return out_l, out_g


class LaMa(nn.Module):
    def __init__(self):
        super().__init__()
        self.pad = 3
        # Encoder
        self.init = FFCBnAct(4, 0, 64, 0, 7, 1, 0, 1)
        self.down1 = FFCBnAct(64, 0, 128, 0, 3, 2, 1, 1)
        self.down2 = FFCBnAct(128, 0, 256, 0, 3, 2, 1, 1)
        self.down3 = FFCBnAct(256, 0, 128, 384, 3, 2, 1, 1)
        # Bottleneck: 18 FFC ResBlocks
        self.blocks = nn.ModuleList([FFCResBlock(128, 384) for _ in range(18)])
        # Decoder
        self.up1 = nn.ConvTranspose2d(512, 256, 3, 2, 1, output_padding=1)
        self.bn_up1 = nn.BatchNorm2d(256)
        self.up2 = nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1)
        self.bn_up2 = nn.BatchNorm2d(128)
        self.up3 = nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1)
        self.bn_up3 = nn.BatchNorm2d(64)
        self.final_conv = nn.Conv2d(64, 3, 7, 1, 0)

    def forward(self, image, mask):
        mask_inv = 1.0 - mask
        img_masked = image * mask_inv
        x = torch.cat([img_masked, mask], dim=1)
        x = F.pad(x, [self.pad]*4, mode='reflect')

        l, g = self.init(x, None)
        l, g = self.down1(l, g)
        l, g = self.down2(l, g)
        l, g = self.down3(l, g)

        for blk in self.blocks:
            l, g = blk(l, g)

        x = torch.cat([l, g], dim=1)
        x = F.relu(self.bn_up1(self.up1(x)))
        x = F.relu(self.bn_up2(self.up2(x)))
        x = F.relu(self.bn_up3(self.up3(x)))
        x = F.pad(x, [self.pad]*4, mode='reflect')
        x = torch.sigmoid(self.final_conv(x))
        H, W = image.shape[2], image.shape[3]
        x = x[:, :, :H, :W]
        return x * mask + image * mask_inv


def remap_keys(sd):
    """Remap safetensors keys to our module names."""
    mapping = {}
    mapping["model.1"] = "init"
    mapping["model.2"] = "down1"
    mapping["model.3"] = "down2"
    mapping["model.4"] = "down3"
    for i in range(18):
        mapping[f"model.{i+5}"] = f"blocks.{i}"
    mapping["model.24"] = "up1"
    mapping["model.25"] = "bn_up1"
    mapping["model.27"] = "up2"
    mapping["model.28"] = "bn_up2"
    mapping["model.30"] = "up3"
    mapping["model.31"] = "bn_up3"
    mapping["model.34"] = "final_conv"

    new_sd = {}
    for k, v in sd.items():
        new_k = k
        for old, new in sorted(mapping.items(), key=lambda x: -len(x[0])):
            if new_k.startswith(old + "."):
                new_k = new + new_k[len(old):]
                break
        # conv1.0.weight → conv1.0.weight (Sequential)
        new_sd[new_k] = v
    return new_sd


if __name__ == "__main__":
    print("1/5 Loading safetensors...", flush=True)
    sd = load_file("models/lama-manga.safetensors")

    print("2/5 Building PyTorch model...", flush=True)
    model = LaMa()
    model.eval()

    new_sd = remap_keys(sd)

    # Debug key mismatches
    model_keys = set(model.state_dict().keys())
    file_keys = set(new_sd.keys())
    missing = model_keys - file_keys
    extra = file_keys - model_keys
    if missing:
        print(f"  Missing {len(missing)} keys:", list(missing)[:5])
    if extra:
        print(f"  Extra {len(extra)} keys:", list(extra)[:5])

    model.load_state_dict(new_sd, strict=False)
    print(f"  Loaded {len(model_keys & file_keys)}/{len(model_keys)} keys", flush=True)

    # Verify with random input
    with torch.no_grad():
        img = torch.randn(1, 3, 512, 512)
        mask = torch.zeros(1, 1, 512, 512)
        mask[0, 0, 100:300, 100:400] = 1.0
        out = model(img, mask)
        print(f"  Output shape: {out.shape}, range: [{out.min():.3f}, {out.max():.3f}]", flush=True)

    print("3/5 Tracing...", flush=True)
    traced = torch.jit.trace(model, (img, mask))

    print("4/5 Converting to CoreML MLProgram (FP16)...", flush=True)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType("image", shape=(1,3,512,512)),
                ct.TensorType("mask", shape=(1,1,512,512))],
        outputs=[ct.TensorType("output")],
        minimum_deployment_target=ct.target.macOS14,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
    )

    print("5/5 Saving...", flush=True)
    mlmodel.save("models/lama_inpaint.mlpackage")
    size = sum(os.path.getsize(os.path.join(dp, f))
               for dp, _, fn in os.walk("models/lama_inpaint.mlpackage") for f in fn)
    print(f"  Saved: {size/1024/1024:.1f}MB", flush=True)

    # Benchmark
    print("\nBenchmark:", flush=True)
    img_np = np.random.rand(1,3,512,512).astype(np.float32)
    mask_np = np.zeros((1,1,512,512), dtype=np.float32)
    mask_np[0,0,100:300,100:400] = 1.0
    _ = mlmodel.predict({"image": img_np, "mask": mask_np})
    t0 = time.time()
    for _ in range(8):
        _ = mlmodel.predict({"image": img_np, "mask": mask_np})
    ms = (time.time() - t0) * 1000 / 8
    print(f"CoreML native: {ms:.0f}ms/tile", flush=True)
