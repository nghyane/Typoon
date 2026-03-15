"""Lightweight manga inpainting model.

Small UNet with dilated convolution bottleneck for manga bubble inpainting.
No FFT/spectral ops — runs on any device (CPU, MPS, CUDA, ONNX, CoreML).

Architecture:
    Input:  4ch [masked_rgb(3), mask(1)] — mask: 1=inpaint, 0=keep
    Encoder: 3 levels (48→96→192), stride-2 downsample
    Bottleneck: 5 dilated convs (d=1,2,4,8,16)
               → receptive field ~256px, captures screentone patterns
    Decoder: 3 levels (192→96→48), bilinear upsample + skip concat
    Output: 3ch RGB [0,1], composited with original

Params: ~1.5M (~6MB ONNX)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv3x3 → InstanceNorm → LeakyReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, dilation: int = 1):
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding,
                              dilation=dilation, bias=False)
        self.norm = nn.InstanceNorm2d(out_ch, affine=True)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class EncoderLevel(nn.Module):
    """Two ConvBlocks + stride-2 downsample."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block1 = ConvBlock(in_ch, out_ch)
        self.block2 = ConvBlock(out_ch, out_ch)
        self.down = nn.Conv2d(out_ch, out_ch, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.block1(x)
        h = self.block2(h)
        skip = h
        return self.down(h), skip


class DecoderLevel(nn.Module):
    """Upsample + concat skip + two ConvBlocks."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.block1 = ConvBlock(in_ch + skip_ch, out_ch)
        self.block2 = ConvBlock(out_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        h = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        h = torch.cat([h, skip], dim=1)
        return self.block2(self.block1(h))


class DilatedBottleneck(nn.Module):
    """Residual dilated conv stack for wide receptive field.
    Dilations [1, 2, 4, 8, 16] → ~256px effective receptive field.
    """

    def __init__(self, channels: int, dilations: tuple[int, ...] = (1, 2, 4, 8, 16)):
        super().__init__()
        self.blocks = nn.ModuleList([
            ConvBlock(channels, channels, kernel_size=3, dilation=d)
            for d in dilations
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = x + block(x)
        return x


class SmallInpaintModel(nn.Module):
    """Lightweight manga inpainting UNet.

    Input:  image [B,3,H,W] in [0,1] + mask [B,1,H,W] in {0,1} (1=inpaint, 0=keep)
    Output: inpainted [B,3,H,W] in [0,1]
    """

    def __init__(self, base_ch: int = 48):
        super().__init__()
        c = base_ch

        self.input_proj = nn.Conv2d(4, c, 3, 1, 1)

        # Encoder: 3 levels
        self.enc1 = EncoderLevel(c, c)          # H→H/2
        self.enc2 = EncoderLevel(c, c * 2)      # H/2→H/4
        self.enc3 = EncoderLevel(c * 2, c * 4)  # H/4→H/8

        # Bottleneck
        self.bottleneck = DilatedBottleneck(c * 4)

        # Decoder: 3 levels
        self.dec3 = DecoderLevel(c * 4, c * 4, c * 2)  # H/8→H/4
        self.dec2 = DecoderLevel(c * 2, c * 2, c)       # H/4→H/2
        self.dec1 = DecoderLevel(c, c, c)                # H/2→H

        # Output head
        self.output_proj = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, 3, 1),
            nn.Sigmoid(),
        )

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        inv_mask = 1.0 - mask
        masked_img = image * inv_mask
        x = self.input_proj(torch.cat([masked_img, mask], dim=1))

        x, s1 = self.enc1(x)
        x, s2 = self.enc2(x)
        x, s3 = self.enc3(x)

        x = self.bottleneck(x)

        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        pred = self.output_proj(x)
        return pred * mask + image * inv_mask


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = SmallInpaintModel(base_ch=48)
    print(f"Parameters: {count_parameters(model):,}")

    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"Model size: {param_bytes / 1024 / 1024:.2f} MB")

    img = torch.rand(1, 3, 512, 512)
    mask = (torch.rand(1, 1, 512, 512) > 0.7).float()
    with torch.no_grad():
        out = model(img, mask)
    print(f"Output: {out.shape}, range=[{out.min():.3f}, {out.max():.3f}]")
