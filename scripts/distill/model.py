"""MI-GAN generator and discriminator for inpainting distillation.

Architecture based on MI-GAN (ICCV 2023):
- UNet encoder-decoder with depthwise-separable convolutions
- Re-parameterizable blocks (RepVGG-style multi-branch → fused single conv)
- Noise injection in decoder for texture generation
- Input: 4ch [mask, masked_rgb], Output: 3ch RGB in [-1,1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class SeparableConv2d(nn.Module):
    """Depthwise-separable convolution (depthwise + pointwise)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, bias: bool = True):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding,
                                   groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class RepConvBlock(nn.Module):
    """Re-parameterizable convolution block (RepVGG-style).

    During training: 3x3 conv + 1x1 conv + identity (if in_ch==out_ch),
    all summed. During inference: fused into a single 3x3 conv.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, use_act: bool = True):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.stride = stride
        self.use_act = use_act
        self.fused = False

        # Training branches
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride, 0, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.identity = (
            nn.BatchNorm2d(out_ch) if (in_ch == out_ch and stride == 1) else None
        )
        self.act = nn.LeakyReLU(0.2, inplace=True) if use_act else nn.Identity()

        # Placeholder for fused weight
        self.fused_conv: Optional[nn.Conv2d] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused and self.fused_conv is not None:
            return self.act(self.fused_conv(x))

        out = self.conv3x3(x) + self.conv1x1(x)
        if self.identity is not None:
            out = out + self.identity(x)
        return self.act(out)

    def _fuse_bn(self, conv: nn.Conv2d, bn: nn.BatchNorm2d):
        """Fuse conv + BN into a single conv with bias."""
        w = conv.weight
        mean = bn.running_mean
        var_sqrt = torch.sqrt(bn.running_var + bn.eps)
        gamma = bn.weight
        beta = bn.bias
        w_fused = w * (gamma / var_sqrt).reshape(-1, 1, 1, 1)
        b_fused = beta - mean * gamma / var_sqrt
        return w_fused, b_fused

    def _pad_1x1_to_3x3(self, w1x1: torch.Tensor) -> torch.Tensor:
        return F.pad(w1x1, [1, 1, 1, 1])

    def _identity_to_conv(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Create a 3x3 identity convolution kernel."""
        assert self.identity is not None
        bn = self.identity
        # identity conv: each input channel maps to same output channel
        w = torch.zeros(self.out_ch, self.out_ch, 3, 3, device=bn.weight.device)
        for i in range(self.out_ch):
            w[i, i, 1, 1] = 1.0
        mean = bn.running_mean
        var_sqrt = torch.sqrt(bn.running_var + bn.eps)
        gamma = bn.weight
        beta = bn.bias
        w_fused = w * (gamma / var_sqrt).reshape(-1, 1, 1, 1)
        b_fused = beta - mean * gamma / var_sqrt
        return w_fused, b_fused

    def fuse_reparam(self):
        """Fuse multi-branch into single 3x3 conv for inference."""
        if self.fused:
            return

        w3, b3 = self._fuse_bn(self.conv3x3[0], self.conv3x3[1])
        w1, b1 = self._fuse_bn(self.conv1x1[0], self.conv1x1[1])
        w1 = self._pad_1x1_to_3x3(w1)

        w_final = w3 + w1
        b_final = b3 + b1

        if self.identity is not None:
            wi, bi = self._identity_to_conv()
            w_final = w_final + wi
            b_final = b_final + bi

        fused = nn.Conv2d(self.in_ch, self.out_ch, 3, self.stride, 1, bias=True)
        fused.weight.data = w_final
        fused.bias.data = b_final
        self.fused_conv = fused
        self.fused = True

        # Remove training branches to save memory
        del self.conv3x3
        del self.conv1x1
        if self.identity is not None:
            del self.identity
            self.identity = None


class NoiseInjection(nn.Module):
    """Per-channel learned noise injection (StyleGAN-style)."""

    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        noise = torch.randn(x.size(0), 1, x.size(2), x.size(3),
                            device=x.device, dtype=x.dtype)
        return x + self.weight * noise


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class EncoderBlock(nn.Module):
    """Encoder block: RepConv → SeparableConv → downsample."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.rep = RepConvBlock(in_ch, out_ch, stride=1)
        self.sep = SeparableConv2d(out_ch, out_ch, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.down = nn.Conv2d(out_ch, out_ch, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.rep(x)
        h = self.act(self.sep(h))
        skip = h
        h = self.down(h)
        return h, skip


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class DecoderBlock(nn.Module):
    """Decoder block: upsample → concat skip → RepConv → SeparableConv → noise."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 4, 2, 1)
        self.rep = RepConvBlock(in_ch + skip_ch, out_ch, stride=1)
        self.sep = SeparableConv2d(out_ch, out_ch, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.noise = NoiseInjection(out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        h = self.up(x)
        # Handle size mismatch from non-power-of-2
        if h.shape[2:] != skip.shape[2:]:
            h = F.interpolate(h, size=skip.shape[2:], mode='bilinear',
                              align_corners=False)
        h = torch.cat([h, skip], dim=1)
        h = self.rep(h)
        h = self.noise(self.act(self.sep(h)))
        return h


# ---------------------------------------------------------------------------
# MI-GAN Generator
# ---------------------------------------------------------------------------

class MIGANGenerator(nn.Module):
    """MI-GAN UNet generator for inpainting.

    Input:  4ch tensor [mask_minus_half, masked_r, masked_g, masked_b]
            where mask channel = mask - 0.5 (1=known→0.5, 0=missing→-0.5)
    Output: 3ch RGB in [-1, 1]

    Channel dims (512x512 input):
        enc: 4→48→96→192→384→512
        bottleneck: 512→512
        dec: 512→384→192→96→48
        out: 48→3
    """

    def __init__(self, in_ch: int = 4, base_ch: int = 32):
        super().__init__()
        c = base_ch  # 32

        # Encoder — 4 levels, ~3-5M params total
        self.enc1 = EncoderBlock(in_ch, c)          # 512→256, 32ch
        self.enc2 = EncoderBlock(c, c * 2)           # 256→128, 64ch
        self.enc3 = EncoderBlock(c * 2, c * 4)       # 128→64,  128ch
        self.enc4 = EncoderBlock(c * 4, c * 8)       # 64→32,   256ch

        # Bottleneck
        self.bottleneck = nn.Sequential(
            RepConvBlock(c * 8, c * 8),
            SeparableConv2d(c * 8, c * 8, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Decoder (mirror encoder)
        self.dec4 = DecoderBlock(c * 8, c * 8, c * 4)  # 32→64,  128ch
        self.dec3 = DecoderBlock(c * 4, c * 4, c * 2)  # 64→128, 64ch
        self.dec2 = DecoderBlock(c * 2, c * 2, c)      # 128→256, 32ch
        self.dec1 = DecoderBlock(c, c, c)               # 256→512, 32ch

        # Output head
        self.to_rgb = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, 3, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        h, s1 = self.enc1(x)   # h: 256, s1: 512
        h, s2 = self.enc2(h)   # h: 128, s2: 256
        h, s3 = self.enc3(h)   # h: 64,  s3: 128
        h, s4 = self.enc4(h)   # h: 32,  s4: 64

        # Bottleneck
        h = self.bottleneck(h)

        # Decoder
        h = self.dec4(h, s4)
        h = self.dec3(h, s3)
        h = self.dec2(h, s2)
        h = self.dec1(h, s1)

        return self.to_rgb(h)

    def fuse_reparam(self):
        """Fuse all RepConvBlocks for inference deployment."""
        for module in self.modules():
            if isinstance(module, RepConvBlock):
                module.fuse_reparam()


# ---------------------------------------------------------------------------
# PatchGAN Discriminator
# ---------------------------------------------------------------------------

class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator with spectral normalization.

    Input: 4ch [RGB image (3ch) + mask (1ch)]
    Output: patch-level real/fake scores
    """

    def __init__(self, in_ch: int = 4, base_ch: int = 64, n_layers: int = 3):
        super().__init__()

        layers = [
            nn.utils.spectral_norm(
                nn.Conv2d(in_ch, base_ch, 4, 2, 1)
            ),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        ch = base_ch
        for i in range(1, n_layers):
            prev_ch = ch
            ch = min(ch * 2, 512)
            layers += [
                nn.utils.spectral_norm(
                    nn.Conv2d(prev_ch, ch, 4, 2, 1, bias=False)
                ),
                nn.InstanceNorm2d(ch),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        prev_ch = ch
        ch = min(ch * 2, 512)
        layers += [
            nn.utils.spectral_norm(
                nn.Conv2d(prev_ch, ch, 4, 1, 1, bias=False)
            ),
            nn.InstanceNorm2d(ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, 1, 4, 1, 1),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    gen = MIGANGenerator()
    disc = PatchDiscriminator()

    print(f"Generator params:     {count_parameters(gen):,}")
    print(f"Discriminator params: {count_parameters(disc):,}")

    x = torch.randn(1, 4, 512, 512)
    with torch.no_grad():
        out = gen(x)
    print(f"Generator output shape: {out.shape}")  # [1, 3, 512, 512]

    d_in = torch.randn(1, 4, 512, 512)
    with torch.no_grad():
        d_out = disc(d_in)
    print(f"Discriminator output shape: {d_out.shape}")

    # Test re-param fusion (must use eval mode for correct BN stats)
    gen.eval()
    with torch.no_grad():
        out_pre = gen(x)
    gen.fuse_reparam()
    with torch.no_grad():
        out_post = gen(x)
    diff = (out_pre - out_post).abs().max().item()
    print(f"Re-param fusion max diff: {diff:.6e}")
