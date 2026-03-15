"""Native PyTorch LaMa FFC-ResNet inpainting model.

Conv2d-based DFT replaces torch.fft.rfft2/irfft2 so the exported ONNX graph
has ZERO FFT ops — eliminates 288 CPU↔GPU memory copies in ONNX Runtime.

Architecture matches lama-manga.safetensors (204 MB):
  - 4-channel input (image [0,1] concat mask [0,1])
  - 3 downsampling + 18 FFC ResBlocks + 3 upsampling + sigmoid
  - FourierUnit uses precomputed DFT matrices (matmul, no torch.fft)

I/O convention (same as existing lama_fp32.onnx):
  image: [B, 3, H, W] float32 [0, 1]
  mask:  [B, 1, H, W] float32 {0, 1}  (1=inpaint, 0=keep)
  output: [B, 3, H, W] float32 [0, 1]
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Conv2d-based DFT (replaces torch.fft.rfft2 / irfft2)
# ---------------------------------------------------------------------------

def _build_dft_matrix(N: int) -> torch.Tensor:
    """Build DFT matrix [N, N] complex → stored as [N, N, 2] (cos, sin).

    DFT[k, n] = exp(-j * 2π * k * n / N)
              = (cos(2π*k*n/N), -sin(2π*k*n/N))
    """
    n = torch.arange(N, dtype=torch.float32)
    k = torch.arange(N, dtype=torch.float32)
    angles = -2.0 * math.pi * k.unsqueeze(1) * n.unsqueeze(0) / N  # [N, N]
    # Stack cos, sin
    return torch.stack([angles.cos(), angles.sin()], dim=-1)  # [N, N, 2]


def _build_idft_matrix(N: int) -> torch.Tensor:
    """Build IDFT matrix [N, N, 2].

    IDFT[n, k] = (1/N) * exp(j * 2π * k * n / N)
               = (1/N) * (cos(2π*k*n/N), sin(2π*k*n/N))
    """
    n = torch.arange(N, dtype=torch.float32)
    k = torch.arange(N, dtype=torch.float32)
    angles = 2.0 * math.pi * k.unsqueeze(0) * n.unsqueeze(1) / N  # [N, N]
    return torch.stack([angles.cos(), angles.sin()], dim=-1) / N  # [N, N, 2]


class ConvDFT2d(nn.Module):
    """rfft2 + irfft2 replacement using Conv1d(kernel_size=1).

    Each axis-wise DFT is a fixed linear map along that axis. By moving the
    transform axis into the channel dimension, it becomes a 1×1 convolution
    (channel mixing at each position). CoreML natively supports Conv1d,
    eliminating the MatMul+Reshape partition splits.

    For ortho normalization, DFT is scaled by 1/sqrt(N) and IDFT by sqrt(N)/N = 1/sqrt(N).
    rfft2 returns only W//2+1 frequency bins along the last axis.
    """

    def __init__(self, H: int, W: int):
        super().__init__()
        self.H = H
        self.W = W
        W_half = W // 2 + 1
        self.W_half = W_half

        scale_h = 1.0 / math.sqrt(H)
        scale_w = 1.0 / math.sqrt(W)

        # DFT matrices
        dft_h = _build_dft_matrix(H)  # [H, H, 2]
        dft_h_cos = dft_h[:, :, 0] * scale_h  # [H, H]
        dft_h_sin = dft_h[:, :, 1] * scale_h  # [H, H]

        dft_w = _build_dft_matrix(W)  # [W, W, 2]
        dft_w_cos = dft_w[:W_half, :, 0] * scale_w  # [W_half, W]
        dft_w_sin = dft_w[:W_half, :, 1] * scale_w  # [W_half, W]

        # IDFT matrices
        idft_h = _build_idft_matrix(H)  # [H, H, 2]
        idft_h_cos = idft_h[:, :, 0] * H * scale_h  # [H, H]
        idft_h_sin = idft_h[:, :, 1] * H * scale_h  # [H, H]

        M_re, M_im = self._build_irfft_w_matrices(W, W_half, scale_w)

        # Build Conv1d weight buffers (all kernel_size=1, stored as [out, in, 1])

        # rfft2 step 1: real→complex DFT along H
        #   input [N, H, W], output [N, 2H, W]
        self.register_buffer('w_h_dft',
            torch.cat([dft_h_cos, dft_h_sin], dim=0).unsqueeze(-1))  # [2H, H, 1]

        # rfft2 step 2: complex DFT along W (axis transposed into channels)
        #   input [N, 2W, H] = [real; imag], output [N, 2W_half, H]
        #   [out_re]   [ cos  -sin ] [in_re]
        #   [out_im] = [ sin   cos ] [in_im]
        top = torch.cat([dft_w_cos, -dft_w_sin], dim=1)   # [W_half, 2W]
        bot = torch.cat([dft_w_sin,  dft_w_cos], dim=1)   # [W_half, 2W]
        self.register_buffer('w_w_dft',
            torch.cat([top, bot], dim=0).unsqueeze(-1))    # [2W_half, 2W, 1]

        # irfft2 step 1: complex IFFT along H
        #   input [N, 2H, W_half] = [real; imag], output [N, 2H, W_half]
        top = torch.cat([idft_h_cos, -idft_h_sin], dim=1)  # [H, 2H]
        bot = torch.cat([idft_h_sin,  idft_h_cos], dim=1)  # [H, 2H]
        self.register_buffer('w_h_idft',
            torch.cat([top, bot], dim=0).unsqueeze(-1))     # [2H, 2H, 1]

        # irfft2 step 2: complex→real IRFFT along W (Hermitian reconstruction)
        #   input [N, 2W_half, H] = [real; imag], output [N, W, H]
        #   out = M_re @ real + M_im @ imag
        self.register_buffer('w_w_irfft',
            torch.cat([M_re, M_im], dim=1).unsqueeze(-1))  # [W, 2W_half, 1]

    @staticmethod
    def _build_irfft_w_matrices(W: int, W_half: int, scale: float):
        """Build real-valued matrices for irfft along W dimension.

        Uses Hermitian symmetry: k=0 and k=W//2 have factor 1, middle bins factor 2.
        """
        n = torch.arange(W, dtype=torch.float32)
        k = torch.arange(W_half, dtype=torch.float32)
        angles = 2.0 * math.pi * k.unsqueeze(0) * n.unsqueeze(1) / W  # [W, W_half]

        factor = torch.ones(W_half)
        factor[1:W_half - (1 if W % 2 == 0 else 0)] = 2.0

        M_re = angles.cos() * factor.unsqueeze(0) * scale   # [W, W_half]
        M_im = -angles.sin() * factor.unsqueeze(0) * scale  # [W, W_half]
        return M_re, M_im

    def rfft2_ortho(self, x: torch.Tensor, channels: int):
        """Compute rfft2 with ortho normalization using Conv1d.

        Input:  x [B, C, H, W]
        Output: (real, imag) each [B, C, H, W//2+1]
        """
        # Export/inference uses fixed batch=1 tiles.
        B = 1
        C = channels
        H = self.H
        W = self.W
        W_half = self.W_half
        N = C

        # Step 1: DFT along H — H is channel dim, W is sequence dim
        x_flat = x.reshape(N, H, W)                            # [N, H, W]
        h = F.conv1d(x_flat, self.w_h_dft)                     # [N, 2H, W]
        h_real, h_imag = torch.split(h, [H, H], dim=1)         # each [N, H, W]

        # Step 2: DFT along W — transpose W into channels, H becomes sequence
        z = torch.cat([h_real.transpose(1, 2),
                       h_imag.transpose(1, 2)], dim=1).contiguous()  # [N, 2W, H]
        y = F.conv1d(z, self.w_w_dft)                          # [N, 2W_half, H]
        y_real, y_imag = torch.split(y, [W_half, W_half], dim=1)
        y_real = y_real.transpose(1, 2).contiguous()           # [N, H, W_half]
        y_imag = y_imag.transpose(1, 2).contiguous()           # [N, H, W_half]

        return y_real.reshape(B, C, H, W_half), y_imag.reshape(B, C, H, W_half)

    def irfft2_ortho(self, real: torch.Tensor, imag: torch.Tensor, channels: int):
        """Compute irfft2 with ortho normalization using Conv1d.

        Input:  real, imag each [B, C, H, W//2+1]
        Output: x [B, C, H, W]
        """
        B = 1
        C = channels
        H = self.H
        W_half = self.W_half
        N = C

        # Step 1: IFFT along H (complex→complex) — H is channel dim
        z = torch.cat([real.reshape(N, H, W_half),
                       imag.reshape(N, H, W_half)], dim=1)    # [N, 2H, W_half]
        h = F.conv1d(z, self.w_h_idft)                         # [N, 2H, W_half]
        h_real, h_imag = torch.split(h, [H, H], dim=1)         # each [N, H, W_half]

        # Step 2: IRFFT along W (complex→real) — transpose W_half into channels
        z = torch.cat([h_real.transpose(1, 2),
                       h_imag.transpose(1, 2)], dim=1).contiguous()  # [N, 2W_half, H]
        x = F.conv1d(z, self.w_w_irfft)                        # [N, W, H]
        x = x.transpose(1, 2).contiguous()                     # [N, H, W]

        return x.reshape(B, C, H, self.W)

# ---------------------------------------------------------------------------
# FourierUnit — uses ConvDFT2d instead of torch.fft
# ---------------------------------------------------------------------------

class FourierUnit(nn.Module):
    """FourierUnit with Conv1d-based DFT (no torch.fft ops).

    Forward: rfft2(x) → cat(real, imag) on channel dim → 1x1 Conv → BN → ReLU
             → split real/imag → irfft2 → output
    """

    def __init__(self, in_channels: int, out_channels: int, H: int, W: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_layer = nn.Conv2d(in_channels * 2, out_channels * 2,
                                    kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)
        self.dft = ConvDFT2d(H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = 1
        C = self.in_channels
        H = self.dft.H
        W_half = self.dft.W_half

        # rfft2 with ortho normalization
        real, imag = self.dft.rfft2_ortho(x, C)  # each [B, C, H, W_half]

        # Permute and reshape to match LaMa convention:
        # LaMa does: stack(real, imag, dim=-1) → [B, C, H, W_half, 2]
        # then permute(0,1,4,2,3) → [B, C, 2, H, W_half]
        # then view(B, C*2, H, W_half)
        ffted = torch.stack([real, imag], dim=2)  # [B, C, 2, H, W_half]
        ffted = ffted.reshape(B, C * 2, H, W_half)  # [B, C*2, H, W_half]

        # 1x1 conv + BN + ReLU
        ffted = self.conv_layer(ffted)  # [B, out_ch*2, H, W_half]
        ffted = self.relu(self.bn(ffted))

        # Split back to real/imag
        # LaMa does: view(B, out_ch, 2, H, W_half) then permute(0,1,3,4,2)
        # then complex(ffted[...,0], ffted[...,1])
        ffted = ffted.reshape(B, self.out_channels, 2, H, W_half)
        out_real, out_imag = torch.split(ffted, [1, 1], dim=2)
        out_real = out_real.squeeze(2)  # [B, out_ch, H, W_half]
        out_imag = out_imag.squeeze(2)  # [B, out_ch, H, W_half]

        # irfft2 with ortho normalization
        output = self.dft.irfft2_ortho(out_real, out_imag, self.out_channels)  # [B, out_ch, H, W]
        return output


# ---------------------------------------------------------------------------
# SpectralTransform
# ---------------------------------------------------------------------------

class SpectralTransform(nn.Module):
    """SpectralTransform with Conv2d-based DFT. No LFU (enable_lfu=False)."""

    def __init__(self, in_channels: int, out_channels: int, H: int, W: int):
        super().__init__()
        half_ch = out_channels // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, half_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(half_ch),
            nn.ReLU(inplace=True),
        )
        self.fu = FourierUnit(half_ch, half_ch, H, W)
        self.conv2 = nn.Conv2d(half_ch, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        fu_out = self.fu(x)
        return self.conv2(x + fu_out)


# ---------------------------------------------------------------------------
# FFC building blocks
# ---------------------------------------------------------------------------

class Conv2dPad(nn.Module):
    """Conv2d with explicit reflect padding (matches LaMa's padding_type='reflect')."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 stride: int = 1, padding: int = 0):
        super().__init__()
        self.pad = padding
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride,
                              padding=0, bias=False)

    @staticmethod
    def _reflect_pad_2d(x: torch.Tensor, pad: int) -> torch.Tensor:
        """Export-friendly reflect padding using Slice+Concat (no ONNX Pad op)."""
        if pad <= 0:
            return x

        left = x[:, :, :, 1:pad + 1].flip(-1)
        right = x[:, :, :, -pad - 1:-1].flip(-1)
        x = torch.cat([left, x, right], dim=-1)

        top = x[:, :, 1:pad + 1, :].flip(-2)
        bottom = x[:, :, -pad - 1:-1, :].flip(-2)
        return torch.cat([top, x, bottom], dim=-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad > 0:
            x = self._reflect_pad_2d(x, self.pad)
        return self.conv(x)


class ConvTranspose2dOutPad1Compat(nn.ConvTranspose2d):
    """Exact replacement for ConvTranspose2d(..., output_padding=1)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 stride: int, padding: int, bias: bool = True):
        super().__init__(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=0,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_w = x[:, :, :, :1] * 0.0
        x = torch.cat([x, z_w], dim=-1)
        z_h = x[:, :, :1, :] * 0.0
        x = torch.cat([x, z_h], dim=-2)
        x = super().forward(x)
        return x[:, :, :-1, :-1]


class FFC(nn.Module):
    """Fast Fourier Convolution: 4-path local/global transform.

    Paths: l2l, l2g, g2l, g2g (SpectralTransform for g2g).
    Some paths may be Identity if ratio_gin=0 or ratio_gout=0.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int, padding: int,
                 ratio_gin: float, ratio_gout: float,
                 spectral_H: int = 0, spectral_W: int = 0):
        super().__init__()
        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        # Local-to-local
        if in_cl > 0 and out_cl > 0:
            self.convl2l = Conv2dPad(in_cl, out_cl, kernel_size, stride, padding)
        else:
            self.convl2l = None

        # Local-to-global
        if in_cl > 0 and out_cg > 0:
            self.convl2g = Conv2dPad(in_cl, out_cg, kernel_size, stride, padding)
        else:
            self.convl2g = None

        # Global-to-local
        if in_cg > 0 and out_cl > 0:
            self.convg2l = Conv2dPad(in_cg, out_cl, kernel_size, stride, padding)
        else:
            self.convg2l = None

        # Global-to-global (SpectralTransform)
        if in_cg > 0 and out_cg > 0:
            self.convg2g = SpectralTransform(in_cg, out_cg, spectral_H, spectral_W)
        else:
            self.convg2g = None

    def forward(self, x):
        x_l, x_g = x if isinstance(x, tuple) else (x, 0)
        out_xl, out_xg = 0, 0

        if self.ratio_gout != 1 and self.convl2l is not None:
            out_xl = self.convl2l(x_l)
            if self.convg2l is not None and torch.is_tensor(x_g):
                out_xl = out_xl + self.convg2l(x_g)

        if self.ratio_gout != 0:
            if self.convl2g is not None:
                out_xg = self.convl2g(x_l)
            if self.convg2g is not None and torch.is_tensor(x_g):
                out_xg = out_xg + self.convg2g(x_g)

        return out_xl, out_xg


class FFCBnAct(nn.Module):
    """FFC + BatchNorm + ReLU for each branch."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int, padding: int,
                 ratio_gin: float, ratio_gout: float,
                 spectral_H: int = 0, spectral_W: int = 0):
        super().__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size, stride, padding,
                       ratio_gin, ratio_gout, spectral_H, spectral_W)

        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg

        self.bn_l = nn.BatchNorm2d(out_cl) if out_cl > 0 else nn.Identity()
        self.bn_g = nn.BatchNorm2d(out_cg) if out_cg > 0 else nn.Identity()
        self.act_l = nn.ReLU(inplace=True) if out_cl > 0 else nn.Identity()
        self.act_g = nn.ReLU(inplace=True) if out_cg > 0 else nn.Identity()

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g


class FFCResBlock(nn.Module):
    """FFC ResNet block: conv1 → conv2 + skip connection."""

    def __init__(self, channels: int, ratio_g: float,
                 spectral_H: int, spectral_W: int):
        super().__init__()
        self.conv1 = FFCBnAct(channels, channels, kernel_size=3, stride=1, padding=1,
                               ratio_gin=ratio_g, ratio_gout=ratio_g,
                               spectral_H=spectral_H, spectral_W=spectral_W)
        self.conv2 = FFCBnAct(channels, channels, kernel_size=3, stride=1, padding=1,
                               ratio_gin=ratio_g, ratio_gout=ratio_g,
                               spectral_H=spectral_H, spectral_W=spectral_W)

    def forward(self, x):
        x_l, x_g = x if isinstance(x, tuple) else (x, 0)
        id_l, id_g = x_l, x_g
        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))
        x_l = id_l + x_l
        x_g = id_g + x_g if torch.is_tensor(id_g) else x_g
        return x_l, x_g


# ---------------------------------------------------------------------------
# Full LaMa Generator
# ---------------------------------------------------------------------------

class LamaGenerator(nn.Module):
    """LaMa FFC-ResNet generator for inpainting.

    Architecture (from lama-manga.safetensors):
      - ReflectionPad2d(3) + FFC_BN_ACT(4→64, k=7, s=1, p=0)      [model.0, model.1]
      - FFC_BN_ACT(64→128, k=3, s=2, p=1)                          [model.2]
      - FFC_BN_ACT(128→256, k=3, s=2, p=1)                         [model.3]
      - FFC_BN_ACT(256→512, k=3, s=2, p=1, ratio_gout=0.75)        [model.4]
      - 18× FFCResBlock(512, ratio=0.75)                            [model.5-22]
      - ConcatTupleLayer                                             [model.23]
      - ConvTranspose2d(512→256, k=3, s=2, p=1, op=1) + BN + ReLU  [model.24-26]
      - ConvTranspose2d(256→128, k=3, s=2, p=1, op=1) + BN + ReLU  [model.27-29]
      - ConvTranspose2d(128→64, k=3, s=2, p=1, op=1) + BN + ReLU   [model.30-32]
      - ReflectionPad2d(3) + Conv2d(64→3, k=7, p=0, bias=True)      [model.33-34]
      - Sigmoid
    """

    def __init__(self, input_nc: int = 4, ngf: int = 64,
                 n_downsampling: int = 3, n_blocks: int = 18):
        super().__init__()
        # Bottleneck spatial size for 512 input: 512 / 2^3 = 64
        spectral_H = 64
        spectral_W = 64
        # ratio_g for bottleneck: 384/(128+384) = 0.75
        ratio_g = 0.75

        # model.0: ReflectionPad2d(3)
        self.pad_init = nn.ReflectionPad2d(3)

        # model.1: FFC_BN_ACT init (local only, ratio_gin=0, ratio_gout=0)
        self.ffc_init = FFCBnAct(input_nc, ngf, kernel_size=7, stride=1, padding=0,
                                  ratio_gin=0, ratio_gout=0)

        # model.2: downsample 1 (local only)
        self.down1 = FFCBnAct(ngf, ngf * 2, kernel_size=3, stride=2, padding=1,
                               ratio_gin=0, ratio_gout=0)

        # model.3: downsample 2 (local only)
        self.down2 = FFCBnAct(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1,
                               ratio_gin=0, ratio_gout=0)

        # model.4: downsample 3 (local → local+global, ratio_gout=0.75)
        # Note: total output channels = 512, but ngf*4*2 = 512
        # in_ch=256, out_ch=512, ratio_gin=0, ratio_gout=0.75
        # out_cl = 512 * 0.25 = 128, out_cg = 512 * 0.75 = 384
        self.down3 = FFCBnAct(ngf * 4, ngf * 8, kernel_size=3, stride=2, padding=1,
                               ratio_gin=0, ratio_gout=ratio_g,
                               spectral_H=spectral_H, spectral_W=spectral_W)

        # model.5-22: 18 FFCResBlocks
        self.resblocks = nn.ModuleList([
            FFCResBlock(ngf * 8, ratio_g, spectral_H, spectral_W)
            for _ in range(n_blocks)
        ])

        # model.23: ConcatTupleLayer (concat local + global)
        # model.24-26: upsample 1
        self.up1 = nn.Sequential(
            ConvTranspose2dOutPad1Compat(ngf * 8, ngf * 4, kernel_size=3,
                                         stride=2, padding=1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True),
        )
        # model.27-29: upsample 2
        self.up2 = nn.Sequential(
            ConvTranspose2dOutPad1Compat(ngf * 4, ngf * 2, kernel_size=3,
                                         stride=2, padding=1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
        )
        # model.30-32: upsample 3
        self.up3 = nn.Sequential(
            ConvTranspose2dOutPad1Compat(ngf * 2, ngf, kernel_size=3,
                                         stride=2, padding=1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
        )

        # model.33-34: output conv
        self.pad_out = nn.ReflectionPad2d(3)
        self.conv_out = nn.Conv2d(ngf, 3, kernel_size=7, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x = Conv2dPad._reflect_pad_2d(x, 3)
        x = self.ffc_init(x)  # returns (x_l, x_g) with x_g=0

        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)  # now x_g is a tensor

        # ResBlocks
        for block in self.resblocks:
            x = block(x)

        # Concat local + global
        x_l, x_g = x
        x = torch.cat([x_l, x_g], dim=1)

        # Decoder
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)

        x = Conv2dPad._reflect_pad_2d(x, 3)
        x = self.conv_out(x)
        x = self.sigmoid(x)
        return x


# ---------------------------------------------------------------------------
# LaMa wrapper with I/O convention matching existing ONNX
# ---------------------------------------------------------------------------

class LamaInpainting(nn.Module):
    """Full LaMa inpainting model with I/O matching existing lama_fp32.onnx.

    Input:
        image: [B, 3, H, W] float32 [0, 1]
        mask:  [B, 1, H, W] float32 {0, 1} (1=inpaint, 0=keep)
    Output:
        [B, 3, H, W] float32 [0, 1]
    """

    def __init__(self):
        super().__init__()
        self.generator = LamaGenerator(input_nc=4, ngf=64,
                                       n_downsampling=3, n_blocks=18)

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Preprocess: zero out inpaint region, then concat with mask
        # (matches existing ONNX: image * (1 - mask) concat mask)
        masked_image = image * (1.0 - mask)
        x = torch.cat([masked_image, mask], dim=1)
        return self.generator(x)


# ---------------------------------------------------------------------------
# Weight loading from safetensors
# ---------------------------------------------------------------------------

def _map_safetensors_to_model(model: LamaInpainting) -> dict:
    """Build mapping from safetensors keys → model state_dict keys."""
    mapping = {}

    # Helper to map FFC conv keys
    def map_ffc_conv(st_prefix: str, mod_prefix: str, has_g2g: bool = False,
                     has_g2l: bool = False, has_l2g: bool = False):
        """Map FFC conv weight keys."""
        mapping[f'{st_prefix}.ffc.convl2l.weight'] = f'{mod_prefix}.ffc.convl2l.conv.weight'

        if has_l2g:
            mapping[f'{st_prefix}.ffc.convl2g.weight'] = f'{mod_prefix}.ffc.convl2g.conv.weight'
        if has_g2l:
            mapping[f'{st_prefix}.ffc.convg2l.weight'] = f'{mod_prefix}.ffc.convg2l.conv.weight'

        if has_g2g:
            g2g_st = f'{st_prefix}.ffc.convg2g'
            g2g_mod = f'{mod_prefix}.ffc.convg2g'
            mapping[f'{g2g_st}.conv1.0.weight'] = f'{g2g_mod}.conv1.0.weight'
            for suffix in ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']:
                mapping[f'{g2g_st}.conv1.1.{suffix}'] = f'{g2g_mod}.conv1.1.{suffix}'
            mapping[f'{g2g_st}.fu.conv_layer.weight'] = f'{g2g_mod}.fu.conv_layer.weight'
            for suffix in ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']:
                mapping[f'{g2g_st}.fu.bn.{suffix}'] = f'{g2g_mod}.fu.bn.{suffix}'
            mapping[f'{g2g_st}.conv2.weight'] = f'{g2g_mod}.conv2.weight'

        # BN
        for suffix in ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']:
            mapping[f'{st_prefix}.bn_l.{suffix}'] = f'{mod_prefix}.bn_l.{suffix}'

        if has_l2g or has_g2g:
            for suffix in ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']:
                mapping[f'{st_prefix}.bn_g.{suffix}'] = f'{mod_prefix}.bn_g.{suffix}'

    # model.1 → generator.ffc_init
    map_ffc_conv('model.1', 'generator.ffc_init')

    # model.2 → generator.down1
    map_ffc_conv('model.2', 'generator.down1')

    # model.3 → generator.down2
    map_ffc_conv('model.3', 'generator.down2')

    # model.4 → generator.down3 (has l2g, but no g2l or g2g since ratio_gin=0)
    map_ffc_conv('model.4', 'generator.down3', has_l2g=True)

    # model.5-22 → generator.resblocks[0-17]
    for i in range(18):
        st_idx = i + 5
        for conv_name in ['conv1', 'conv2']:
            st_pref = f'model.{st_idx}.{conv_name}'
            mod_pref = f'generator.resblocks.{i}.{conv_name}'
            map_ffc_conv(st_pref, mod_pref,
                         has_g2g=True, has_g2l=True, has_l2g=True)

    # model.24 → generator.up1[0] (ConvTranspose2d)
    mapping['model.24.weight'] = 'generator.up1.0.weight'
    mapping['model.24.bias'] = 'generator.up1.0.bias'
    for suffix in ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']:
        mapping[f'model.25.{suffix}'] = f'generator.up1.1.{suffix}'

    # model.27 → generator.up2[0]
    mapping['model.27.weight'] = 'generator.up2.0.weight'
    mapping['model.27.bias'] = 'generator.up2.0.bias'
    for suffix in ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']:
        mapping[f'model.28.{suffix}'] = f'generator.up2.1.{suffix}'

    # model.30 → generator.up3[0]
    mapping['model.30.weight'] = 'generator.up3.0.weight'
    mapping['model.30.bias'] = 'generator.up3.0.bias'
    for suffix in ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']:
        mapping[f'model.31.{suffix}'] = f'generator.up3.1.{suffix}'

    # model.34 → generator.conv_out
    mapping['model.34.weight'] = 'generator.conv_out.weight'
    mapping['model.34.bias'] = 'generator.conv_out.bias'

    return mapping


def load_from_safetensors(safetensors_path: str) -> LamaInpainting:
    """Load LaMa model from safetensors file."""
    from safetensors import safe_open

    model = LamaInpainting()
    state_dict = model.state_dict()
    mapping = _map_safetensors_to_model(model)

    f = safe_open(safetensors_path, framework='pt', device='cpu')
    st_keys = set(f.keys())

    loaded = 0
    for st_key, mod_key in mapping.items():
        if st_key not in st_keys:
            print(f"WARNING: Missing safetensors key: {st_key}")
            continue
        if mod_key not in state_dict:
            print(f"WARNING: Missing model key: {mod_key}")
            continue
        tensor = f.get_tensor(st_key)
        if tensor.shape != state_dict[mod_key].shape:
            print(f"WARNING: Shape mismatch for {st_key}: "
                  f"safetensors {list(tensor.shape)} vs model {list(state_dict[mod_key].shape)}")
            continue
        state_dict[mod_key] = tensor
        loaded += 1

    # Check for unmapped safetensors keys
    mapped_st_keys = set(mapping.keys())
    unmapped = st_keys - mapped_st_keys
    if unmapped:
        print(f"WARNING: {len(unmapped)} unmapped safetensors keys:")
        for k in sorted(unmapped)[:10]:
            print(f"  {k}")

    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded {loaded}/{len(mapping)} weights from {safetensors_path}")
    return model


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_onnx(model: LamaInpainting, output_path: str):
    """Export model to ONNX with fixed 512×512 input."""
    import onnx

    model.eval()
    device = next(model.parameters()).device

    dummy_image = torch.rand(1, 3, 512, 512, device=device)
    dummy_mask = (torch.rand(1, 1, 512, 512, device=device) > 0.5).float()

    with torch.no_grad():
        pt_output = model(dummy_image, dummy_mask)

    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        model,
        (dummy_image, dummy_mask),
        output_path,
        opset_version=17,
        input_names=["image", "mask"],
        output_names=["output"],
        dynamic_axes=None,
        do_constant_folding=True,
    )

    # Verify
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    # Check for FFT ops
    fft_ops = [n.op_type for n in onnx_model.graph.node
               if 'fft' in n.op_type.lower() or 'dft' in n.op_type.lower()]
    if fft_ops:
        print(f"ERROR: Found FFT ops in ONNX: {fft_ops}")
    else:
        print("OK: No FFT/DFT ops in ONNX graph")

    # Verify with ONNX Runtime
    import onnxruntime as ort
    sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    ort_out = sess.run(["output"], {
        "image": dummy_image.cpu().numpy(),
        "mask": dummy_mask.cpu().numpy(),
    })[0]

    ort_out_t = torch.from_numpy(ort_out).to(device)
    max_diff = (pt_output - ort_out_t).abs().max().item()
    print(f"PyTorch vs ONNX: max_diff={max_diff:.6e}")

    import os
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"Model size: {size_mb:.2f} MB")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="LaMa model with Conv-based DFT")
    p.add_argument("--safetensors", type=str, default="../../models/lama-manga.safetensors")
    p.add_argument("--reference_onnx", type=str, default="../../models/lama_fp32.onnx")
    p.add_argument("--export", type=str, default=None,
                   help="Export to ONNX at this path")
    p.add_argument("--verify", action="store_true", default=True,
                   help="Verify against reference ONNX")
    args = p.parse_args()

    print("Loading model from safetensors...")
    model = load_from_safetensors(args.safetensors)

    if args.verify and args.reference_onnx:
        print("\nVerifying against reference ONNX...")
        import onnxruntime as ort
        import numpy as np

        ref_sess = ort.InferenceSession(args.reference_onnx,
                                        providers=["CPUExecutionProvider"])

        # Test with random input
        np.random.seed(42)
        test_img = np.random.rand(1, 3, 512, 512).astype(np.float32)
        test_mask = (np.random.rand(1, 1, 512, 512) > 0.5).astype(np.float32)

        ref_out = ref_sess.run(["output"], {
            "image": test_img, "mask": test_mask
        })[0]

        with torch.no_grad():
            pt_out = model(
                torch.from_numpy(test_img),
                torch.from_numpy(test_mask),
            ).numpy()

        max_diff = np.abs(pt_out - ref_out).max()
        mean_diff = np.abs(pt_out - ref_out).mean()
        print(f"Max diff:  {max_diff:.6e}")
        print(f"Mean diff: {mean_diff:.6e}")
        if max_diff < 1e-4:
            print("PASS: Output matches reference ONNX")
        elif max_diff < 1e-3:
            print("WARN: Small differences, likely float precision")
        else:
            print("FAIL: Large difference from reference ONNX!")

    if args.export:
        export_onnx(model, args.export)
