"""PP-LCNetV3 backbone in MLX — shared by PP-OCR det and rec.

Port of transformers/models/pp_lcnet_v3/modeling_pp_lcnet_v3.py.
Architecture: stem conv → 5 stages of DepthwiseSeparable blocks.
Each block uses LearnableRepLayer (multi-branch conv → sum → LAB).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


def _hardsigmoid(x: mx.array) -> mx.array:
    return mx.clip(x / 6.0 + 0.5, 0.0, 1.0)


def _hardswish(x: mx.array) -> mx.array:
    return x * _hardsigmoid(x)


_ACT = {
    "hardswish": _hardswish,
    "relu": nn.relu,
    "silu": nn.silu,
}


Stride = int | list[int] | tuple[int, int]


def _stride2(s: Stride) -> tuple[int, int]:
    """Normalize stride to (h, w) tuple."""
    if isinstance(s, (list, tuple)):
        return (s[0], s[1])
    return (s, s)


def _stride_is_1(s: Stride) -> bool:
    h, w = _stride2(s)
    return h == 1 and w == 1


def _stride_is_downscale(s: Stride) -> bool:
    """True if stride is uniform 2 (standard downscale). False for asymmetric."""
    h, w = _stride2(s)
    return h == 2 and w == 2


def _make_divisible(v: int, divisor: int = 8) -> int:
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


@dataclass
class LCNetConfig:
    scale: float = 1.0
    divisor: int = 8
    stem_channels: int = 16
    stem_stride: int = 2
    hidden_act: str = "hardswish"
    conv_symmetric_num: int = 4
    reduction: int = 4
    block_configs: list | None = None
    out_indices: list[int] | None = None

    def __post_init__(self):
        if self.block_configs is None:
            self.block_configs = [
                [[3, 16, 32, 1, False]],
                [[3, 32, 64, 2, False], [3, 64, 64, 1, False]],
                [[3, 64, 128, 2, False], [3, 128, 128, 1, False]],
                [[3, 128, 256, 2, False], [5, 256, 256, 1, False],
                 [5, 256, 256, 1, False], [5, 256, 256, 1, False], [5, 256, 256, 1, False]],
                [[5, 256, 512, 2, True], [5, 512, 512, 1, True],
                 [5, 512, 512, 1, False], [5, 512, 512, 1, False]],
            ]
        if self.out_indices is None:
            self.out_indices = [2, 3, 4, 5]


# ── Layers ───────────────────────────────────────────────────────


class ConvBN(nn.Module):
    """Conv2d + BatchNorm + optional activation."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 stride: Stride = 1, groups: int = 1, act: str | None = "hardswish"):
        super().__init__()
        self.convolution = nn.Conv2d(in_ch, out_ch, kernel, stride=_stride2(stride),
                                     padding=kernel // 2, bias=False)
        self.normalization = nn.BatchNorm(out_ch)
        self._act = _ACT.get(act) if act else None

    def __call__(self, x: mx.array) -> mx.array:
        x = self.normalization(self.convolution(x))
        return self._act(x) if self._act else x


class DepthwiseConvBN(nn.Module):
    """Depthwise Conv2d + BatchNorm (groups = in_channels).

    Uses a nested 'convolution' module to match safetensors key structure:
    `.convolution.weight` instead of `.weight`.
    """

    def __init__(self, channels: int, kernel: int = 3, stride: Stride = 1, act: str | None = None):
        super().__init__()
        self._channels = channels
        self._stride = _stride2(stride)
        self._padding = kernel // 2
        # Wrap weight in a sub-module so the key path is .convolution.weight
        self.convolution = _DepthwiseConvWeight(channels, kernel)
        self.normalization = nn.BatchNorm(channels)
        self._act = _ACT.get(act) if act else None

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.conv2d(x, self.convolution.weight, stride=self._stride,
                      padding=self._padding, groups=self._channels)
        x = self.normalization(x)
        return self._act(x) if self._act else x


class _DepthwiseConvWeight(nn.Module):
    def __init__(self, channels: int, kernel: int):
        super().__init__()
        self.weight = mx.zeros((channels, kernel, kernel, 1))
        nn.init.glorot_uniform()(self.weight)


class LearnableAffineBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = mx.ones((1,))
        self.bias = mx.zeros((1,))

    def __call__(self, x: mx.array) -> mx.array:
        return self.scale * x + self.bias


class LearnableRepLayer(nn.Module):
    """Multi-branch conv → sum → LAB → optional activation.

    At init: keeps branches for weight loading.
    After fuse(): merges all branches into single conv+BN for fast inference.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: Stride,
                 groups: int, num_branches: int, act: str):
        super().__init__()
        self._has_identity = (out_ch == in_ch and _stride_is_1(stride))
        self._kernel = kernel
        self._stride = _stride2(stride)
        self._groups = groups
        self._in_ch = in_ch
        self._out_ch = out_ch
        self._is_depthwise = (groups == in_ch and in_ch == out_ch)

        if self._has_identity:
            self.identity = nn.BatchNorm(in_ch)

        if self._is_depthwise:
            self.conv_symmetric = [
                DepthwiseConvBN(in_ch, kernel, stride) for _ in range(num_branches)
            ]
            self.conv_small_symmetric = DepthwiseConvBN(in_ch, 1, stride) if kernel > 1 else None
        else:
            self.conv_symmetric = [
                ConvBN(in_ch, out_ch, kernel, stride, groups, act=None)
                for _ in range(num_branches)
            ]
            self.conv_small_symmetric = (
                ConvBN(in_ch, out_ch, 1, stride, groups, act=None) if kernel > 1 else None
            )

        self.lab = LearnableAffineBlock()
        self.act = _ActLAB(act)

        # Fused weight/bias — populated by fuse()
        self._fused_w: mx.array | None = None
        self._fused_b: mx.array | None = None

    def fuse(self) -> None:
        """Merge all branches into a single conv for fast inference."""
        k = self._kernel
        out_ch = self._out_ch
        in_per_group = 1 if self._is_depthwise else self._in_ch

        # Accumulate fused weight (OHWI) and bias
        w = mx.zeros((out_ch, k, k, in_per_group))
        b = mx.zeros((out_ch,))

        # Fuse each conv_symmetric branch
        for conv in self.conv_symmetric:
            cw, cb = _fuse_conv_bn(conv)
            w = w + cw
            b = b + cb

        # Fuse conv_small_symmetric (1×1 → pad to k×k)
        if self.conv_small_symmetric is not None:
            cw, cb = _fuse_conv_bn(self.conv_small_symmetric)
            if k > 1:
                pad = k // 2
                cw = mx.pad(cw, [(0, 0), (pad, pad), (pad, pad), (0, 0)])
            w = w + cw
            b = b + cb

        # Fuse identity BN (→ 1×1 identity conv equiv)
        if self._has_identity:
            iw, ib = _bn_to_conv(self.identity, out_ch, self._is_depthwise)
            if k > 1:
                pad = k // 2
                iw = mx.pad(iw, [(0, 0), (pad, pad), (pad, pad), (0, 0)])
            w = w + iw
            b = b + ib

        self._fused_w = w
        self._fused_b = b

    def __call__(self, x: mx.array) -> mx.array:
        if self._fused_w is not None:
            out = mx.conv2d(x, self._fused_w, stride=self._stride,
                            padding=self._kernel // 2, groups=self._groups)
            out = out + self._fused_b
        else:
            out = None
            if self._has_identity:
                out = self.identity(x)
            if self.conv_small_symmetric is not None:
                r = self.conv_small_symmetric(x)
                out = r if out is None else out + r
            for conv in self.conv_symmetric:
                r = conv(x)
                out = r if out is None else out + r

        out = self.lab(out)
        if not _stride_is_downscale(self._stride):
            out = self.act(out)
        return out


def _fuse_conv_bn(conv_bn) -> tuple[mx.array, mx.array]:
    """Fuse Conv+BN into single weight+bias. Works for ConvBN and DepthwiseConvBN."""
    if isinstance(conv_bn, DepthwiseConvBN):
        w = conv_bn.convolution.weight
        bn = conv_bn.normalization
    else:  # ConvBN
        w = conv_bn.convolution.weight
        bn = conv_bn.normalization

    # BN: y = (x - mean) / sqrt(var + eps) * gamma + beta
    # Fused: y = w_fused * input + b_fused
    gamma = bn.weight
    beta = bn.bias
    mean = bn.running_mean
    var = bn.running_var
    std = mx.sqrt(var + 1e-5)
    scale = gamma / std  # per-channel

    # Weight: scale each output channel
    # w shape: (out, kH, kW, in_per_group) OHWI
    fused_w = w * scale[:, None, None, None]
    fused_b = beta - mean * scale
    return fused_w, fused_b


def _bn_to_conv(bn: nn.BatchNorm, channels: int, depthwise: bool) -> tuple[mx.array, mx.array]:
    """Convert standalone BN (identity branch) to equivalent 1×1 conv weight+bias."""
    gamma = bn.weight
    beta = bn.bias
    mean = bn.running_mean
    var = bn.running_var
    std = mx.sqrt(var + 1e-5)
    scale = gamma / std

    if depthwise:
        # Identity depthwise: each channel maps to itself
        w = mx.zeros((channels, 1, 1, 1))
        for i in range(channels):
            w = w.at[i, 0, 0, 0].add(scale[i])
    else:
        w = mx.diag(scale).reshape(channels, 1, 1, channels)

    b = beta - mean * scale
    return w, b


def fuse_rep_layers(model: nn.Module) -> None:
    """Fuse all LearnableRepLayers in a model for fast inference."""
    count = 0
    for _, module in model.named_modules():
        if isinstance(module, LearnableRepLayer):
            module.fuse()
            count += 1
    if count:
        mx.eval(model.parameters())
        logger.debug("Fused %d RepLayers", count)


class _ActLAB(nn.Module):
    """Activation + LearnableAffineBlock."""

    def __init__(self, act: str):
        super().__init__()
        self._act_fn = _ACT[act]
        self.lab = LearnableAffineBlock()

    def __call__(self, x: mx.array) -> mx.array:
        return self.lab(self._act_fn(x))


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = channels // reduction
        # transformers uses ModuleList [Conv, ReLU, Conv, Hardsigmoid]
        # Placeholders at indices 1,3 so key paths .convolutions.0/.2 match
        self.convolutions = [
            nn.Conv2d(channels, mid, 1),  # 0
            None,                          # 1 (ReLU, no weights)
            nn.Conv2d(mid, channels, 1),  # 2
            None,                          # 3 (Hardsigmoid, no weights)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        s = mx.mean(x, axis=[1, 2], keepdims=True)
        s = nn.relu(self.convolutions[0](s))
        s = _hardsigmoid(self.convolutions[2](s))
        return x * s


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: Stride,
                 use_se: bool, config: LCNetConfig):
        super().__init__()
        self.depthwise_convolution = LearnableRepLayer(
            in_ch, in_ch, kernel, stride,
            groups=in_ch, num_branches=config.conv_symmetric_num, act=config.hidden_act,
        )
        if use_se:
            self.squeeze_excitation_module = SqueezeExcitation(in_ch, config.reduction)
        self._use_se = use_se
        self.pointwise_convolution = LearnableRepLayer(
            in_ch, out_ch, 1, 1,
            groups=1, num_branches=config.conv_symmetric_num, act=config.hidden_act,
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.depthwise_convolution(x)
        if self._use_se:
            x = self.squeeze_excitation_module(x)
        return self.pointwise_convolution(x)


class Block(nn.Module):
    def __init__(self, config: LCNetConfig, stage_idx: int):
        super().__init__()
        blocks = config.block_configs[stage_idx]
        self.layers = []
        for kernel, in_ch, out_ch, stride, use_se in blocks:
            s_in = _make_divisible(in_ch * config.scale, config.divisor)
            s_out = _make_divisible(out_ch * config.scale, config.divisor)
            self.layers.append(DepthwiseSeparableConv(s_in, s_out, kernel, stride, use_se, config))


    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x


class _Encoder(nn.Module):
    """PP-LCNetV3 encoder: stem conv + 5 blocks."""

    def __init__(self, config: LCNetConfig):
        super().__init__()
        stem_ch = _make_divisible(config.stem_channels * config.scale, config.divisor)
        self.convolution = ConvBN(3, stem_ch, 3, stride=config.stem_stride, act=None)
        self.blocks = [Block(config, i) for i in range(5)]

    def __call__(self, x: mx.array) -> list[mx.array]:
        features = []
        x = self.convolution(x)
        features.append(x)
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features


class PPLCNetV3Backbone(nn.Module):
    """PP-LCNetV3 backbone: encoder → multi-scale feature extraction."""

    def __init__(self, config: LCNetConfig):
        super().__init__()
        self._out_indices = config.out_indices
        self.encoder = _Encoder(config)

        # Compute output channels per stage
        stem_ch = _make_divisible(config.stem_channels * config.scale, config.divisor)
        self.out_channels = [stem_ch]
        for block_cfg in config.block_configs:
            self.out_channels.append(
                _make_divisible(block_cfg[-1][2] * config.scale, config.divisor)
            )

    def __call__(self, x: mx.array) -> list[mx.array]:
        features = self.encoder(x)
        return [features[i] for i in self._out_indices]
