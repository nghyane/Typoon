"""PP-OCRv5 mobile text detection — MLX backend.

Architecture: PP-LCNetV3 backbone → RSEFPN neck → DBNet head → sigmoid prob map.
Port of transformers/models/pp_ocrv5_mobile_det/modeling_pp_ocrv5_mobile_det.py.
"""

from __future__ import annotations

import json
import logging

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from safetensors import safe_open

from .._pp_lcnet_v3 import (
    LCNetConfig,
    PPLCNetV3Backbone,
    fuse_rep_layers,
)

logger = logging.getLogger(__name__)


class _SEBlock(nn.Module):
    """Simplified SE for neck (clamp-based activation)."""

    def __init__(self, channels: int, reduction: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels // reduction, 1)
        self.conv2 = nn.Conv2d(channels // reduction, channels, 1)

    def __call__(self, x: mx.array) -> mx.array:
        s = mx.mean(x, axis=[1, 2], keepdims=True)
        s = nn.relu(self.conv1(s))
        s = mx.clip(0.2 * self.conv2(s) + 0.5, 0.0, 1.0)
        return x * s


class _RSELayer(nn.Module):
    """Residual SE layer: conv + SE residual."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, reduction: int):
        super().__init__()
        self.in_conv = nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False)
        self.squeeze_excitation_block = _SEBlock(out_ch, reduction)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.in_conv(x)
        return x + self.squeeze_excitation_block(x)


class _Neck(nn.Module):
    """RSEFPN: multi-scale feature fusion."""

    def __init__(self, in_channels: list[int], out_ch: int, reduction: int):
        super().__init__()
        self.insert_conv = [_RSELayer(c, out_ch, 1, reduction) for c in in_channels]
        self.input_conv = [_RSELayer(out_ch, out_ch // 4, 3, reduction) for _ in in_channels]

    def __call__(self, features: list[mx.array]) -> mx.array:
        # Apply insert_conv
        fused = [conv(f) for conv, f in zip(self.insert_conv, features)]

        # Top-down fusion: p4 += up(p5), p3 += up(p4), p2 += up(p3)
        for i in range(2, -1, -1):
            h, w = fused[i].shape[1], fused[i].shape[2]
            up = _upsample_nearest(fused[i + 1], h, w)
            fused[i] = fused[i] + up

        # Apply input_conv
        out = [conv(f) for conv, f in zip(self.input_conv, fused)]

        # Upsample all to p2 size and concat (reversed: p5, p4, p3, p2)
        h0, w0 = out[0].shape[1], out[0].shape[2]
        processed = []
        for i, feat in enumerate(out):
            if i > 0:
                feat = _upsample_nearest(feat, h0, w0)
            processed.append(feat)
        return mx.concatenate(processed[::-1], axis=3)


class _ConvBN(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int = 1,
                 padding: int = 1, transpose: bool = False):
        super().__init__()
        if transpose:
            self.convolution = nn.ConvTranspose2d(in_ch, out_ch, kernel, stride=stride)
        else:
            self.convolution = nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding)
        self.norm = nn.BatchNorm(out_ch)

    def __call__(self, x: mx.array) -> mx.array:
        return nn.relu(self.norm(self.convolution(x)))


class _Head(nn.Module):
    """DBNet head: conv_down → conv_up → conv_final → sigmoid."""

    def __init__(self, in_ch: int, kernel_list: list[int]):
        super().__init__()
        mid = in_ch // 4
        self.conv_down = _ConvBN(in_ch, mid, kernel_list[0], padding=kernel_list[0] // 2)
        self.conv_up = _ConvBN(mid, mid, kernel_list[1], stride=2, transpose=True)
        self.conv_final = nn.ConvTranspose2d(mid, 1, kernel_list[2], stride=2)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_down(x)
        x = self.conv_up(x)
        x = self.conv_final(x)
        return mx.sigmoid(x)


class _CoreModel(nn.Module):
    """Inner model matching safetensors key prefix `model.`."""

    def __init__(self, config: dict):
        super().__init__()
        bb_cfg = config["backbone_config"]
        lcnet_cfg = LCNetConfig(
            scale=bb_cfg["scale"],
            divisor=bb_cfg.get("divisor", 16),
            out_indices=bb_cfg.get("out_indices", [2, 3, 4, 5]),
            block_configs=bb_cfg.get("block_configs"),
        )
        self.backbone = PPLCNetV3Backbone(lcnet_cfg)

        # 1x1 projection layers (backbone → neck input channels)
        layer_out = config["layer_list_out_channels"]
        bb_out = [self.backbone.out_channels[i] for i in lcnet_cfg.out_indices]
        self.layer = [nn.Conv2d(b, l, 1) for b, l in zip(bb_out, layer_out)]

        self.neck = _Neck(layer_out, config["neck_out_channels"], config.get("reduction", 4))

    def __call__(self, x: mx.array) -> mx.array:
        features = self.backbone(x)
        projected = [conv(f) for conv, f in zip(self.layer, features)]
        return self.neck(projected)


class _DetModel(nn.Module):
    """Top-level: model + head, matching safetensors key hierarchy."""

    def __init__(self, config: dict):
        super().__init__()
        self.model = _CoreModel(config)
        self.head = _Head(config["neck_out_channels"], config["kernel_list"])

    def __call__(self, x: mx.array) -> mx.array:
        return self.head(self.model(x))


def _upsample_nearest(x: mx.array, target_h: int, target_w: int) -> mx.array:
    """Nearest-neighbor upsample NHWC tensor."""
    b, h, w, c = x.shape
    if h == target_h and w == target_w:
        return x
    # Repeat along height
    x = mx.repeat(x, repeats=target_h // h, axis=1)
    # Repeat along width
    x = mx.repeat(x, repeats=target_w // w, axis=2)
    return x


_CONV_TRANSPOSE_KEYS = {"head.conv_up.convolution.weight", "head.conv_final.weight"}


def _load_weights(model: _DetModel, path: str) -> None:
    """Load safetensors with PyTorch→MLX conv weight transposition."""
    weights: list[tuple[str, mx.array]] = []
    with safe_open(path, framework="numpy") as f:
        for key in f.keys():
            if key.endswith("num_batches_tracked"):
                continue
            arr = mx.array(f.get_tensor(key))
            if arr.ndim == 4 and key.endswith(".weight"):
                if key in _CONV_TRANSPOSE_KEYS:
                    # ConvTranspose2d: PyTorch (I,O,H,W) → MLX (O,H,W,I)
                    arr = mx.transpose(arr, axes=[1, 2, 3, 0])
                else:
                    # Conv2d: PyTorch (O,I,H,W) → MLX (O,H,W,I)
                    arr = mx.transpose(arr, axes=[0, 2, 3, 1])
            weights.append((key, arr))
    model.load_weights(weights, strict=False)
    logger.info("PP-OCR det: loaded %d tensors from %s", len(weights), path)


class TextDetector:
    """PP-OCR text detection via MLX."""

    def __init__(self, model_path: str, config_path: str) -> None:
        self._model_path = model_path
        self._config_path = config_path
        self._model: _DetModel | None = None

    def _ensure_loaded(self) -> _DetModel:
        if self._model is None:
            with open(self._config_path) as f:
                config = json.load(f)
            model = _DetModel(config)
            _load_weights(model, self._model_path)
            model.eval()
            fuse_rep_layers(model)
            mx.eval(model.parameters())
            self._compiled = mx.compile(model)
            self._model = model
            logger.info("PP-OCR det ready (MLX, fused+compiled)")
        return self._model

    _MEAN = mx.array([0.485, 0.456, 0.406])
    _STD = mx.array([0.229, 0.224, 0.225])

    def detect(self, image: np.ndarray) -> np.ndarray:
        """Run detection. image: RGB uint8 HWC, preprocessed/padded.

        Returns: probability map float32 HW, same size as input.
        """
        self._ensure_loaded()

        # Normalize on Metal — single copy numpy→MLX, no CPU preprocessing
        x = mx.array(image[np.newaxis]).astype(mx.float32) * (1.0 / 255.0)
        x = (x - self._MEAN) / self._STD
        prob = self._compiled(x)

        # np.array() implicitly syncs — no need for mx.eval()
        return np.array(prob[0, :, :, 0])
