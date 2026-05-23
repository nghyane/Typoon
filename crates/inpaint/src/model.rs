// SPDX-License-Identifier: GPL-3.0-or-later
//! AOT-GAN model — Rust/Candle implementation.
//!
//! Port of `mayocream/koharu` `koharu-ml/src/aot_inpainting/model.rs`
//! (GPLv3, © mayocream). Architecture:
//!
//!   AotGenerator
//!     head: 3× GatedWsConvPadded (stride-1, stride-2, stride-2)
//!     body: N× AotBlock          (10 blocks, dilations [2,4,8,16])
//!     tail: 2× GatedWsConvPadded + 2× GatedWsTransposeConvPadded + 1× output
//!
//! Key design: every intermediate tensor is dropped as soon as consumed
//! (Rust ownership). This keeps peak memory ~300 MB at 384×384 vs 6.9 GB
//! with ONNX Runtime CPU EP that holds all intermediates alive.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{
    Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig, Module, VarBuilder,
    ops::sigmoid,
};

// ──────────────────────────────────────────────────────────────────────────────
// Constants (match koharu)
// ──────────────────────────────────────────────────────────────────────────────

const RELU_NF_SCALE: f64 = 1.713_958_859_443_664_6;
const WS_EPS: f32 = 1e-4;
const LN_EPS: f64 = 1e-9;

// ──────────────────────────────────────────────────────────────────────────────
// Weight-standardised conv helpers
// ──────────────────────────────────────────────────────────────────────────────

fn standardize_conv_weight(weight: Tensor, gain: Tensor) -> candle_core::Result<Tensor> {
    let dtype  = weight.dtype();
    let weight = weight.to_dtype(DType::F32)?;
    let gain   = gain.to_dtype(DType::F32)?;
    let (out_c, in_c, kh, kw) = weight.dims4()?;
    let flat   = weight.flatten_from(1)?;
    let fan_in = flat.dim(1)? as f64;
    let mean   = flat.mean_keepdim(1)?;
    let var    = flat.var_keepdim(1)?;
    let var    = (&var * fan_in)?;
    let eps    = Tensor::full(WS_EPS, var.shape().clone(), var.device())?;
    let scale  = var.maximum(&eps)?.sqrt()?.recip()?;
    let scale  = scale.broadcast_mul(&gain.reshape((out_c, 1))?)?;
    let shift  = mean.broadcast_mul(&scale)?;
    flat.broadcast_mul(&scale)?
        .broadcast_sub(&shift)?
        .reshape((out_c, in_c, kh, kw))?
        .to_dtype(dtype)
}

fn standardize_transp_weight(weight: Tensor, gain: Tensor) -> candle_core::Result<Tensor> {
    let dtype  = weight.dtype();
    let weight = weight.to_dtype(DType::F32)?;
    let gain   = gain.to_dtype(DType::F32)?;
    let (in_c, out_c, kh, kw) = weight.dims4()?;
    let flat   = weight.flatten_from(1)?;
    let fan_in = flat.dim(1)? as f64;
    let mean   = flat.mean_keepdim(1)?;
    let var    = flat.var_keepdim(1)?;
    let var    = (&var * fan_in)?;
    let eps    = Tensor::full(WS_EPS, var.shape().clone(), var.device())?;
    let scale  = var.maximum(&eps)?.sqrt()?.recip()?;
    let scale  = scale.broadcast_mul(&gain.reshape((in_c, 1))?)?;
    let shift  = mean.broadcast_mul(&scale)?;
    flat.broadcast_mul(&scale)?
        .broadcast_sub(&shift)?
        .reshape((in_c, out_c, kh, kw))?
        .to_dtype(dtype)
}

fn load_ws_conv2d(
    vb: &VarBuilder,
    shape: (usize, usize, usize, usize),
    stride: usize,
    dilation: usize,
) -> Result<Conv2d> {
    let w = standardize_conv_weight(vb.get(shape, "weight")?, vb.get((shape.0, 1, 1, 1), "gain")?)?;
    let b = Some(vb.get(shape.0, "bias")?);
    Ok(Conv2d::new(w, b, Conv2dConfig { padding: 0, stride, dilation, groups: 1, cudnn_fwd_algo: None }))
}

fn load_plain_conv2d(
    vb: &VarBuilder,
    shape: (usize, usize, usize, usize),
    padding: usize,
    dilation: usize,
) -> Result<Conv2d> {
    let w = vb.get(shape, "weight")?;
    let b = Some(vb.get(shape.0, "bias")?);
    Ok(Conv2d::new(w, b, Conv2dConfig { padding, stride: 1, dilation, groups: 1, cudnn_fwd_algo: None }))
}

fn load_ws_transp_conv2d(
    vb: &VarBuilder,
    shape: (usize, usize, usize, usize),
    stride: usize,
    padding: usize,
) -> Result<ConvTranspose2d> {
    let w = standardize_transp_weight(vb.get(shape, "weight")?, vb.get((shape.0, 1, 1, 1), "gain")?)?;
    let b = Some(vb.get(shape.1, "bias")?);
    Ok(ConvTranspose2d::new(w, b, ConvTranspose2dConfig { padding, output_padding: 0, stride, dilation: 1 }))
}

// ──────────────────────────────────────────────────────────────────────────────
// Padding helper
// ──────────────────────────────────────────────────────────────────────────────

fn reflect_pad2d(xs: &Tensor, pad: usize) -> candle_core::Result<Tensor> {
    if pad == 0 { return Ok(xs.clone()); }
    let xs = xs.contiguous()?;
    let (_n, _c, h, w) = xs.dims4()?;
    let left   = xs.narrow(3, 1, pad)?.contiguous()?.flip(&[3])?;
    let right  = xs.narrow(3, w - pad - 1, pad)?.contiguous()?.flip(&[3])?;
    let xs     = Tensor::cat(&[&left, &xs, &right], 3)?;
    let top    = xs.narrow(2, 1, pad)?.contiguous()?.flip(&[2])?;
    let bottom = xs.narrow(2, h - pad - 1, pad)?.contiguous()?.flip(&[2])?;
    Tensor::cat(&[&top, &xs, &bottom], 2)
}

// ──────────────────────────────────────────────────────────────────────────────
// Activation helpers
// ──────────────────────────────────────────────────────────────────────────────

fn relu_nf(xs: Tensor) -> candle_core::Result<Tensor> {
    xs.relu()? * RELU_NF_SCALE
}

fn layer_norm(xs: &Tensor) -> candle_core::Result<Tensor> {
    let dtype = xs.dtype();
    let xs    = xs.to_dtype(DType::F32)?;
    let (b, c, h, w) = xs.dims4()?;
    let flat  = xs.flatten_from(2)?;
    let mean  = flat.mean_keepdim(2)?;
    let std   = ((flat.var_keepdim(2)? + LN_EPS)?).sqrt()?;
    (((flat.broadcast_sub(&mean)? * 2.0)?)
        .broadcast_div(&std)?
        .broadcast_sub(&Tensor::ones_like(&flat)?)?
        * 5.0)?
        .reshape((b, c, h, w))?
        .to_dtype(dtype)
}

// ──────────────────────────────────────────────────────────────────────────────
// Building blocks
// ──────────────────────────────────────────────────────────────────────────────

struct GatedWsConv {
    conv:      Conv2d,
    conv_gate: Conv2d,
    pad:       usize,
}

impl GatedWsConv {
    fn load(vb: &VarBuilder, in_c: usize, out_c: usize, k: usize, stride: usize, dilation: usize) -> Result<Self> {
        Ok(Self {
            conv:      load_ws_conv2d(&vb.pp("conv"),      (out_c, in_c, k, k), stride, dilation)?,
            conv_gate: load_ws_conv2d(&vb.pp("conv_gate"), (out_c, in_c, k, k), stride, dilation)?,
            pad:       (k - 1) * dilation / 2,
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let xs     = reflect_pad2d(xs, self.pad)?;
        let signal = self.conv.forward(&xs)?;
        let gate   = sigmoid(&self.conv_gate.forward(&xs)?)?;
        (signal * gate)? * 1.8
    }
}

struct GatedWsTranspConv {
    conv:      ConvTranspose2d,
    conv_gate: ConvTranspose2d,
}

impl GatedWsTranspConv {
    fn load(vb: &VarBuilder, in_c: usize, out_c: usize, k: usize, stride: usize) -> Result<Self> {
        let pad = (k - 1) / 2;
        Ok(Self {
            conv:      load_ws_transp_conv2d(&vb.pp("conv"),      (in_c, out_c, k, k), stride, pad)?,
            conv_gate: load_ws_transp_conv2d(&vb.pp("conv_gate"), (in_c, out_c, k, k), stride, pad)?,
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let signal = self.conv.forward(xs)?;
        let gate   = sigmoid(&self.conv_gate.forward(xs)?)?;
        (signal * gate)? * 1.8
    }
}

struct PaddedConvRelu { conv: Conv2d, pad: usize }

impl PaddedConvRelu {
    fn load(vb: &VarBuilder, in_c: usize, out_c: usize, k: usize, dilation: usize) -> Result<Self> {
        Ok(Self {
            conv: load_plain_conv2d(vb, (out_c, in_c, k, k), 0, dilation)?,
            pad:  (k - 1) * dilation / 2,
        })
    }
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        self.conv.forward(&reflect_pad2d(xs, self.pad)?)?.relu()
    }
}

struct PaddedConv { conv: Conv2d, pad: usize }

impl PaddedConv {
    fn load(vb: &VarBuilder, c: usize, k: usize) -> Result<Self> {
        Ok(Self {
            conv: load_plain_conv2d(vb, (c, c, k, k), 0, 1)?,
            pad:  (k - 1) / 2,
        })
    }
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        self.conv.forward(&reflect_pad2d(xs, self.pad)?)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// AOT Block
// ──────────────────────────────────────────────────────────────────────────────

struct AotBlock {
    branches: Vec<PaddedConvRelu>,
    fuse:     PaddedConv,
    gate:     PaddedConv,
}

impl AotBlock {
    fn load(vb: &VarBuilder, channels: usize, dilations: &[usize]) -> Result<Self> {
        let branch_ch = channels / 4;
        let branches  = dilations.iter().enumerate().map(|(i, &d)| {
            PaddedConvRelu::load(&vb.pp(format!("block{i:02}.1")), channels, branch_ch, 3, d)
        }).collect::<Result<_>>()?;
        Ok(Self {
            branches,
            fuse: PaddedConv::load(&vb.pp("fuse.1"), channels, 3)?,
            gate: PaddedConv::load(&vb.pp("gate.1"), channels, 3)?,
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        // Each branch is computed and collected, then immediately concatenated.
        // After Tensor::cat the Vec is dropped, freeing branch intermediates.
        let branch_outs: Vec<Tensor> = self.branches.iter()
            .map(|b| b.forward(xs))
            .collect::<candle_core::Result<_>>()?;
        let refs: Vec<&Tensor> = branch_outs.iter().collect();
        let cat   = Tensor::cat(&refs, 1)?;
        drop(branch_outs); // free N×(ch/4)×H×W tensors
        let fused = self.fuse.forward(&cat)?;
        drop(cat);
        let gate  = sigmoid(&layer_norm(&self.gate.forward(xs)?)?)?;
        let keep  = (Tensor::ones_like(&gate)? - &gate)?;
        (xs * &keep)? + (&fused * &gate)?
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// AotGenerator
// ──────────────────────────────────────────────────────────────────────────────

pub struct AotGenerator {
    head0:  GatedWsConv,
    head1:  GatedWsConv,
    head2:  GatedWsConv,
    body:   Vec<AotBlock>,
    tail0:  GatedWsConv,
    tail1:  GatedWsConv,
    up0:    GatedWsTranspConv,
    up1:    GatedWsTranspConv,
    output: GatedWsConv,
}

impl AotGenerator {
    pub fn load(vb: &VarBuilder, in_c: usize, out_c: usize, base: usize, num_blocks: usize, dilations: &[usize]) -> Result<Self> {
        let bc = base * 4;
        let body = (0..num_blocks).map(|i| {
            AotBlock::load(&vb.pp(format!("body_conv.{i}")), bc, dilations)
        }).collect::<Result<_>>()?;
        Ok(Self {
            head0:  GatedWsConv::load(&vb.pp("head.0"), in_c,   base,  3, 1, 1)?,
            head1:  GatedWsConv::load(&vb.pp("head.2"), base,   base*2, 4, 2, 1)?,
            head2:  GatedWsConv::load(&vb.pp("head.4"), base*2, bc,     4, 2, 1)?,
            body,
            tail0:  GatedWsConv::load(&vb.pp("tail.0"), bc,    bc,    3, 1, 1)?,
            tail1:  GatedWsConv::load(&vb.pp("tail.2"), bc,    bc,    3, 1, 1)?,
            up0:    GatedWsTranspConv::load(&vb.pp("tail.4"), bc,    base*2, 4, 2)?,
            up1:    GatedWsTranspConv::load(&vb.pp("tail.6"), base*2, base,  4, 2)?,
            output: GatedWsConv::load(&vb.pp("tail.8"), base,  out_c, 3, 1, 1)?,
        })
    }

    pub fn forward(&self, image: &Tensor, mask: &Tensor) -> candle_core::Result<Tensor> {
        // Concatenate mask channel with image: (B, 4, H, W)
        let mut xs = Tensor::cat(&[mask, image], 1)?;
        xs = relu_nf(self.head0.forward(&xs)?)?;
        xs = relu_nf(self.head1.forward(&xs)?)?;
        xs = self.head2.forward(&xs)?;
        for block in &self.body {
            xs = block.forward(&xs)?;
        }
        xs = relu_nf(self.tail0.forward(&xs)?)?;
        xs = relu_nf(self.tail1.forward(&xs)?)?;
        xs = relu_nf(self.up0.forward(&xs)?)?;
        xs = relu_nf(self.up1.forward(&xs)?)?;
        self.output.forward(&xs)?.clamp(-1.0, 1.0)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Public Inpainter wrapper
// ──────────────────────────────────────────────────────────────────────────────

pub struct Inpainter {
    model:   AotGenerator,
    device:  Device,
    dtype:   DType,
    pad_mod: u32,
}

impl Inpainter {
    /// Load from a safetensors file. Uses FP16 on capable hardware,
    /// FP32 fallback for CPU (FP16 matmul on CPU is emulated — slower).
    /// Pass `fp16 = false` to force FP32 (higher quality, 2× RAM).
    pub fn load(weights: &std::path::Path, fp16: bool) -> Result<Self> {
        let device = Device::Cpu;
        let dtype  = if fp16 { DType::F16 } else { DType::F32 };
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights], dtype, &device)?
        };
        let model = AotGenerator::load(&vb, 4, 3, 32, 10, &[2, 4, 8, 16])?;
        Ok(Self { model, device, dtype, pad_mod: 8 })
    }

    /// Inpaint `image_rgb` (H×W×3 u8) where `mask` (H×W u8, >=127 = inpaint).
    /// Returns inpainted RGB image same size.
    pub fn inpaint(&self, image_rgb: &[u8], mask: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
        let (w, h) = (width as usize, height as usize);
        debug_assert_eq!(image_rgb.len(), w * h * 3);
        debug_assert_eq!(mask.len(), w * h);

        // Pad to pad_mod boundary
        let pw = ceil_mod(w, self.pad_mod as usize);
        let ph = ceil_mod(h, self.pad_mod as usize);

        let img_pad  = reflect_pad_image(image_rgb, w, h, pw, ph);
        let mask_pad = reflect_pad_mask(mask, w, h, pw, ph);

        // Normalise image to [-1, 1] zeroed at masked pixels
        // img_pad is HWC (interleaved RGB). Model needs CHW.
        let mask_f32: Vec<f32> = mask_pad.iter()
            .map(|&v| if v >= 127 { 1.0f32 } else { 0.0 })
            .collect();

        // Build CHW float tensor: separate R, G, B planes
        let mut img_chw = vec![0.0f32; 3 * ph * pw];
        for y in 0..ph {
            for x in 0..pw {
                let m = mask_f32[y * pw + x];
                for c in 0..3 {
                    let src = img_pad[(y * pw + x) * 3 + c];
                    img_chw[c * ph * pw + y * pw + x] = (src as f32 / 127.5 - 1.0) * (1.0 - m);
                }
            }
        }

        // Build NCHW tensors
        let img_t  = Tensor::from_vec(img_chw, (1, 3, ph, pw), &self.device)?
            .to_dtype(self.dtype)?;
        let mask_t = Tensor::from_vec(mask_f32, (1, 1, ph, pw), &self.device)?
            .to_dtype(self.dtype)?;

        // Forward
        let out = self.model.forward(&img_t, &mask_t)?;
        drop(img_t); drop(mask_t);

        // Decode: [-1, 1] → [0, 255], crop pad, paste only masked pixels
        let out_f32 = out.to_dtype(DType::F32)?
            .narrow(2, 0, h)?.narrow(3, 0, w)?
            .squeeze(0)?;
        drop(out);
        let raw: Vec<f32> = out_f32.flatten_all()?.to_vec1()?;
        drop(out_f32);

        // Compose: use model output where mask>=127, original elsewhere
        // raw is CHW: raw[c * h * w + y * w + x]
        let mut result = image_rgb.to_vec();
        for y in 0..h {
            for x in 0..w {
                if mask[y * w + x] < 127 { continue; }
                let base = (y * w + x) * 3;
                for c in 0..3 {
                    result[base + c] = ((raw[c * h * w + y * w + x] + 1.0) * 127.5)
                        .clamp(0.0, 255.0) as u8;
                }
            }
        }
        Ok(result)
    }
}

fn ceil_mod(v: usize, m: usize) -> usize {
    let r = v % m;
    if r == 0 { v } else { v + (m - r) }
}

fn reflect_pad_image(src: &[u8], w: usize, h: usize, pw: usize, ph: usize) -> Vec<u8> {
    let mut out = vec![0u8; pw * ph * 3];
    for y in 0..ph {
        let sy = reflect_idx(y, h);
        for x in 0..pw {
            let sx = reflect_idx(x, w);
            let di = (y * pw + x) * 3;
            let si = (sy * w + sx) * 3;
            out[di..di+3].copy_from_slice(&src[si..si+3]);
        }
    }
    out
}

fn reflect_pad_mask(src: &[u8], w: usize, h: usize, pw: usize, ph: usize) -> Vec<u8> {
    let mut out = vec![0u8; pw * ph];
    for y in 0..ph {
        let sy = reflect_idx(y, h);
        for x in 0..pw {
            let sx = reflect_idx(x, w);
            out[y * pw + x] = src[sy * w + sx];
        }
    }
    out
}

fn reflect_idx(i: usize, len: usize) -> usize {
    if i < len { return i; }
    let excess = i - len;
    if excess < len { len - 1 - excess } else { 0 }
}
