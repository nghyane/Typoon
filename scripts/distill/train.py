"""MI-GAN distillation training from LaMa teacher.

Trains a lightweight MI-GAN student generator using:
- Adversarial loss (non-saturating logistic GAN)
- Knowledge distillation: L1 between student and LaMa teacher in masked region
- R1 gradient penalty on discriminator
- EMA on generator weights
- Mixed precision (torch.amp) for speed

Usage:
    python train.py --data_dir /path/to/manga_images
    python train.py --data_dir /path/to/images --teacher_cache /path/to/cache
"""

import argparse
import copy
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from model import MIGANGenerator, PatchDiscriminator
from dataset import MangaInpaintDataset, TeacherCacheDataset


# ---------------------------------------------------------------------------
# LaMa teacher wrapper (ONNX)
# ---------------------------------------------------------------------------

class LamaTeacher:
    """Runs LaMa inpainting via ONNX Runtime.

    LaMa input:  image [B,3,H,W] float32 [0,1], mask [B,1,H,W] float32 {0,1}
    LaMa mask:   1 = inpaint, 0 = keep (OPPOSITE of MI-GAN convention)
    LaMa output: [B,3,H,W] float32 [0,1]
    """

    def __init__(self, onnx_path: str, device_id: int = 0):
        available = ort.get_available_providers()
        providers = []
        if "CUDAExecutionProvider" in available:
            providers.append(("CUDAExecutionProvider", {"device_id": device_id}))
        providers.append("CPUExecutionProvider")
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        print(f"[LamaTeacher] Loaded from {onnx_path}")
        print(f"  inputs:  {self.input_names}")
        print(f"  outputs: {self.output_names}")

    @torch.no_grad()
    def __call__(self, image_01: torch.Tensor, mask_migan: torch.Tensor) -> torch.Tensor:
        """Run LaMa inference.

        Args:
            image_01: [B,3,H,W] float32 in [0,1] — the ORIGINAL clean image
            mask_migan: [B,1,H,W] float32, MI-GAN convention (1=known, 0=masked)

        Returns:
            teacher output [B,3,H,W] float32 in [0,1]
        """
        # Flip mask: MI-GAN 1=known,0=masked → LaMa 1=inpaint,0=keep
        lama_mask = 1.0 - mask_migan

        # Prepare masked image for LaMa: zero out the inpaint region
        masked_image = image_01 * mask_migan  # keep known, zero masked

        img_np = masked_image.cpu().numpy()
        mask_np = lama_mask.cpu().numpy()

        feed = {self.input_names[0]: img_np, self.input_names[1]: mask_np}
        outputs = self.session.run(self.output_names, feed)
        result = torch.from_numpy(outputs[0]).to(image_01.device)
        return result.clamp(0, 1)


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = copy.deepcopy(model)
        self.model.eval()
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module):
        for ema_p, src_p in zip(self.model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(src_p.data, alpha=1 - self.decay)
        for ema_b, src_b in zip(self.model.buffers(), model.buffers()):
            ema_b.data.copy_(src_b.data)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def g_nonsaturating_loss(d_fake: torch.Tensor) -> torch.Tensor:
    """Non-saturating logistic GAN loss for generator."""
    return F.softplus(-d_fake).mean()


def r1_penalty(d_real: torch.Tensor, real_images: torch.Tensor) -> torch.Tensor:
    """R1 gradient penalty on real images."""
    grad, = torch.autograd.grad(
        outputs=d_real.sum(), inputs=real_images,
        create_graph=True, retain_graph=True,
    )
    return grad.pow(2).reshape(grad.size(0), -1).sum(1).mean()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(args):
    device = get_device()
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    use_amp = args.use_amp and device.type == "cuda"  # AMP only on CUDA
    print(f"Device: {device} (AMP: {use_amp})")

    # ---- Data ----
    use_cache = args.teacher_cache is not None
    if use_cache:
        dataset = TeacherCacheDataset(args.teacher_cache, augment=True, preload=args.preload_cache)
        print(f"Using teacher cache: {len(dataset)} samples")
    else:
        dataset = MangaInpaintDataset(args.data_dir, mask_dir=args.mask_dir,
                                       image_size=512, augment=True)
        print(f"Training images: {len(dataset)}")

    pin_mem = device.type == "cuda"
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=pin_mem,
                        drop_last=True, persistent_workers=args.num_workers > 0)

    # ---- Models ----
    gen = MIGANGenerator(in_ch=4, base_ch=48).to(device)
    disc = PatchDiscriminator(in_ch=4, base_ch=64).to(device)
    ema = EMA(gen, decay=0.999)

    from model import count_parameters
    print(f"Generator params:     {count_parameters(gen):,}")
    print(f"Discriminator params: {count_parameters(disc):,}")

    # ---- Teacher ----
    teacher = None
    if not use_cache:
        teacher = LamaTeacher(args.lama_model, device_id=0)

    # ---- Optimizers ----
    opt_g = torch.optim.Adam(gen.parameters(), lr=args.lr, betas=(0.0, 0.99))
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr, betas=(0.0, 0.99))

    # ---- AMP ----
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ---- Logging ----
    log_dir = os.path.join(args.output_dir, "logs")
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # ---- Resume ----
    start_kimg = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        gen.load_state_dict(ckpt["gen"])
        disc.load_state_dict(ckpt["disc"])
        opt_g.load_state_dict(ckpt["opt_g"])
        opt_d.load_state_dict(ckpt["opt_d"])
        ema.load_state_dict(ckpt["ema"])
        start_kimg = ckpt.get("kimg", 0)
        print(f"Resumed from {args.resume} at {start_kimg} kimg")

    # ---- Training ----
    total_images = args.total_kimg * 1000
    images_seen = start_kimg * 1000
    pbar = tqdm(total=total_images, initial=images_seen, unit="img",
                desc="Training", smoothing=0.1)

    data_iter = iter(loader)
    tick_start = time.time()

    while images_seen < total_images:
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        if use_cache:
            real, mask, teacher_out = [b.to(device) for b in batch]
            # teacher_out is already in [-1,1]
        else:
            real, mask = batch[0].to(device), batch[1].to(device)
            teacher_out = None

        bs = real.size(0)

        # Prepare student input: [mask-0.5, masked_rgb]
        # real is [-1,1], mask is [1=known, 0=masked]
        masked_rgb = real * mask  # zero out masked region
        mask_ch = mask - 0.5     # shift to [-0.5, 0.5]
        gen_input = torch.cat([mask_ch, masked_rgb], dim=1)  # [B, 4, H, W]

        # ==================================================================
        # Discriminator step
        # ==================================================================
        gen.requires_grad_(False)
        disc.requires_grad_(True)

        with torch.autocast(device_type=amp_device, enabled=use_amp):
            fake = gen(gen_input)
            # Composite: keep known pixels from real, fill masked with generated
            composite = fake * (1 - mask) + real * mask

            # Discriminator inputs: [rgb, mask]
            d_real_out = disc(torch.cat([real, mask], dim=1))
            d_fake_out = disc(torch.cat([composite.detach(), mask], dim=1))

            loss_d_real = F.softplus(-d_real_out).mean()
            loss_d_fake = F.softplus(d_fake_out).mean()
            loss_d = loss_d_real + loss_d_fake

        # R1 penalty (every 16 steps, lazy regularization)
        r1_interval = 16
        loss_r1 = torch.tensor(0.0, device=device)
        if images_seen % (r1_interval * bs) < bs:
            real_r1 = real.detach().requires_grad_(True)
            d_real_r1 = disc(torch.cat([real_r1, mask], dim=1))
            loss_r1 = r1_penalty(d_real_r1, real_r1) * (args.r1_gamma * 0.5 * r1_interval)
            loss_d = loss_d + loss_r1

        opt_d.zero_grad(set_to_none=True)
        scaler.scale(loss_d).backward()
        scaler.step(opt_d)

        # ==================================================================
        # Generator step
        # ==================================================================
        gen.requires_grad_(True)
        disc.requires_grad_(False)

        with torch.autocast(device_type=amp_device, enabled=use_amp):
            fake = gen(gen_input)
            composite = fake * (1 - mask) + real * mask

            # Adversarial loss
            d_fake_for_g = disc(torch.cat([composite, mask], dim=1))
            loss_g_adv = g_nonsaturating_loss(d_fake_for_g)

            # Knowledge distillation loss (L1 in masked region)
            if teacher_out is None:
                # Run LaMa teacher online
                # Convert real from [-1,1] to [0,1] for LaMa
                real_01 = (real + 1) * 0.5
                with torch.no_grad():
                    t_out_01 = teacher(real_01, mask)
                # Convert teacher output to [-1,1]
                teacher_out_curr = t_out_01 * 2 - 1
            else:
                teacher_out_curr = teacher_out

            # L1 in masked region only
            mask_inv = 1 - mask  # 1 where masked
            n_masked = mask_inv.sum().clamp(min=1)
            loss_kd = (composite - teacher_out_curr).abs() * mask_inv
            loss_kd = loss_kd.sum() / n_masked * args.kd_weight

            loss_g = loss_g_adv + loss_kd

        opt_g.zero_grad(set_to_none=True)
        scaler.scale(loss_g).backward()
        scaler.step(opt_g)

        scaler.update()
        ema.update(gen)

        images_seen += bs
        pbar.update(bs)
        kimg = images_seen / 1000

        # ---- Logging ----
        if images_seen % (args.log_every * bs) < bs:
            elapsed = time.time() - tick_start
            imgs_per_sec = args.log_every * bs / max(elapsed, 1e-6)
            tick_start = time.time()

            writer.add_scalar("loss/d_total", loss_d.item(), images_seen)
            writer.add_scalar("loss/d_real", loss_d_real.item(), images_seen)
            writer.add_scalar("loss/d_fake", loss_d_fake.item(), images_seen)
            writer.add_scalar("loss/r1", loss_r1.item(), images_seen)
            writer.add_scalar("loss/g_adv", loss_g_adv.item(), images_seen)
            writer.add_scalar("loss/g_kd", loss_kd.item(), images_seen)
            writer.add_scalar("loss/g_total", loss_g.item(), images_seen)
            writer.add_scalar("speed/imgs_per_sec", imgs_per_sec, images_seen)

            pbar.set_postfix({
                "kimg": f"{kimg:.1f}",
                "G": f"{loss_g.item():.3f}",
                "D": f"{loss_d.item():.3f}",
                "KD": f"{loss_kd.item():.3f}",
                "ips": f"{imgs_per_sec:.0f}",
            })

        # ---- Visualize ----
        if images_seen % (args.vis_every * bs) < bs:
            with torch.no_grad():
                ema.model.eval()
                vis_fake = ema.model(gen_input[:4])
                vis_comp = vis_fake * (1 - mask[:4]) + real[:4] * mask[:4]
                ema.model.train()

            # Denorm from [-1,1] to [0,1]
            grid = make_grid(torch.cat([
                (real[:4] + 1) / 2,
                (masked_rgb[:4] + 1) / 2,
                (vis_comp + 1) / 2,
            ], dim=0), nrow=4, normalize=False)
            writer.add_image("samples/real_masked_output", grid, images_seen)

        # ---- Save checkpoint ----
        if images_seen % (args.save_every * bs) < bs:
            ckpt_path = os.path.join(ckpt_dir, f"ckpt_{kimg:.0f}kimg.pt")
            torch.save({
                "gen": gen.state_dict(),
                "disc": disc.state_dict(),
                "opt_g": opt_g.state_dict(),
                "opt_d": opt_d.state_dict(),
                "ema": ema.state_dict(),
                "kimg": kimg,
                "args": vars(args),
            }, ckpt_path)
            print(f"\n  Saved checkpoint: {ckpt_path}")

            # Also save a "best" copy (latest = best for now)
            best_path = os.path.join(ckpt_dir, "best.pt")
            torch.save({
                "gen": gen.state_dict(),
                "ema": ema.state_dict(),
                "kimg": kimg,
            }, best_path)

    pbar.close()
    writer.close()
    print("Training complete!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="MI-GAN distillation from LaMa teacher")

    # Data
    p.add_argument("--data_dir", type=str, required=True,
                   help="Root directory of training images")
    p.add_argument("--mask_dir", type=str, default=None,
                   help="Optional directory of pre-computed masks")
    p.add_argument("--teacher_cache", type=str, default=None,
                   help="Directory of pre-generated teacher cache (.npz)")

    # Model
    p.add_argument("--lama_model", type=str,
                   default="../../models/lama_fp32.onnx",
                   help="Path to LaMa ONNX model")

    # Training
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--total_kimg", type=int, default=10000,
                   help="Total training images in thousands")
    p.add_argument("--kd_weight", type=float, default=2.0,
                   help="Knowledge distillation loss weight")
    p.add_argument("--r1_gamma", type=float, default=10.0,
                   help="R1 gradient penalty weight")
    p.add_argument("--use_amp", action="store_true", default=True,
                   help="Use mixed precision training")
    p.add_argument("--no_amp", action="store_true",
                   help="Disable mixed precision")

    # Logging
    p.add_argument("--output_dir", type=str, default="runs/migan_distill")
    p.add_argument("--log_every", type=int, default=100,
                   help="Log metrics every N iterations")
    p.add_argument("--vis_every", type=int, default=500,
                   help="Visualize samples every N iterations")
    p.add_argument("--save_every", type=int, default=5000,
                   help="Save checkpoint every N iterations")

    # System
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--preload_cache", action="store_true",
                   help="Preload teacher cache into RAM")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from")

    args = p.parse_args()
    if args.no_amp:
        args.use_amp = False

    train(args)


if __name__ == "__main__":
    main()
