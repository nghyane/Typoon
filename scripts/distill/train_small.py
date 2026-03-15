"""Self-supervised manga inpainting training.

No teacher model needed — trains directly on clean manga images + random masks.
Ground truth = original image. Model learns to reconstruct masked regions.

Usage:
    # Full pipeline: overfit test → train → export
    python train_small.py --data_dir /path/to/manga_images

    # Skip overfit test, just train
    python train_small.py --data_dir /path/to/manga_images --skip_overfit

    # Resume from checkpoint
    python train_small.py --data_dir /path/to/manga_images --resume checkpoints/latest.pt
"""

import argparse
import copy
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

from small_model import SmallInpaintModel, count_parameters
from dataset import MangaInpaintDataset, generate_freeform_mask, find_images


# ---------------------------------------------------------------------------
# Discriminator (lightweight PatchGAN)
# ---------------------------------------------------------------------------

class PatchDiscriminator(nn.Module):
    """Simple PatchGAN discriminator with spectral norm."""

    def __init__(self, in_ch: int = 4, base_ch: int = 64, n_layers: int = 3):
        super().__init__()
        layers = [
            nn.utils.spectral_norm(nn.Conv2d(in_ch, base_ch, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        ch = base_ch
        for i in range(1, n_layers):
            prev_ch = ch
            ch = min(ch * 2, 512)
            layers += [
                nn.utils.spectral_norm(nn.Conv2d(prev_ch, ch, 4, 2, 1, bias=False)),
                nn.InstanceNorm2d(ch),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        prev_ch = ch
        ch = min(ch * 2, 512)
        layers += [
            nn.utils.spectral_norm(nn.Conv2d(prev_ch, ch, 4, 1, 1, bias=False)),
            nn.InstanceNorm2d(ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, 1, 4, 1, 1),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class EMA:
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

    def load_state_dict(self, sd):
        self.model.load_state_dict(sd)


# ---------------------------------------------------------------------------
# Overfit test
# ---------------------------------------------------------------------------

def overfit_test(data_dir: str, device: torch.device, base_ch: int = 48) -> bool:
    """Quick overfit test: can the model memorize a few manga images?"""
    print("\n" + "=" * 60)
    print("OVERFIT TEST — Validating model capacity")
    print("=" * 60)

    paths = find_images(data_dir)
    random.seed(42)
    selected = random.sample(paths, min(8, len(paths)))

    images, masks = [], []
    for p in selected:
        img = Image.open(p).convert("RGB")
        w, h = img.size
        s = 512
        if min(w, h) < s:
            sc = s / min(w, h)
            img = img.resize((int(w * sc) + 1, int(h * sc) + 1), Image.LANCZOS)
            w, h = img.size
        x, y = (w - s) // 2, (h - s) // 2
        img = img.crop((x, y, x + s, y + s))
        arr = np.array(img, dtype=np.float32) / 255.0
        images.append(torch.from_numpy(arr.transpose(2, 0, 1)))
        m = generate_freeform_mask(s, s)
        masks.append(torch.from_numpy(1.0 - m).unsqueeze(0))

    images = torch.stack(images).to(device)
    masks = torch.stack(masks).to(device)
    print(f"  Images: {len(selected)}, Masked: {masks.mean().item()*100:.0f}%")

    model = SmallInpaintModel(base_ch=base_ch).to(device)
    print(f"  Params: {count_parameters(model):,}")
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    model.train()

    t0 = time.time()
    for step in range(1, 501):
        opt.zero_grad()
        out = model(images, masks)
        loss = ((out - images).abs() * masks).sum() / masks.sum().clamp(min=1)
        loss.backward()
        opt.step()
        if step % 100 == 0:
            with torch.no_grad():
                mse = ((out - images) ** 2 * masks).sum() / masks.sum() / 3
                p = -10 * torch.log10(mse.clamp(min=1e-10)).item()
            print(f"  Step {step:3d} | Loss {loss.item():.5f} | PSNR {p:.1f} dB | {time.time()-t0:.1f}s")

    model.eval()
    with torch.no_grad():
        final = model(images, masks)
        mse = ((final - images) ** 2 * masks).sum() / masks.sum() / 3
        fp = -10 * torch.log10(mse.clamp(min=1e-10)).item()

    print(f"\n  Final PSNR: {fp:.1f} dB")

    # Save visual comparisons
    os.makedirs("overfit_output", exist_ok=True)
    for i in range(min(4, len(selected))):
        orig = (images[i].cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        masked = ((images[i] * (1 - masks[i])).cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        result = (final[i].cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        comp = np.concatenate([orig, masked, result], axis=1)
        Image.fromarray(comp).save(f"overfit_output/compare_{i:02d}.png")
    print(f"  Saved comparisons to overfit_output/")

    if fp > 35:
        print("  ✅ PASS — Architecture can learn manga patterns")
        return True
    elif fp > 28:
        print("  ⚠️  MARGINAL — May need more capacity")
        return True
    else:
        print("  ❌ FAIL — Architecture too small")
        return False


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(args):
    device = get_device()
    use_amp = args.use_amp and device.type == "cuda"
    print(f"Device: {device} (AMP: {use_amp})")

    # ---- Overfit test ----
    if not args.skip_overfit:
        ok = overfit_test(args.data_dir, device, args.base_ch)
        if not ok:
            print("\nOverfit test failed. Aborting training.")
            print("Try increasing --base_ch (default 48)")
            return

    # ---- Dataset ----
    # Mask convention: MangaInpaintDataset returns mask with 1=known, 0=masked
    # SmallInpaintModel expects 1=inpaint, 0=keep — we flip in training loop
    dataset = MangaInpaintDataset(
        args.data_dir, mask_dir=args.mask_dir,
        image_size=512, augment=True,
    )
    print(f"\nTraining images: {len(dataset)}")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        drop_last=True, persistent_workers=args.num_workers > 0,
    )

    # ---- Models ----
    gen = SmallInpaintModel(base_ch=args.base_ch).to(device)
    disc = PatchDiscriminator(in_ch=4, base_ch=64).to(device)
    ema = EMA(gen, decay=0.999)

    print(f"Generator:     {count_parameters(gen):,} params")
    print(f"Discriminator: {count_parameters(disc):,} params")

    # ---- Optimizers ----
    opt_g = torch.optim.Adam(gen.parameters(), lr=args.lr, betas=(0.0, 0.99))
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr, betas=(0.0, 0.99))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ---- Logging ----
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    try:
        from torch.utils.tensorboard import SummaryWriter
        from torchvision.utils import make_grid
        log_dir = os.path.join(args.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
        has_tb = True
    except ImportError:
        writer = None
        has_tb = False

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

    # ---- Training loop ----
    total_images = args.total_kimg * 1000
    images_seen = int(start_kimg * 1000)
    pbar = tqdm(total=total_images, initial=images_seen, unit="img",
                desc="Training", smoothing=0.1)

    data_iter = iter(loader)
    tick_start = time.time()

    while images_seen < total_images:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        # Dataset returns: image [-1,1], mask_known [1=known, 0=masked]
        real_neg11, mask_known = batch[0].to(device), batch[1].to(device)

        # Convert to model convention
        real = (real_neg11 + 1) * 0.5       # → [0,1]
        mask_inpaint = 1.0 - mask_known     # → 1=inpaint, 0=keep

        bs = real.size(0)

        # ---- Discriminator step ----
        gen.requires_grad_(False)
        disc.requires_grad_(True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            fake = gen(real, mask_inpaint)
            # Disc input: [rgb, mask_inpaint]
            d_real = disc(torch.cat([real, mask_inpaint], dim=1))
            d_fake = disc(torch.cat([fake.detach(), mask_inpaint], dim=1))
            loss_d = F.softplus(-d_real).mean() + F.softplus(d_fake).mean()

        opt_d.zero_grad(set_to_none=True)
        scaler.scale(loss_d).backward()
        scaler.step(opt_d)

        # ---- Generator step ----
        gen.requires_grad_(True)
        disc.requires_grad_(False)

        with torch.amp.autocast("cuda", enabled=use_amp):
            fake = gen(real, mask_inpaint)

            # Adversarial loss
            d_fake_g = disc(torch.cat([fake, mask_inpaint], dim=1))
            loss_g_adv = F.softplus(-d_fake_g).mean()

            # L1 reconstruction loss (masked region only)
            n_masked = mask_inpaint.sum().clamp(min=1)
            loss_l1 = ((fake - real).abs() * mask_inpaint).sum() / n_masked

            loss_g = loss_g_adv + loss_l1 * args.l1_weight

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
            ips = args.log_every * bs / max(elapsed, 1e-6)
            tick_start = time.time()

            if has_tb:
                writer.add_scalar("loss/d", loss_d.item(), images_seen)
                writer.add_scalar("loss/g_adv", loss_g_adv.item(), images_seen)
                writer.add_scalar("loss/g_l1", loss_l1.item(), images_seen)
                writer.add_scalar("loss/g_total", loss_g.item(), images_seen)
                writer.add_scalar("speed/ips", ips, images_seen)

            pbar.set_postfix({
                "kimg": f"{kimg:.1f}",
                "G": f"{loss_g.item():.3f}",
                "D": f"{loss_d.item():.3f}",
                "L1": f"{loss_l1.item():.3f}",
                "ips": f"{ips:.0f}",
            })

        # ---- Visualize ----
        if has_tb and images_seen % (args.vis_every * bs) < bs:
            with torch.no_grad():
                ema.model.eval()
                vis = ema.model(real[:4], mask_inpaint[:4])
                ema.model.train()
            grid = make_grid(torch.cat([
                real[:4],
                real[:4] * (1 - mask_inpaint[:4]),
                vis,
            ], dim=0), nrow=4, normalize=False)
            writer.add_image("samples", grid, images_seen)

        # ---- Checkpoint ----
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

            # Latest copy for easy resume
            latest_path = os.path.join(ckpt_dir, "latest.pt")
            torch.save({
                "gen": gen.state_dict(),
                "disc": disc.state_dict(),
                "opt_g": opt_g.state_dict(),
                "opt_d": opt_d.state_dict(),
                "ema": ema.state_dict(),
                "kimg": kimg,
                "args": vars(args),
            }, latest_path)

            # EMA-only for export
            best_path = os.path.join(ckpt_dir, "best.pt")
            torch.save({
                "gen": ema.state_dict(),
                "kimg": kimg,
                "base_ch": args.base_ch,
            }, best_path)

            tqdm.write(f"  Saved checkpoint: {ckpt_path}")

    pbar.close()
    if writer:
        writer.close()

    # ---- Export ONNX ----
    print("\n" + "=" * 60)
    print("Exporting to ONNX...")
    export_onnx(ema.model, args.base_ch, args.output_dir)
    print("=" * 60)
    print("Training complete!")


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_onnx(model: nn.Module, base_ch: int, output_dir: str):
    """Export trained model to ONNX."""
    model = copy.deepcopy(model)
    model.eval().cpu()

    dummy_img = torch.rand(1, 3, 512, 512)
    dummy_mask = (torch.rand(1, 1, 512, 512) > 0.5).float()

    with torch.no_grad():
        pt_out = model(dummy_img, dummy_mask)

    onnx_path = os.path.join(output_dir, "manga_inpaint.onnx")
    torch.onnx.export(
        model,
        (dummy_img, dummy_mask),
        onnx_path,
        opset_version=17,
        input_names=["image", "mask"],
        output_names=["output"],
        dynamic_axes=None,
    )
    file_size = os.path.getsize(onnx_path)
    print(f"  Saved: {onnx_path} ({file_size/1024/1024:.1f} MB)")

    # Verify
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        ort_out = sess.run(["output"], {
            "image": dummy_img.numpy(),
            "mask": dummy_mask.numpy(),
        })[0]
        diff = np.abs(pt_out.numpy() - ort_out).max()
        print(f"  ONNX verification: max_diff={diff:.6e} {'✅' if diff < 1e-4 else '⚠️'}")
    except ImportError:
        print("  (onnxruntime not installed, skipping verification)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Self-supervised manga inpainting training")

    # Data
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--mask_dir", type=str, default=None)

    # Model
    p.add_argument("--base_ch", type=int, default=48)

    # Training
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--total_kimg", type=int, default=2000)
    p.add_argument("--l1_weight", type=float, default=10.0)
    p.add_argument("--use_amp", action="store_true", default=True)
    p.add_argument("--no_amp", action="store_true")

    # Logging
    p.add_argument("--output_dir", type=str, default="runs/manga_inpaint")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--vis_every", type=int, default=200)
    p.add_argument("--save_every", type=int, default=2000)

    # System
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--skip_overfit", action="store_true")

    args = p.parse_args()
    if args.no_amp:
        args.use_amp = False

    train(args)


if __name__ == "__main__":
    main()
