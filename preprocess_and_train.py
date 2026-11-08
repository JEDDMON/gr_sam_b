#!/usr/bin/env python3
"""
Step 2 & 3: Precompute SAM ViT-B Teacher Embeddings to a fast memory-mapped file (.dat),
then train the Gr-SAM-B Student Encoder with mixed precision, gradient checkpointing,
and frequent resumable checkpoints optimized for an RTX 4050 (6GB VRAM) + 32GB RAM + 1TB NVMe.
"""

import os
import time
import json
import argparse
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from segment_anything import sam_model_registry
from model import GrassmannSAMEncoder

# Standard SAM pixel normalization constants
PIXEL_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
PIXEL_STD = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)


def load_and_preprocess_sam_image(img_path: Path, target_size: int = 1024) -> torch.Tensor:
    """
    Loads an image, resizes longest side to 1024, normalizes with SAM pixel stats,
    and zero-pads bottom/right to (3, 1024, 1024).
    """
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    scale = target_size / max(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    if (new_w, new_h) != (w, h):
        img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    arr = np.asarray(img, dtype=np.float32)  # (H, W, 3)
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)
    tensor = (tensor - PIXEL_MEAN) / PIXEL_STD

    pad_h = target_size - new_h
    pad_w = target_size - new_w
    if pad_h > 0 or pad_w > 0:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), value=0.0)
    return tensor


class RawImageDataset(Dataset):
    """Used during Phase 1 (Teacher Embedding Precomputation)."""
    def __init__(self, image_paths: List[Path]):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        try:
            tensor = load_and_preprocess_sam_image(path)
            return idx, tensor, True
        except Exception:
            # Fallback blank tensor if an image is corrupted
            return idx, torch.zeros(3, 1024, 1024, dtype=torch.float32), False


class MemmapDistillationDataset(Dataset):
    """
    Streams raw images + precomputed fp16 ViT-B teacher embeddings from a single
    contiguous numpy.memmap file. Linux automatically caches hot pages in your 32GB RAM.
    """
    def __init__(self, image_paths: List[Path], memmap_path: Path, num_samples: int):
        self.image_paths = image_paths[:num_samples]
        self.memmap_path = str(memmap_path)
        self.num_samples = num_samples
        self._mmap = None

    def _get_mmap(self):
        # Lazy initialization per DataLoader worker process
        if self._mmap is None:
            self._mmap = np.memmap(
                self.memmap_path,
                dtype=np.float16,
                mode="r",
                shape=(self.num_samples, 256, 64, 64),
            )
        return self._mmap

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        img_tensor = load_and_preprocess_sam_image(self.image_paths[idx])
        mmap = self._get_mmap()
        # Copy slice from OS page cache / NVMe into a PyTorch tensor
        teacher_emb = torch.from_numpy(np.array(mmap[idx], dtype=np.float16, copy=True))
        return img_tensor, teacher_emb


class FeatureDistillationLoss(nn.Module):
    """Combines MSE Loss with Channel-wise Cosine Distance on the (B, 256, 64, 64) grid."""
    def __init__(self, alpha_cos: float = 0.5):
        super().__init__()
        self.mse = nn.MSELoss()
        self.alpha_cos = alpha_cos

    def forward(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
        target = teacher_feat.to(student_feat.dtype)
        loss_mse = self.mse(student_feat, target)
        cos_sim = F.cosine_similarity(student_feat, target, dim=1)
        loss_cos = 1.0 - cos_sim.mean()
        return loss_mse + self.alpha_cos * loss_cos


def precompute_teacher_embeddings(
    image_paths: List[Path],
    sam_ckpt: Path,
    memmap_path: Path,
    progress_path: Path,
    batch_size: int = 2,
    num_workers: int = 6,
    device: str = "cuda",
):
    N = len(image_paths)
    expected_bytes = N * 256 * 64 * 64 * 2  # fp16 = 2 bytes
    print(f"[*] Precomputing {N} ViT-B embeddings -> {memmap_path} ({expected_bytes / (1024**3):.2f} GB)")

    start_idx = 0
    if progress_path.exists() and memmap_path.exists():
        with open(progress_path, "r") as f:
            start_idx = json.load(f).get("completed_index", 0)
        if start_idx >= N:
            print(f"[✓] All {N} teacher embeddings are already precomputed!")
            return
        print(f"[*] Resuming precomputation from index {start_idx}/{N}...")
        mmap_mode = "r+"
    else:
        mmap_mode = "w+"

    mmap = np.memmap(str(memmap_path), dtype=np.float16, mode=mmap_mode, shape=(N, 256, 64, 64))

    # Load only the SAM ViT-B image_encoder to save VRAM
    sam = sam_model_registry["vit_b"](checkpoint=str(sam_ckpt))
    teacher = sam.image_encoder.to(device).eval()
    del sam
    for p in teacher.parameters():
        p.requires_grad = False

    subset_paths = image_paths[start_idx:]
    dataset = RawImageDataset(subset_paths)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    pbar = tqdm(total=N, initial=start_idx, desc="Precomputing ViT-B Features", unit="img")
    processed = start_idx

    with torch.no_grad():
        for rel_indices, imgs, valid_mask in loader:
            imgs = imgs.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                feats = teacher(imgs)  # (B, 256, 64, 64)

            feats_np = feats.cpu().numpy().astype(np.float16)
            abs_indices = (rel_indices + start_idx).numpy()
            mmap[abs_indices] = feats_np

            processed += len(imgs)
            pbar.update(len(imgs))

            # Flush to disk and record progress every 500 images
            if processed % 500 < batch_size or processed == N:
                mmap.flush()
                with open(progress_path, "w") as f:
                    json.dump({"completed_index": processed, "total": N}, f)

    pbar.close()
    del teacher
    torch.cuda.empty_cache()
    print(f"[✓] Precomputation complete! Saved to {memmap_path}")


def save_checkpoint(
    ckpt_dir: Path,
    filename: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    best_loss: float,
):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / filename
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "best_loss": best_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
    }
    # Atomic write so power loss never corrupts latest.pt
    tmp_path = path.with_suffix(".tmp")
    torch.save(state, tmp_path)
    tmp_path.replace(path)


def train_student(
    image_paths: List[Path],
    memmap_path: Path,
    sam_ckpt: Path,
    ckpt_dir: Path,
    epochs: int = 35,
    warmup_freeze_epochs: int = 3,
    batch_size: int = 2,
    grad_accum_steps: int = 4,
    lr: float = 3e-4,
    num_workers: int = 8,
    save_every_steps: int = 1000,
    device: str = "cuda",
):
    N = len(image_paths)
    dataset = MemmapDistillationDataset(image_paths, memmap_path, N)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )

    student = GrassmannSAMEncoder(
        img_size=1024,
        patch_size=16,
        embed_dim=768,
        depth=12,
        rank=16,
        out_chans=256,
        use_checkpoint=False,
    )
    student.load_sam_stem_and_neck(str(sam_ckpt), freeze_stem_neck=(warmup_freeze_epochs > 0))
    student = student.to(device)

    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)
    total_steps = (len(loader) // grad_accum_steps) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1), eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    criterion = FeatureDistillationLoss(alpha_cos=0.5)

    start_epoch = 0
    global_step = 0
    best_loss = float("inf")

    latest_ckpt = ckpt_dir / "latest.pt"
    if latest_ckpt.exists():
        print(f"[*] Resuming training from {latest_ckpt}...")
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=True)
        student.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"]
        global_step = ckpt["global_step"]
        best_loss = ckpt["best_loss"]
        print(f"[✓] Resumed at Epoch {start_epoch + 1}, Global Step {global_step} (Best Loss: {best_loss:.5f})")

    for epoch in range(start_epoch, epochs):
        # Unfreeze stem & neck after warm-up epochs
        if epoch >= warmup_freeze_epochs:
            student.set_stem_neck_trainable(True)
        else:
            student.set_stem_neck_trainable(False)

        student.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch [{epoch+1}/{epochs}]")
        for step, (imgs, teacher_emb) in pbar:
            imgs = imgs.to(device, non_blocking=True)
            teacher_emb = teacher_emb.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                student_emb = student(imgs)
                loss = criterion(student_emb, teacher_emb)
                scaled_loss = loss / grad_accum_steps

            scaler.scale(scaled_loss).backward()
            running_loss += loss.item()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                avg_loss = running_loss / (step + 1)
                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    "step": global_step,
                })

                # Intra-epoch step checkpointing
                if global_step % save_every_steps == 0:
                    save_checkpoint(
                        ckpt_dir, "latest.pt", student, optimizer, scheduler, scaler, epoch, global_step, best_loss
                    )

        epoch_loss = running_loss / len(loader)
        print(f"[✓] Epoch {epoch+1} Complete | Avg Loss: {epoch_loss:.5f}")

        # Save end-of-epoch checkpoints
        save_checkpoint(ckpt_dir, "latest.pt", student, optimizer, scheduler, scaler, epoch + 1, global_step, best_loss)
        save_checkpoint(ckpt_dir, f"epoch_{epoch+1:03d}.pt", student, optimizer, scheduler, scaler, epoch + 1, global_step, best_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_checkpoint(ckpt_dir, "best_model.pt", student, optimizer, scheduler, scaler, epoch + 1, global_step, best_loss)
            # Also export standalone student weights ready for Hugging Face / SAM drop-in
            torch.save(student.state_dict(), ckpt_dir / "gr_sam_b_weights.pth")
            print(f"[★] New Best Loss ({best_loss:.5f})! Saved `best_model.pt` and `gr_sam_b_weights.pth`.")


def main():
    parser = argparse.ArgumentParser(description="Precompute ViT-B embeddings and train Gr-SAM-B Student")
    parser.add_argument("--data_dir", type=str, default="./data", help="Root data directory from download_data.py")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints_gr_sam", help="Directory to save student checkpoints")
    parser.add_argument("--mode", type=str, choices=["all", "preprocess", "train"], default="all", help="Pipeline stage to run")
    parser.add_argument("--max_images", type=int, default=100000, help="Max number of images to use")
    parser.add_argument("--epochs", type=int, default=35, help="Total distillation epochs")
    parser.add_argument("--warmup_freeze_epochs", type=int, default=3, help="Epochs to keep stem/neck frozen")
    parser.add_argument("--batch_size", type=int, default=2, help="Micro-batch size per GPU step (2 or 3 for 6GB VRAM)")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader CPU workers (8 is ideal for i5-13420H + 32GB RAM)")
    parser.add_argument("--save_every_steps", type=int, default=1000, help="Save latest.pt every N optimizer steps")
    args = parser.parse_args()

    data_root = Path(args.data_dir)
    sam_ckpt = data_root / "checkpoints" / "sam_vit_b_01ec64.pth"
    images_dir = data_root / "images"
    memmap_path = data_root / "teacher_vit_b_fp16.dat"
    progress_path = data_root / "preprocess_progress.json"
    ckpt_dir = Path(args.ckpt_dir)

    if not sam_ckpt.exists():
        raise FileNotFoundError(f"SAM checkpoint not found at {sam_ckpt}. Run `python download_data.py` first.")

    image_paths = sorted(images_dir.glob("*/*.jpg"))[:args.max_images]
    if not image_paths:
        raise RuntimeError(f"No images found in {images_dir}. Run `python download_data.py` first.")

    print(f"[*] Found {len(image_paths)} images in {images_dir}")

    if args.mode in ("all", "preprocess"):
        precompute_teacher_embeddings(
            image_paths=image_paths,
            sam_ckpt=sam_ckpt,
            memmap_path=memmap_path,
            progress_path=progress_path,
            batch_size=args.batch_size,
            num_workers=min(args.num_workers, 6),
        )

    if args.mode in ("all", "train"):
        train_student(
            image_paths=image_paths,
            memmap_path=memmap_path,
            sam_ckpt=sam_ckpt,
            ckpt_dir=ckpt_dir,
            epochs=args.epochs,
            warmup_freeze_epochs=args.warmup_freeze_epochs,
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum,
            lr=args.lr,
            num_workers=args.num_workers,
            save_every_steps=args.save_every_steps,
        )


if __name__ == "__main__":
    main()
