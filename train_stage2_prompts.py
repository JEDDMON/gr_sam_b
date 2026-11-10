#!/usr/bin/env python3
"""
Phase 2: Full Multi-Modal Prompt-Guided Decoder Alignment (`train_stage2_prompts.py`).
Supports ALL SAM prompt modalities & combinations:
  1. Single positive foreground points (label=1) alongside multimask_output=True (3 mask tokens).
  2. Multi-point prompts combining positive points (label=1) and negative exclusion points (label=0).
  3. Standalone bounding box prompts (boxes=[x1, y1, x2, y2]).
  4. Combined Box + Point prompts (box + positive/negative refinement points).
  5. Iterative Dense Mask Prompts (feeding coarse 256x256 logits back into `masks=` for step-2 refinement).

Optimized for 6GB VRAM:
  - Uses precomputed teacher embeddings from `teacher_vit_b_fp16.dat` (zero ViT-B VRAM!).
  - Evaluates all 5 prompt regimes in lightweight batched decoder passes.
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from segment_anything import sam_model_registry
from model import GrassmannSAMEncoder
from preprocess_and_train import MemmapDistillationDataset, FeatureDistillationLoss


def sample_comprehensive_prompts(
    batch_size: int,
    num_single_pts: int = 4,
    num_pos_neg_pairs: int = 4,
    num_boxes: int = 4,
    num_box_pt_combos: int = 4,
    img_size: int = 1024,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    Samples all 4 spatial prompt regimes inside [64, img_size - 64]:
      1. single_pts:    (B * num_single_pts, 1, 2), labels = 1
      2. pos_neg_pts:   (B * num_pos_neg_pairs, 3, 2), labels = [1, 1, 0] or [1, 0, 0]
      3. boxes_only:    (B * num_boxes, 4)
      4. combo_boxes:   (B * num_box_pt_combos, 4) + combo_pts: (B * num_box_pt_combos, 2, 2) with labels [1, 0]
    """
    margin = 64.0
    span = float(img_size) - 2.0 * margin

    # 1. Single Positive Points
    n1 = batch_size * num_single_pts
    single_pts = margin + torch.rand(n1, 1, 2, device=device) * span
    single_lbls = torch.ones(n1, 1, dtype=torch.int64, device=device)

    # 2. Multi-Point (Positive + Negative Points)
    # Sample a positive anchor point, a nearby positive point, and an offset negative background point
    n2 = batch_size * num_pos_neg_pairs
    pos_anchor = margin + torch.rand(n2, 1, 2, device=device) * span
    pos_near = (pos_anchor + (torch.rand(n2, 1, 2, device=device) - 0.5) * 96.0).clamp(16.0, img_size - 16.0)
    # Negative point placed 96px to 320px away from positive anchor
    angle = torch.rand(n2, 1, 1, device=device) * (2.0 * np.pi)
    dist = 96.0 + torch.rand(n2, 1, 1, device=device) * 224.0
    neg_offset = torch.cat([torch.cos(angle) * dist, torch.sin(angle) * dist], dim=-1)
    neg_pt = (pos_anchor + neg_offset).clamp(16.0, img_size - 16.0)

    pos_neg_pts = torch.cat([pos_anchor, pos_near, neg_pt], dim=1)  # (n2, 3, 2)
    pos_neg_lbls = torch.tensor([[1, 1, 0]], dtype=torch.int64, device=device).expand(n2, 3).contiguous()

    # Helper to sample valid [x1, y1, x2, y2] boxes
    def _sample_boxes(count: int) -> torch.Tensor:
        cx = margin + torch.rand(count, device=device) * span
        cy = margin + torch.rand(count, device=device) * span
        bw = 64.0 + torch.rand(count, device=device) * 448.0
        bh = 64.0 + torch.rand(count, device=device) * 448.0
        x1 = (cx - bw * 0.5).clamp(0.0, img_size - 16.0)
        y1 = (cy - bh * 0.5).clamp(0.0, img_size - 16.0)
        x2 = (cx + bw * 0.5).clamp(x1 + 16.0, float(img_size))
        y2 = (cy + bh * 0.5).clamp(y1 + 16.0, float(img_size))
        return torch.stack([x1, y1, x2, y2], dim=-1)

    # 3. Standalone Boxes
    n3 = batch_size * num_boxes
    boxes_only = _sample_boxes(n3)

    # 4. Combined Box + Positive (interior) + Negative (near-border/exterior) Points
    n4 = batch_size * num_box_pt_combos
    combo_boxes = _sample_boxes(n4)
    bx1, by1, bx2, by2 = combo_boxes.unbind(dim=-1)
    bw, bh = bx2 - bx1, by2 - by1

    # Positive point strictly inside inner 60% of the box
    in_x = bx1 + (0.2 + 0.6 * torch.rand(n4, device=device)) * bw
    in_y = by1 + (0.2 + 0.6 * torch.rand(n4, device=device)) * bh
    pt_in = torch.stack([in_x, in_y], dim=-1).unsqueeze(1)  # (n4, 1, 2)

    # Negative point near box corner / just inside or outside boundary to sharpen edges
    out_x = (bx2 + (torch.rand(n4, device=device) - 0.3) * 48.0).clamp(0.0, float(img_size))
    out_y = (by2 + (torch.rand(n4, device=device) - 0.3) * 48.0).clamp(0.0, float(img_size))
    pt_out = torch.stack([out_x, out_y], dim=-1).unsqueeze(1)  # (n4, 1, 2)

    combo_pts = torch.cat([pt_in, pt_out], dim=1)  # (n4, 2, 2)
    combo_lbls = torch.tensor([[1, 0]], dtype=torch.int64, device=device).expand(n4, 2).contiguous()

    return {
        "single_pts": single_pts,
        "single_lbls": single_lbls,
        "pos_neg_pts": pos_neg_pts,
        "pos_neg_lbls": pos_neg_lbls,
        "boxes_only": boxes_only,
        "combo_boxes": combo_boxes,
        "combo_pts": combo_pts,
        "combo_lbls": combo_lbls,
        "counts": (num_single_pts, num_pos_neg_pairs, num_boxes, num_box_pt_combos),
    }


def soft_dice_loss(student_logits: torch.Tensor, teacher_probs: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    student_probs = torch.sigmoid(student_logits).flatten(1)
    target_probs = teacher_probs.flatten(1)
    intersection = (student_probs * target_probs).sum(dim=1)
    denominator = student_probs.sum(dim=1) + target_probs.sum(dim=1)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return (1.0 - dice).mean()


def run_all_prompt_modalities(
    img_emb: torch.Tensor,
    prompt_encoder: nn.Module,
    mask_decoder: nn.Module,
    prompts: Dict[str, torch.Tensor],
    coarse_mask_input: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Executes all 5 SAM prompt combinations through the frozen PromptEncoder and MaskDecoder:
      Mode 1: Single positive point (with `multimask_output=True` to align all 3 ambiguity mask heads!)
      Mode 2: Multi-point (Positive + Negative points, `multimask_output=False`)
      Mode 3: Box only (`multimask_output=False`)
      Mode 4: Box + Positive/Negative Point combination (`multimask_output=False`)
      Mode 5: Iterative Dense Mask Prompt + Positive/Negative Points (`masks=coarse_mask_input`)
    Returns:
      all_masks: (Total_Queries, 1, 256, 256)
      all_ious:  (Total_Queries, 1)
      coarse_masks_for_refinement: detached (N2, 1, 256, 256) logits from Mode 2
    """
    dense_pe = prompt_encoder.get_dense_pe()
    n_single, n_pair, n_box, n_combo = prompts["counts"]

    # --- Mode 1: Single Foreground Point (multimask_output=True -> 3 masks per query) ---
    sp1, dn1 = prompt_encoder(points=(prompts["single_pts"], prompts["single_lbls"]), boxes=None, masks=None)
    m1, iou1 = mask_decoder(
        image_embeddings=img_emb.repeat_interleave(n_single, dim=0),
        image_pe=dense_pe,
        sparse_prompt_embeddings=sp1,
        dense_prompt_embeddings=dn1,
        multimask_output=True,  # Aligns all 3 hierarchical ambiguity tokens!
    )
    # Reshape (N1, 3, 256, 256) -> (N1 * 3, 1, 256, 256)
    m1 = m1.reshape(-1, 1, 256, 256)
    iou1 = iou1.reshape(-1, 1)

    # --- Mode 2: Positive + Negative Points (multimask_output=False) ---
    sp2, dn2 = prompt_encoder(points=(prompts["pos_neg_pts"], prompts["pos_neg_lbls"]), boxes=None, masks=None)
    emb_pair = img_emb.repeat_interleave(n_pair, dim=0)
    m2, iou2 = mask_decoder(
        image_embeddings=emb_pair,
        image_pe=dense_pe,
        sparse_prompt_embeddings=sp2,
        dense_prompt_embeddings=dn2,
        multimask_output=False,
    )

    # --- Mode 3: Bounding Box Only ---
    sp3, dn3 = prompt_encoder(points=None, boxes=prompts["boxes_only"], masks=None)
    m3, iou3 = mask_decoder(
        image_embeddings=img_emb.repeat_interleave(n_box, dim=0),
        image_pe=dense_pe,
        sparse_prompt_embeddings=sp3,
        dense_prompt_embeddings=dn3,
        multimask_output=False,
    )

    # --- Mode 4: Combined Box + Positive & Negative Refinement Points ---
    sp4, dn4 = prompt_encoder(
        points=(prompts["combo_pts"], prompts["combo_lbls"]),
        boxes=prompts["combo_boxes"],
        masks=None,
    )
    m4, iou4 = mask_decoder(
        image_embeddings=img_emb.repeat_interleave(n_combo, dim=0),
        image_pe=dense_pe,
        sparse_prompt_embeddings=sp4,
        dense_prompt_embeddings=dn4,
        multimask_output=False,
    )

    # --- Mode 5: Dense Mask Prompt + Points (2nd-step interactive refinement) ---
    mask_in = m2.detach() if coarse_mask_input is None else coarse_mask_input
    sp5, dn5 = prompt_encoder(
        points=(prompts["pos_neg_pts"], prompts["pos_neg_lbls"]),
        boxes=None,
        masks=mask_in,
    )
    m5, iou5 = mask_decoder(
        image_embeddings=emb_pair,
        image_pe=dense_pe,
        sparse_prompt_embeddings=sp5,
        dense_prompt_embeddings=dn5,
        multimask_output=False,
    )

    all_masks = torch.cat([m1, m2, m3, m4, m5], dim=0)
    all_ious = torch.cat([iou1, iou2, iou3, iou4, iou5], dim=0)
    return all_masks, all_ious, m2.detach()


def train_stage2(
    image_paths: List[Path],
    memmap_path: Path,
    sam_ckpt: Path,
    student_weights_path: Path,
    ckpt_dir: Path,
    epochs: int = 2,
    batch_size: int = 1,
    grad_accum_steps: int = 8,
    lr: float = 5e-5,
    num_workers: int = 8,
    save_every_steps: int = 500,
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

    print(f"[*] Loading frozen SAM Prompt Encoder & Mask Decoder from {sam_ckpt}...")
    sam = sam_model_registry["vit_b"](checkpoint=str(sam_ckpt))
    prompt_encoder = sam.prompt_encoder.to(device).eval()
    mask_decoder = sam.mask_decoder.to(device).eval()
    del sam.image_encoder
    del sam
    torch.cuda.empty_cache()

    for p in prompt_encoder.parameters():
        p.requires_grad = False
    for p in mask_decoder.parameters():
        p.requires_grad = False

    student = GrassmannSAMEncoder(
        img_size=1024,
        patch_size=16,
        embed_dim=768,
        depth=12,
        rank=16,
        out_chans=256,
        use_checkpoint=True,
    )
    print(f"[*] Loading Stage 1 Student weights from {student_weights_path}...")
    state = torch.load(student_weights_path, map_location="cpu", weights_only=True)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    student.load_state_dict(state)
    student.set_stem_neck_trainable(True)
    student = student.to(device)

    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)
    total_steps = max((len(loader) // grad_accum_steps) * epochs, 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    feat_criterion = FeatureDistillationLoss(alpha_cos=0.5)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    global_step = 0

    for epoch in range(epochs):
        student.train()
        running_total = 0.0
        running_mask = 0.0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Stage 2 All-Prompts [{epoch+1}/{epochs}]")
        for step, (imgs, teacher_emb) in pbar:
            imgs = imgs.to(device, non_blocking=True)
            teacher_emb = teacher_emb.to(device, non_blocking=True)

            prompts = sample_comprehensive_prompts(batch_size=imgs.shape[0], device=device)

            # A. Teacher Decoder Pass across all 5 prompt combinations
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                t_masks, t_iou, t_coarse = run_all_prompt_modalities(
                    teacher_emb, prompt_encoder, mask_decoder, prompts, coarse_mask_input=None
                )
                t_probs = torch.sigmoid(t_masks)

            # B. Student Encoder + Frozen Decoder Pass across all 5 prompt combinations
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                s_emb = student(imgs)
                s_masks, s_iou, _ = run_all_prompt_modalities(
                    s_emb, prompt_encoder, mask_decoder, prompts, coarse_mask_input=t_coarse
                )

                loss_feat = feat_criterion(s_emb, teacher_emb)
                loss_bce = F.binary_cross_entropy_with_logits(s_masks, t_probs)
                loss_dice = soft_dice_loss(s_masks, t_probs)
                loss_iou = F.mse_loss(s_iou, t_iou)

                loss_mask = loss_bce + loss_dice
                total_loss = loss_feat + 1.5 * loss_mask + 0.5 * loss_iou
                scaled_loss = total_loss / grad_accum_steps

            scaler.scale(scaled_loss).backward()
            running_total += total_loss.item()
            running_mask += loss_mask.item()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                avg_tot = running_total / (step + 1)
                avg_msk = running_mask / (step + 1)
                pbar.set_postfix({
                    "tot": f"{avg_tot:.4f}",
                    "msk": f"{avg_msk:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    "step": global_step,
                })

                if global_step % save_every_steps == 0:
                    torch.save(student.state_dict(), ckpt_dir / "gr_sam_b_stage2_latest.pth")

        epoch_loss = running_total / len(loader)
        print(f"[✓] Stage 2 Epoch {epoch+1} Complete | Avg Total Loss: {epoch_loss:.5f}")
        torch.save(student.state_dict(), ckpt_dir / f"gr_sam_b_stage2_ep{epoch+1}.pth")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(student.state_dict(), ckpt_dir / "gr_sam_b_stage2_best.pth")
            print(f"[★] Saved `gr_sam_b_stage2_best.pth` (Loss: {best_loss:.5f})")


def main():
    parser = argparse.ArgumentParser(description="Stage 2 Full Multi-Modal Prompt Alignment for Gr-SAM-B")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints_gr_sam")
    parser.add_argument("--student_ckpt", type=str, default="./checkpoints_gr_sam/gr_sam_b_weights.pth")
    parser.add_argument("--max_images", type=int, default=100000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--save_every_steps", type=int, default=500)
    args = parser.parse_args()

    data_root = Path(args.data_dir)
    sam_ckpt = data_root / "checkpoints" / "sam_vit_b_01ec64.pth"
    images_dir = data_root / "images"
    memmap_path = data_root / "teacher_vit_b_fp16.dat"
    image_paths = sorted(images_dir.glob("*/*.jpg"))[:args.max_images]

    train_stage2(
        image_paths=image_paths,
        memmap_path=memmap_path,
        sam_ckpt=sam_ckpt,
        student_weights_path=Path(args.student_ckpt),
        ckpt_dir=Path(args.ckpt_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        num_workers=args.num_workers,
        save_every_steps=args.save_every_steps,
    )


if __name__ == "__main__":
    main()
