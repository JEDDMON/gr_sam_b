#!/usr/bin/env python3
"""
gr_sam_wrapper.py
Drop-in Wrapper for Gr-SAM-B (`GrassmannSAMEncoder`) matching Meta's `segment_anything` API.

Features:
  1. `build_gr_sam_b()`: Builds a standard `Sam` object with `GrassmannSAMEncoder` swapped in,
     compatible with both `SamPredictor` and `SamAutomaticMaskGenerator`.
  2. `GrSamPredictor`: Enhanced drop-in replacement for `SamPredictor` with two operating modes:
     - `mode="single_tile"`: Standard SAM behavior (resizes longest side to 1024, single GPU pass).
     - `mode="multi_tile"` : Gigapixel halo-tiling mode with Multi-GPU parallel CUDA stream execution,
                             disk-backed `numpy.memmap` support, and O(1) viewport-cropped decoding.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from segment_anything import sam_model_registry, SamPredictor
from segment_anything.modeling import Sam

from model import GrassmannSAMEncoder


def build_gr_sam_b(
    sam_checkpoint: str = "./data/checkpoints/sam_vit_b_01ec64.pth",
    student_checkpoint: Optional[str] = "./checkpoints_gr_sam/gr_sam_b_stage2_best.pth",
    device: str = "cuda",
    eval_mode: bool = True,
) -> Sam:
    """
    Constructs a standard Meta `Sam` model instance with the ViT-B `image_encoder`
    replaced by your trained `GrassmannSAMEncoder` (`Gr-SAM-B`).
    """
    sam = sam_model_registry["vit_b"](checkpoint=str(sam_checkpoint))

    student_encoder = GrassmannSAMEncoder(
        img_size=1024,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        rank=16,
        out_chans=256,
        use_checkpoint=False,
    )

    if student_checkpoint is not None and Path(student_checkpoint).exists():
        state = torch.load(student_checkpoint, map_location="cpu", weights_only=True)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        student_encoder.load_state_dict(state)
        print(f"[✓] Loaded Gr-SAM-B encoder weights from: {student_checkpoint}")
    else:
        print(f"[!] Warning: Student checkpoint '{student_checkpoint}' not found. Using current weights.")

    # Swap ViT-B encoder with Gr-SAM-B encoder
    del sam.image_encoder
    sam.image_encoder = student_encoder
    sam = sam.to(device)
    if eval_mode:
        sam.eval()
    return sam


class GrSamPredictor(SamPredictor):
    """
    Drop-in replacement for Meta's `SamPredictor` supporting both:
      - `mode="single_tile"`: Standard 1024x1024 resize + pad inference.
      - `mode="multi_tile"` : Multi-GPU parallel halo-padded tiling for high-res / gigapixel scans.
    """

    def __init__(
        self,
        sam_model: Sam,
        mode: str = "single_tile",
        device_ids: Optional[List[int]] = None,
        halo_patches: int = 16,
        dtype: torch.dtype = torch.float16,
        memmap_dir: Optional[str] = None,
    ) -> None:
        """
        Args:
            sam_model: Model built via `build_gr_sam_b()`.
            mode: `"single_tile"`, `"multi_tile"`, or `"auto"` (triggers `multi_tile` if max(H, W) > 1024).
            device_ids: List of GPU IDs for parallel tiling, e.g. `[0, 1, 2, 3]`. Defaults to all visible GPUs.
            halo_patches: Border patches cropped from each side of a 64x64 tile in `multi_tile` mode.
                          Default 16 (= 256px halo on all sides, leaving a 32x32 patch / 512x512px core).
            dtype: Precision used during `multi_tile` encoding (`torch.float16` or `torch.bfloat16`).
            memmap_dir: Optional directory to store stitched gigapixel embeddings on NVMe disk instead of RAM.
        """
        super().__init__(sam_model)
        assert mode in ("single_tile", "multi_tile", "auto"), "mode must be 'single_tile', 'multi_tile', or 'auto'"
        assert 0 <= halo_patches < 32, "halo_patches must be in [0, 31] so core_patches = (64 - 2*halo) > 0"

        self.default_mode = mode
        self.active_mode = mode
        self.halo_patches = halo_patches
        self.core_patches = 64 - 2 * halo_patches  # 32 patches = 512px when halo_patches=16
        self.tile_dtype = dtype
        self.memmap_dir = Path(memmap_dir) if memmap_dir else None

        if device_ids is None:
            if torch.cuda.is_available() and self.device.type == "cuda":
                self.device_ids = list(range(torch.cuda.device_count()))
            else:
                self.device_ids = []
        else:
            self.device_ids = list(device_ids)

        # Lazy-initialized multi-GPU encoder replicas & CUDA streams
        self._encoder_replicas: List[nn.Module] = []
        self._cuda_streams: List[torch.cuda.Stream] = []

        # Multi-tile state
        self.global_features: Optional[Union[np.ndarray, torch.Tensor]] = None
        self.padded_hw: Optional[Tuple[int, int]] = None
        self.grid_patches_hw: Optional[Tuple[int, int]] = None
        self._mmap_path: Optional[Path] = None

    def _init_multigpu_replicas(self) -> None:
        """Replicates the Gr-SAM-B image encoder across `self.device_ids` once."""
        if self._encoder_replicas:
            return

        if not self.device_ids:
            # Single CPU or default device fallback
            enc = self.model.image_encoder.to(device=self.device, dtype=self.tile_dtype).eval()
            self._encoder_replicas = [enc]
            self._cuda_streams = []
            return

        for idx, dev_id in enumerate(self.device_ids):
            dev = f"cuda:{dev_id}"
            if idx == 0 and str(self.device) == dev:
                rep = self.model.image_encoder.to(device=dev, dtype=self.tile_dtype).eval()
            else:
                import copy
                rep = copy.deepcopy(self.model.image_encoder).to(device=dev, dtype=self.tile_dtype).eval()
            self._encoder_replicas.append(rep)
            self._cuda_streams.append(torch.cuda.Stream(device=dev))

    def set_image(
        self,
        image: np.ndarray,
        image_format: str = "RGB",
        mode: Optional[str] = None,
    ) -> None:
        """
        Calculates the image embeddings for the provided image (`H, W, 3` uint8 numpy array or memmap).
        Automatically routes to `single_tile` or multi-GPU `multi_tile` based on `mode`.
        """
        assert image_format in ["RGB", "BGR"], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.model.image_format:
            image = image[..., ::-1]

        chosen_mode = mode if mode is not None else self.default_mode
        h, w = image.shape[:2]
        if chosen_mode == "auto":
            chosen_mode = "multi_tile" if max(h, w) > 1024 else "single_tile"

        self.active_mode = chosen_mode

        if chosen_mode == "single_tile":
            self.global_features = None
            super().set_image(image, image_format=self.model.image_format)
        else:
            self._set_image_multi_tile(image)

    @torch.no_grad()
    def _set_image_multi_tile(self, image: np.ndarray) -> None:
        """
        Slices an arbitrary-size or gigapixel image into overlapping 1024x1024 tiles,
        dispatches them in parallel across `self.device_ids`, crops the `halo_patches` border,
        and stitches a seamless global feature map of shape `(256, H_patches, W_patches)`.
        """
        self.reset_image()
        orig_h, orig_w = image.shape[:2]
        self.original_size = (orig_h, orig_w)

        # Ensure canvas dimensions are at least 1024 and divisible by 16 (1 patch = 16x16 px)
        pad_h = max(1024, ((orig_h + 15) // 16) * 16)
        pad_w = max(1024, ((orig_w + 15) // 16) * 16)
        self.padded_hw = (pad_h, pad_w)
        self.input_size = (pad_h, pad_w)

        total_py = pad_h // 16
        total_px = pad_w // 16
        self.grid_patches_hw = (total_py, total_px)

        # Allocate stitched feature buffer: (256, total_py, total_px) in FP16
        # Uses disk-backed numpy.memmap if memmap_dir is set or if feature map > 2 GB
        feat_bytes = 256 * total_py * total_px * 2
        if self.memmap_dir is not None or feat_bytes > 2 * (1024 ** 3):
            mmap_dir = self.memmap_dir if self.memmap_dir is not None else Path("./tmp_grsam_mmap")
            mmap_dir.mkdir(parents=True, exist_ok=True)
            self._mmap_path = mmap_dir / f"stitched_emb_{total_py}x{total_px}.dat"
            self.global_features = np.memmap(
                str(self._mmap_path), dtype=np.float16, mode="w+", shape=(256, total_py, total_px)
            )
        else:
            self._mmap_path = None
            self.global_features = np.zeros((256, total_py, total_px), dtype=np.float16)

        self._init_multigpu_replicas()
        num_workers = len(self._encoder_replicas)

        # Build list of core patch regions across the global patch grid
        core = self.core_patches
        halo = self.halo_patches
        tiles: List[Tuple[int, int, int, int]] = []
        for py0 in range(0, total_py, core):
            for px0 in range(0, total_px, core):
                py1 = min(py0 + core, total_py)
                px1 = min(px0 + core, total_px)
                tiles.append((py0, py1, px0, px1))

        # Pixel normalization constants on CPU for fast tile preparation
        mean_np = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(1, 1, 3)
        std_np = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(1, 1, 3)

        # Execute tiles in parallel multi-GPU waves
        for wave_start in range(0, len(tiles), num_workers):
            wave = tiles[wave_start : wave_start + num_workers]
            gpu_jobs = []

            for w_idx, (py0, py1, px0, px1) in enumerate(wave):
                rep = self._encoder_replicas[w_idx]
                dev = next(rep.parameters()).device
                stream = self._cuda_streams[w_idx] if self._cuda_streams else None

                # Center a 64x64 patch (1024x1024 px) window around the core, clamped to valid canvas bounds
                wy0 = max(0, min(py0 - halo, total_py - 64))
                wx0 = max(0, min(px0 - halo, total_px - 64))

                y_px0, y_px1 = wy0 * 16, (wy0 + 64) * 16
                x_px0, x_px1 = wx0 * 16, (wx0 + 64) * 16

                # Extract raw pixel slice, zero-padding bottom/right if the image edge is < padded boundary
                valid_y1 = min(y_px1, orig_h)
                valid_x1 = min(x_px1, orig_w)
                raw_crop = np.asarray(image[y_px0:valid_y1, x_px0:valid_x1, :], dtype=np.float32)
                norm_crop = (raw_crop - mean_np) / std_np

                if norm_crop.shape[0] < 1024 or norm_crop.shape[1] < 1024:
                    tile_hwc = np.zeros((1024, 1024, 3), dtype=np.float32)
                    tile_hwc[: norm_crop.shape[0], : norm_crop.shape[1], :] = norm_crop
                else:
                    tile_hwc = norm_crop

                # Convert (1024, 1024, 3) -> (1, 3, 1024, 1024)
                tile_chw = np.ascontiguousarray(tile_hwc.transpose(2, 0, 1))

                if stream is not None:
                    with torch.cuda.stream(stream):
                        t_gpu = torch.from_numpy(tile_chw).unsqueeze(0).to(device=dev, dtype=self.tile_dtype, non_blocking=True)
                        emb_64 = rep(t_gpu)  # (1, 256, 64, 64)
                        ry0, ry1 = py0 - wy0, py1 - wy0
                        rx0, rx1 = px0 - wx0, px1 - wx0
                        cropped_core = emb_64[0, :, ry0:ry1, rx0:rx1]
                        gpu_jobs.append((w_idx, py0, py1, px0, px1, cropped_core))
                else:
                    t_gpu = torch.from_numpy(tile_chw).unsqueeze(0).to(device=dev, dtype=self.tile_dtype)
                    emb_64 = rep(t_gpu)
                    ry0, ry1 = py0 - wy0, py1 - wy0
                    rx0, rx1 = px0 - wx0, px1 - wx0
                    cropped_core = emb_64[0, :, ry0:ry1, rx0:rx1]
                    gpu_jobs.append((w_idx, py0, py1, px0, px1, cropped_core))

            # Synchronize wave and copy stitched core features into global map
            for w_idx, py0, py1, px0, px1, cropped_core in gpu_jobs:
                if self._cuda_streams:
                    self._cuda_streams[w_idx].synchronize()
                self.global_features[:, py0:py1, px0:px1] = cropped_core.to(torch.float16).cpu().numpy()

        if isinstance(self.global_features, np.memmap):
            self.global_features.flush()

        self.is_image_set = True

    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        return_viewport_only: bool = False,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int, int, int]],
    ]:
        """
        Standard SAM `.predict()` interface.
        - In `single_tile` mode: runs standard `SamPredictor.predict()`.
        - In `multi_tile` mode: slices the local 64x64 (1024x1024px) feature viewport centered
          around the prompt, runs the lightweight SAM MaskDecoder in O(1) VRAM, and maps the
          resulting mask back to global image coordinates (or returns the 1024x1024 viewport crop
          + `(y0, x0, y1, x1)` offset if `return_viewport_only=True`).
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        if self.active_mode == "single_tile":
            return super().predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                mask_input=mask_input,
                multimask_output=multimask_output,
                return_logits=return_logits,
            )

        return self._predict_multi_tile_viewport(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            mask_input=mask_input,
            multimask_output=multimask_output,
            return_logits=return_logits,
            return_viewport_only=return_viewport_only,
        )

    @torch.no_grad()
    def _predict_multi_tile_viewport(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        return_viewport_only: bool = False,
    ):
        """
        Extracts a 64x64 patch (1024x1024 px) feature window centered on the prompt's centroid
        from `self.global_features` and decodes masks with O(1) VRAM.
        """
        total_py, total_px = self.grid_patches_hw
        orig_h, orig_w = self.original_size

        # 1. Determine prompt centroid in global pixel coordinates (cx, cy)
        pts_list = []
        if point_coords is not None:
            pts_list.append(point_coords.reshape(-1, 2))
        if box is not None:
            b_pts = box.reshape(-1, 2)
            pts_list.append(b_pts)
        if not pts_list:
            raise ValueError("In multi_tile mode, at least `point_coords` or `box` must be provided to locate the viewport.")

        all_pts = np.concatenate(pts_list, axis=0)
        cx_px, cy_px = all_pts.mean(axis=0)

        # 2. Snap viewport origin (vp_py0, vp_px0) to the 16x16 patch grid so features align exactly
        center_py = int(round(cy_px / 16.0))
        center_px = int(round(cx_px / 16.0))
        vp_py0 = max(0, min(center_py - 32, total_py - 64))
        vp_px0 = max(0, min(center_px - 32, total_px - 64))

        vp_y0 = vp_py0 * 16
        vp_x0 = vp_px0 * 16
        vp_y1 = min(vp_y0 + 1024, orig_h)
        vp_x1 = min(vp_x0 + 1024, orig_w)

        # 3. Slice only the (1, 256, 64, 64) window into GPU VRAM
        vp_feat_np = np.array(
            self.global_features[:, vp_py0 : vp_py0 + 64, vp_px0 : vp_px0 + 64],
            dtype=np.float32,
            copy=True,
        )
        vp_features = torch.from_numpy(vp_feat_np).unsqueeze(0).to(self.device)

        # 4. Translate prompts into local [0, 1024] viewport pixel coordinates
        coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None
        if point_coords is not None:
            assert point_labels is not None, "point_labels must be supplied if point_coords is supplied."
            local_pts = point_coords.copy().astype(np.float32)
            local_pts[..., 0] = np.clip(local_pts[..., 0] - vp_x0, 0.0, 1024.0)
            local_pts[..., 1] = np.clip(local_pts[..., 1] - vp_y0, 0.0, 1024.0)
            coords_torch = torch.as_tensor(local_pts, dtype=torch.float, device=self.device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            if coords_torch.ndim == 2:
                coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]

        if box is not None:
            local_box = box.copy().astype(np.float32)
            local_box[..., 0::2] = np.clip(local_box[..., 0::2] - vp_x0, 0.0, 1024.0)
            local_box[..., 1::2] = np.clip(local_box[..., 1::2] - vp_y0, 0.0, 1024.0)
            box_torch = torch.as_tensor(local_box, dtype=torch.float, device=self.device)
            if box_torch.ndim == 1:
                box_torch = box_torch[None, :]

        if mask_input is not None:
            mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float, device=self.device)
            if mask_input_torch.ndim == 3:
                mask_input_torch = mask_input_torch[None, :, :, :]

        # 5. Run frozen SAM PromptEncoder & MaskDecoder on the 64x64 viewport
        points = (coords_torch, labels_torch) if coords_torch is not None else None
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points,
            boxes=box_torch,
            masks=mask_input_torch,
        )
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=vp_features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        # Upscale logits from 256x256 to 1024x1024 viewport pixels
        vp_masks = F.interpolate(
            low_res_masks,
            size=(1024, 1024),
            mode="bilinear",
            align_corners=False,
        )
        valid_h = vp_y1 - vp_y0
        valid_w = vp_x1 - vp_x0
        vp_masks = vp_masks[..., :valid_h, :valid_w]

        if not return_logits:
            vp_masks = vp_masks > self.model.mask_threshold

        vp_masks_np = vp_masks[0].cpu().numpy()
        ious_np = iou_predictions[0].cpu().numpy()
        low_res_np = low_res_masks[0].cpu().numpy()

        # For extreme gigapixel scans where allocating a full (C, H, W) mask array in RAM is unwanted
        if return_viewport_only:
            return vp_masks_np, ious_np, low_res_np, (vp_y0, vp_x0, vp_y1, vp_x1)

        # Place local viewport mask into full-canvas coordinates (C, orig_h, orig_w)
        num_masks = vp_masks_np.shape[0]
        full_dtype = np.float32 if return_logits else bool
        full_masks = np.zeros((num_masks, orig_h, orig_w), dtype=full_dtype)
        full_masks[:, vp_y0:vp_y1, vp_x0:vp_x1] = vp_masks_np

        return full_masks, ious_np, low_res_np

    def reset_image(self) -> None:
        super().reset_image()
        self.global_features = None
        self.padded_hw = None
        self.grid_patches_hw = None
        if self._mmap_path is not None and self._mmap_path.exists():
            try:
                self._mmap_path.unlink()
            except OSError:
                pass
            self._mmap_path = None