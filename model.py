#!/usr/bin/env python3
"""
Gr-SAM-B: 2D Bidirectional Spatial Grassmann Backbone for Segment Anything.
Matches SAM ViT-B's exact input/output contract:
  Input:  (B, 3, 1024, 1024) normalized image tensor
  Output: (B, 256, 64, 64) spatial feature embedding
Supports warm-starting `patch_embed`, `pos_embed`, and `neck` directly from Meta's Apache 2.0 ViT-B checkpoint.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class LayerNorm2d(nn.Module):
    """Exact clone of SAM's 2D LayerNorm used in the Neck."""
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class SpatialGrassmannBlock2D(nn.Module):
    """
    2D Multi-Scale Plücker-Coordinate Grassmann Mixing Block.
    Replaces O(N^2) self-attention over 64x64=4096 tokens with O(N) exterior (wedge) products
    on the Grassmann manifold Gr(2, r).
    """
    def __init__(self, d_model: int = 768, rank: int = 16, scales=(1, 2, 4, 8, 16, 32), mlp_ratio: float = 4.0):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.plucker_dim = (rank * (rank - 1)) // 2  # 120 for rank=16
        self.max_scale = max(scales)

        # 4 directional axes (Horizontal, Vertical, Main-Diag, Anti-Diag) across dilation scales
        axes = [(0, 1), (1, 0), (1, 1), (1, -1)]
        self.offsets = [(dy * s, dx * s) for s in scales for dy, dx in axes]
        self.num_offsets = len(self.offsets)

        # Precompute strictly upper-triangular indices (i < j) to compute wedge product without r x r outer matrix
        idx_i, idx_j = torch.triu_indices(rank, rank, offset=1)
        self.register_buffer("idx_i", idx_i, persistent=False)
        self.register_buffer("idx_j", idx_j, persistent=False)

        # 1. Subspace reduction
        self.norm1 = nn.LayerNorm(d_model)
        self.w_red = nn.Linear(d_model, rank, bias=False)

        # 2. Learnable scale-and-direction weighting + Plücker projection back to d_model
        self.offset_weights = nn.Parameter(torch.ones(self.num_offsets, self.plucker_dim) / self.num_offsets)
        self.w_plucker = nn.Linear(self.plucker_dim, d_model)

        # 3. Gated mixing & Channel MLP
        self.gate_proj = nn.Linear(d_model * 2, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        hidden_dim = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

    def _grassmann_mix(self, x_norm: torch.Tensor) -> torch.Tensor:
        # Project to low-rank subspace: (B, H, W, r)
        z = self.w_red(x_norm)
        B, H, W, _ = z.shape

        # Gather z_i and z_j once: (B, H, W, 120)
        z_i = z[..., self.idx_i]
        z_j = z[..., self.idx_j]

        # Zero-pad spatial dimensions by max_scale to avoid wrap-around boundary artifacts
        pad = self.max_scale
        z_padded = F.pad(z, (0, 0, pad, pad, pad, pad))

        plucker_accum = torch.zeros_like(z_i)
        for k, (dy, dx) in enumerate(self.offsets):
            z_shift = z_padded[:, pad + dy : pad + dy + H, pad + dx : pad + dx + W, :]
            # Exterior wedge product: p_{ij} = z_i * z'_j - z_j * z'_i
            p_k = z_i * z_shift[..., self.idx_j] - z_j * z_shift[..., self.idx_i]
            # L2 normalize onto Plücker sphere of Gr(2, r)
            p_k = F.normalize(p_k, p=2, dim=-1, eps=1e-6)
            plucker_accum = plucker_accum + p_k * self.offset_weights[k]

        g_feat = self.w_plucker(plucker_accum)
        gate = torch.sigmoid(self.gate_proj(torch.cat([x_norm, g_feat], dim=-1)))
        return gate * g_feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self._grassmann_mix(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed2D(nn.Module):
    """Matches SAM ViT-B patch_embed structure so weights copy cleanly."""
    def __init__(self, img_size: int = 1024, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, H//16, W//16, D)
        return self.proj(x).permute(0, 2, 3, 1)


class GrassmannSAMEncoder(nn.Module):
    """
    Drop-in replacement for SAM's `image_encoder` (ViT-B).
    """
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        rank: int = 16,
        out_chans: int = 256,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.use_checkpoint = use_checkpoint
        grid_size = img_size // patch_size  # 64

        # 1. Stem (Matches SAM ViT-B naming for weight transfer)
        self.patch_embed = PatchEmbed2D(img_size, patch_size, in_chans, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, grid_size, grid_size, embed_dim))

        # 2. Grassmann Backbone Blocks
        self.blocks = nn.ModuleList([
            SpatialGrassmannBlock2D(d_model=embed_dim, rank=rank)
            for _ in range(depth)
        ])

        # 3. Neck (Matches SAM ViT-B neck naming for weight transfer)
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False),
            LayerNorm2d(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_chans),
        )

    def load_sam_stem_and_neck(self, sam_ckpt_path: str, freeze_stem_neck: bool = False):
        """Copies pretrained patch_embed, pos_embed, and neck weights from SAM ViT-B."""
        state_dict = torch.load(sam_ckpt_path, map_location="cpu", weights_only=True)
        own_state = self.state_dict()
        copied_keys = []

        for k, v in state_dict.items():
            if k.startswith("image_encoder."):
                short_k = k[len("image_encoder."):]
                if short_k.startswith(("patch_embed.", "pos_embed", "neck.")) and short_k in own_state:
                    if own_state[short_k].shape == v.shape:
                        own_state[short_k].copy_(v)
                        copied_keys.append(short_k)

        print(f"[✓] Warm-started {len(copied_keys)} tensors from SAM ViT-B ({copied_keys})")
        if freeze_stem_neck:
            self.set_stem_neck_trainable(False)

    def set_stem_neck_trainable(self, trainable: bool = True):
        for name, param in self.named_parameters():
            if name.startswith(("patch_embed.", "pos_embed", "neck.")):
                param.requires_grad = trainable
        status = "UNFROZEN" if trainable else "FROZEN"
        print(f"[*] Stem and Neck parameters are now {status}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x + self.pos_embed

        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)

        # Permute (B, 64, 64, D) -> (B, D, 64, 64) for Neck
        x = self.neck(x.permute(0, 3, 1, 2))
        return x
