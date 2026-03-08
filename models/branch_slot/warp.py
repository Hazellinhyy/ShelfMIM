# -*- coding: utf-8 -*-
"""
Warp slot masks using homography.

AMP-safe version:
All homography matrix operations are forced to float32.
"""

from __future__ import annotations
from typing import Tuple, Union

import torch
import torch.nn.functional as F


def safe_inverse_homography(H: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    H: (B,3,3)
    returns inv(H) with fallback to identity if near-singular.

    AMP-safe: compute inverse in float32.
    """
    if H.dim() != 3 or H.shape[-2:] != (3, 3):
        raise ValueError(f"H must be (B,3,3), got {H.shape}")

    orig_dtype = H.dtype
    H = H.float()

    B = H.shape[0]
    det = torch.det(H)

    eye = torch.eye(3, device=H.device, dtype=torch.float32).unsqueeze(0).repeat(B, 1, 1)

    good = det.abs() > eps
    H_inv = eye.clone()

    if good.any():
        H_inv[good] = torch.inverse(H[good])

    return H_inv.to(orig_dtype)


def _patch_to_pixel_affine(patch_size: int, device, dtype) -> torch.Tensor:
    ps = float(patch_size)
    c = (ps - 1.0) / 2.0

    return torch.tensor(
        [[ps, 0.0, c],
         [0.0, ps, c],
         [0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )


def homography_pix_to_patch(
    H_src_to_dst_pix: torch.Tensor,
    patch_size: int,
) -> torch.Tensor:
    """
    Convert pixel-space homography to patch-grid-space homography.

    AMP-safe: matrix ops in float32.
    """
    B = H_src_to_dst_pix.shape[0]
    orig_dtype = H_src_to_dst_pix.dtype

    H = H_src_to_dst_pix.float()
    device = H.device

    A = _patch_to_pixel_affine(patch_size, device, torch.float32)
    A_inv = torch.inverse(A)

    A_b = A.unsqueeze(0).repeat(B, 1, 1)
    A_inv_b = A_inv.unsqueeze(0).repeat(B, 1, 1)

    H_patch = A_inv_b @ H @ A_b

    return H_patch.to(orig_dtype)


def _build_perspective_grid(
    H_out_to_in: torch.Tensor,
    out_hw: Tuple[int, int],
    in_hw: Tuple[int, int],
    align_corners: bool = True,
) -> torch.Tensor:
    """
    Create grid for torch.grid_sample.
    """
    B = H_out_to_in.shape[0]
    out_h, out_w = out_hw
    in_h, in_w = in_hw

    device = H_out_to_in.device
    dtype = H_out_to_in.dtype

    ys = torch.linspace(0, out_h - 1, out_h, device=device, dtype=dtype)
    xs = torch.linspace(0, out_w - 1, out_w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    ones = torch.ones_like(xx)
    coords = torch.stack([xx, yy, ones], dim=0).reshape(3, -1)
    coords = coords.unsqueeze(0).repeat(B, 1, 1)

    mapped = torch.bmm(H_out_to_in, coords)
    mapped = torch.nan_to_num(mapped, nan=0.0, posinf=1e6, neginf=-1e6)

    x = mapped[:, 0, :] / mapped[:, 2, :].clamp(min=1e-6)
    y = mapped[:, 1, :] / mapped[:, 2, :].clamp(min=1e-6)

    if align_corners:
        x_norm = 2.0 * x / max(in_w - 1, 1) - 1.0
        y_norm = 2.0 * y / max(in_h - 1, 1) - 1.0
    else:
        x_norm = (2.0 * x + 1.0) / in_w - 1.0
        y_norm = (2.0 * y + 1.0) / in_h - 1.0

    grid = torch.stack([x_norm, y_norm], dim=-1)
    grid = torch.nan_to_num(grid, nan=2.0, posinf=2.0, neginf=-2.0)
    grid = grid.clamp(min=-2.0, max=2.0)
    grid = grid.reshape(B, out_h, out_w, 2)
    return grid


def warp_masks_src_to_dst(
    masks_src: torch.Tensor,
    H_src_to_dst_pix: torch.Tensor,
    src_hw: Tuple[int, int],
    dst_hw: Tuple[int, int],
    patch_size: int = 16,
    mode: str = "bilinear",
    align_corners: bool = True,
    return_valid_mask: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

    if masks_src.dim() == 3:
        B, K, N = masks_src.shape
        Hs, Ws = src_hw
        if N != Hs * Ws:
            raise ValueError(f"N={N} != Hs*Ws={Hs*Ws}")
        masks_src = masks_src.reshape(B, K, Hs, Ws)
    elif masks_src.dim() == 4:
        B, K, Hs, Ws = masks_src.shape
        if (Hs, Ws) != src_hw:
            raise ValueError(f"masks_src hw {Hs,Ws} != src_hw {src_hw}")
    else:
        raise ValueError(f"masks_src must be 3D or 4D, got {masks_src.shape}")

    if H_src_to_dst_pix.shape != (B, 3, 3):
        raise ValueError(f"H_src_to_dst_pix must be (B,3,3), got {H_src_to_dst_pix.shape}")

    H_src_to_dst_patch = homography_pix_to_patch(
        H_src_to_dst_pix,
        patch_size=patch_size,
    )

    H_dst_to_src_patch = safe_inverse_homography(H_src_to_dst_patch)

    Hd, Wd = dst_hw

    grid = _build_perspective_grid(
        H_out_to_in=H_dst_to_src_patch,
        out_hw=(Hd, Wd),
        in_hw=(Hs, Ws),
        align_corners=align_corners,
    )

    gx = grid[..., 0]
    gy = grid[..., 1]
    valid_grid = torch.isfinite(gx) & torch.isfinite(gy)
    valid_grid = valid_grid & (gx >= -1.0) & (gx <= 1.0) & (gy >= -1.0) & (gy <= 1.0)

    x = masks_src.reshape(B * K, 1, Hs, Ws)
    grid_rep = grid.repeat_interleave(K, dim=0)

    warped = F.grid_sample(
        x,
        grid_rep,
        mode=mode,
        padding_mode="zeros",
        align_corners=align_corners,
    )

    warped = warped.reshape(B, K, Hd, Wd)
    warped = warped * valid_grid.unsqueeze(1).to(dtype=warped.dtype)
    warped = torch.nan_to_num(warped, nan=0.0, posinf=0.0, neginf=0.0)

    if return_valid_mask:
        return warped, valid_grid.unsqueeze(1).to(dtype=warped.dtype)
    return warped


def warp_masks_b_to_a(
    masks_b: torch.Tensor,
    H_b2a_pix: torch.Tensor,
    grid_hw: Tuple[int, int],
    patch_size: int = 16,
    return_valid_mask: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    return warp_masks_src_to_dst(
        masks_src=masks_b,
        H_src_to_dst_pix=H_b2a_pix,
        src_hw=grid_hw,
        dst_hw=grid_hw,
        patch_size=patch_size,
        return_valid_mask=return_valid_mask,
    )


def warp_masks_a_to_b(
    masks_a: torch.Tensor,
    H_b2a_pix: torch.Tensor,
    grid_hw: Tuple[int, int],
    patch_size: int = 16,
    return_valid_mask: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    H_a2b_pix = safe_inverse_homography(H_b2a_pix)
    return warp_masks_src_to_dst(
        masks_src=masks_a,
        H_src_to_dst_pix=H_a2b_pix,
        src_hw=grid_hw,
        dst_hw=grid_hw,
        patch_size=patch_size,
        return_valid_mask=return_valid_mask,
    )


@torch.no_grad()
def warp_quality_metrics(
    H_src_to_dst_pix: torch.Tensor,
    grid_hw: Tuple[int, int],
    patch_size: int = 16,
    det_eps: float = 1e-8,
) -> dict:
    """Return lightweight diagnostics for homography reliability."""
    if H_src_to_dst_pix.dim() != 3 or H_src_to_dst_pix.shape[-2:] != (3, 3):
        raise ValueError(f"H_src_to_dst_pix must be (B,3,3), got {H_src_to_dst_pix.shape}")

    H = H_src_to_dst_pix.float()
    det = torch.det(H)
    det_small_ratio = (det.abs() <= float(det_eps)).float().mean()

    H_patch = homography_pix_to_patch(H_src_to_dst_pix, patch_size=patch_size).float()
    H_inv = safe_inverse_homography(H_patch).float()
    Hd, Wd = grid_hw
    grid = _build_perspective_grid(H_out_to_in=H_inv, out_hw=(Hd, Wd), in_hw=grid_hw, align_corners=True)

    grid_invalid = ~torch.isfinite(grid)
    grid_invalid_ratio = grid_invalid.float().mean()
    x, y = grid[..., 0], grid[..., 1]
    oob = (x < -1.0) | (x > 1.0) | (y < -1.0) | (y > 1.0)
    grid_oob_ratio = oob.float().mean()

    return {
        "homography_det_small_ratio": det_small_ratio,
        "warp_invalid_ratio": grid_invalid_ratio,
        "grid_oob_ratio": grid_oob_ratio,
    }
