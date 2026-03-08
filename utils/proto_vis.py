# -*- coding: utf-8 -*-
"""
Cluster-faithful prototype visualization (2x2 PCA), paper-style layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import os
import numpy as np
import torch
import matplotlib.pyplot as plt


def _l2n(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def _to_np(x: torch.Tensor) -> np.ndarray:
    return torch.nan_to_num(x.detach().float(), nan=0.0, posinf=0.0, neginf=0.0).cpu().numpy()


def pca2_svd(X: np.ndarray) -> Tuple[np.ndarray, float, float]:
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 2:
        z = np.zeros((max(int(X.shape[0]), 1), 2), dtype=np.float32)
        return z, 0.0, 0.0

    X = X.astype(np.float32)
    X = X - X.mean(axis=0, keepdims=True)
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    X2 = X @ Vt[:2].T

    var = (S ** 2) / (X.shape[0] - 1 + 1e-9)
    total = var.sum() + 1e-9
    evr1 = float(var[0] / total) if var.size > 0 else 0.0
    evr2 = float(var[1] / total) if var.size > 1 else 0.0
    return X2, evr1, evr2


def _topM_prototypes_by_usage(q: Optional[torch.Tensor], M: int, P: int, device: torch.device) -> torch.Tensor:
    if q is None:
        return torch.arange(min(int(M), int(P)), device=device)
    if q.dim() != 2:
        q = q.reshape(-1, q.size(-1))
    usage = q.mean(dim=0)
    M = min(int(M), int(usage.numel()))
    return torch.topk(usage, k=M, largest=True).indices


def _select_points_per_proto(
    emb: torch.Tensor,
    q: torch.Tensor,
    top_ids: torch.Tensor,
    topK_per_proto: int = 200,
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    if q.dim() != 2:
        q = q.reshape(-1, q.size(-1))

    N = emb.shape[0]
    topK = int(topK_per_proto)
    emb_list = []
    grp_mod10 = []
    grp_rank = []

    q_det = q.detach()
    for rank_idx, pid_t in enumerate(top_ids):
        pid = int(pid_t.item())
        scores = q_det[:, pid]
        k = min(topK, N)
        idx = torch.topk(scores, k=k, largest=True).indices
        emb_list.append(emb[idx])
        grp_mod10.append(np.full((k,), pid % 10, dtype=np.int32))
        grp_rank.append(np.full((k,), rank_idx, dtype=np.int32))

    emb_sel = torch.cat(emb_list, dim=0) if emb_list else emb[:0]
    mod10 = np.concatenate(grp_mod10, axis=0) if grp_mod10 else np.zeros((0,), dtype=np.int32)
    rank = np.concatenate(grp_rank, axis=0) if grp_rank else np.zeros((0,), dtype=np.int32)
    return emb_sel, mod10, rank


def _pca_embed_with_prototypes(emb: torch.Tensor, proto: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, float, float]:
    emb = _l2n(emb)
    proto = _l2n(proto)
    X = np.concatenate([_to_np(emb), _to_np(proto)], axis=0)
    X2, evr1, evr2 = pca2_svd(X)
    N = emb.shape[0]
    return X2[:N], X2[N:], evr1, evr2


def _shared_xy_limits(*arrays: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    pts = [a for a in arrays if a is not None and a.size > 0]
    if not pts:
        return (-5.0, 5.0), (-5.0, 5.0)
    allp = np.concatenate(pts, axis=0)
    xmin, ymin = allp.min(axis=0)
    xmax, ymax = allp.max(axis=0)
    dx = max(1e-6, float(xmax - xmin))
    dy = max(1e-6, float(ymax - ymin))
    px = 0.10 * dx
    py = 0.10 * dy
    return (float(xmin - px), float(xmax + px)), (float(ymin - py), float(ymax + py))


def plot_proto_pca_2x2(
    save_path: str,
    u_a: torch.Tensor,
    u_b: torch.Tensor,
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    proto_g: torch.Tensor,
    proto_p: torch.Tensor,
    qg_a: Optional[torch.Tensor] = None,
    qg_b: Optional[torch.Tensor] = None,
    qp_a: Optional[torch.Tensor] = None,
    qp_b: Optional[torch.Tensor] = None,
    topM_global: int = 100,
    topM_part: int = 60,
    topK_per_proto_global: int = 80,
    topK_per_proto_part: int = 120,
    title: str = "ShelfMIM Prototype-Alignment PCA (2x2)",
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if qg_a is None or qg_b is None or qp_a is None or qp_b is None:
        raise ValueError("This cluster view requires qg_a/qg_b/qp_a/qp_b assignments.")

    idx_g = _topM_prototypes_by_usage(0.5 * (qg_a + qg_b), topM_global, proto_g.size(0), proto_g.device)
    idx_p = _topM_prototypes_by_usage(0.5 * (qp_a + qp_b), topM_part, proto_p.size(0), proto_p.device)

    pg = proto_g[idx_g]
    pp = proto_p[idx_p]

    ug_a, cg_a_mod10, cg_a_rank = _select_points_per_proto(u_a, qg_a, idx_g, topK_per_proto_global)
    ug_b, cg_b_mod10, cg_b_rank = _select_points_per_proto(u_b, qg_b, idx_g, topK_per_proto_global)
    zp_a, cp_a_mod10, cp_a_rank = _select_points_per_proto(z_a, qp_a, idx_p, topK_per_proto_part)
    zp_b, cp_b_mod10, cp_b_rank = _select_points_per_proto(z_b, qp_b, idx_p, topK_per_proto_part)

    g_a2, g_p2, g1, g2 = _pca_embed_with_prototypes(ug_a, pg)
    g_b2, _, _, _ = _pca_embed_with_prototypes(ug_b, pg)
    p_a2, p_p2, p1, p2 = _pca_embed_with_prototypes(zp_a, pp)
    p_b2, _, _, _ = _pca_embed_with_prototypes(zp_b, pp)

    gxlim, gylim = _shared_xy_limits(g_a2, g_b2, g_p2)
    pxlim, pylim = _shared_xy_limits(p_a2, p_b2, p_p2)

    Mg = int(idx_g.numel())
    Mp = int(idx_p.numel())

    plt.rcParams.update({
        "axes.facecolor": "#f2f2f2",
        "figure.facecolor": "#f2f2f2",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.color": "#c8c8c8",
    })

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 11), constrained_layout=True)
    fig.suptitle(
        f"{title} - Final layout\\n"
        f"Global PCA: PC1={g1*100:.2f}%, PC2={g2*100:.2f}% | Part PCA: PC1={p1*100:.2f}%, PC2={p2*100:.2f}%\n"
        f"Dots = embeddings; Stars = Top-M representative prototypes (for readability).",
        fontsize=18,
        fontweight="bold",
    )

    def _panel(ax, pts2, grp_mod10, prot2, subtitle, xlim, ylim):
        sc = ax.scatter(
            pts2[:, 0],
            pts2[:, 1],
            s=14,
            alpha=0.55,
            c=grp_mod10,
            cmap="viridis",
            vmin=0,
            vmax=9,
            linewidths=0.0,
        )
        ax.scatter(
            prot2[:, 0],
            prot2[:, 1],
            marker="*",
            s=280,
            c="#20b2aa",
            edgecolors="k",
            linewidths=0.8,
            zorder=5,
        )
        ax.set_title(subtitle, fontsize=18, fontweight="bold")
        ax.set_xlabel("PC1", fontsize=14)
        ax.set_ylabel("PC2", fontsize=14)
        ax.tick_params(labelsize=12)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        return sc

    sc0 = _panel(axes[0, 0], g_a2, cg_a_mod10, g_p2, f"Global (image-level) | View A  |  Top-M={Mg}", gxlim, gylim)
    _panel(axes[0, 1], g_b2, cg_b_mod10, g_p2, f"Global (image-level) | View B  |  Top-M={Mg}", gxlim, gylim)
    _panel(axes[1, 0], p_a2, cp_a_mod10, p_p2, f"Part (part-level) | View A  |  Top-M={Mp}", pxlim, pylim)
    _panel(axes[1, 1], p_b2, cp_b_mod10, p_p2, f"Part (part-level) | View B  |  Top-M={Mp}", pxlim, pylim)

    cbar = fig.colorbar(sc0, ax=axes.ravel().tolist(), fraction=0.026, pad=0.02)
    cbar.set_label("Group (prototype id mod 10)", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    plt.savefig(save_path, dpi=280, bbox_inches="tight")
    plt.close(fig)


@dataclass
class ProtoVisBuffer:
    max_u: int = 2000
    max_z: int = 5000
    device: str = "cpu"

    u_a: Optional[torch.Tensor] = None
    u_b: Optional[torch.Tensor] = None
    qg_a: Optional[torch.Tensor] = None
    qg_b: Optional[torch.Tensor] = None
    z_a: Optional[torch.Tensor] = None
    z_b: Optional[torch.Tensor] = None
    qp_a: Optional[torch.Tensor] = None
    qp_b: Optional[torch.Tensor] = None

    def _append(self, buf: Optional[torch.Tensor], x: torch.Tensor, max_n: int) -> torch.Tensor:
        x = x.detach()
        if self.device == "cpu":
            x = x.cpu()
        buf = x if buf is None else torch.cat([buf, x], dim=0)
        if buf.size(0) > max_n:
            buf = buf[-max_n:]
        return buf

    def push_global(self, u_a: torch.Tensor, u_b: torch.Tensor, qg_a: Optional[torch.Tensor], qg_b: Optional[torch.Tensor]):
        self.u_a = self._append(self.u_a, u_a, self.max_u)
        self.u_b = self._append(self.u_b, u_b, self.max_u)
        if qg_a is not None:
            self.qg_a = self._append(self.qg_a, qg_a, self.max_u)
        if qg_b is not None:
            self.qg_b = self._append(self.qg_b, qg_b, self.max_u)

    def push_part(self, z_a: torch.Tensor, z_b: torch.Tensor, qp_a: Optional[torch.Tensor], qp_b: Optional[torch.Tensor]):
        self.z_a = self._append(self.z_a, z_a, self.max_z)
        self.z_b = self._append(self.z_b, z_b, self.max_z)
        if qp_a is not None:
            self.qp_a = self._append(self.qp_a, qp_a, self.max_z)
        if qp_b is not None:
            self.qp_b = self._append(self.qp_b, qp_b, self.max_z)

    def ready(self) -> bool:
        return (
            self.u_a is not None and self.u_b is not None and
            self.z_a is not None and self.z_b is not None and
            self.u_a.size(0) >= min(32, self.max_u) and
            self.z_a.size(0) >= min(128, self.max_z)
        )

    def get(self) -> Dict[str, torch.Tensor]:
        return {
            "u_a": self.u_a,
            "u_b": self.u_b,
            "qg_a": self.qg_a,
            "qg_b": self.qg_b,
            "z_a": self.z_a,
            "z_b": self.z_b,
            "qp_a": self.qp_a,
            "qp_b": self.qp_b,
        }

