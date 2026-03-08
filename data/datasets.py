# -*- coding: utf-8 -*-
"""
Datasets for ShelfMIM pretraining.
"""

from __future__ import annotations

import os
from typing import Callable, List, Optional, Sequence, Set

from PIL import Image
from torch.utils.data import Dataset


_DEFAULT_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _norm_exts(exts: Sequence[str]) -> Set[str]:
    return {e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts}


def _is_image_file(path_or_name: str, exts: Set[str]) -> bool:
    _, ext = os.path.splitext(path_or_name)
    return ext.lower() in exts


def _walk_images(
    dir_path: str,
    recursive: bool,
    exts: Set[str],
    skip_dir_keywords: Sequence[str] = (),
) -> List[str]:
    out: List[str] = []
    if not os.path.isdir(dir_path):
        return out

    if recursive:
        for root, dirs, files in os.walk(dir_path):
            pruned = []
            for d in dirs:
                if d.startswith("."):
                    continue
                dl = d.lower()
                if any(k in dl for k in skip_dir_keywords):
                    continue
                pruned.append(d)
            dirs[:] = pruned

            for fn in files:
                if _is_image_file(fn, exts):
                    out.append(os.path.join(root, fn))
    else:
        for fn in os.listdir(dir_path):
            p = os.path.join(dir_path, fn)
            if os.path.isfile(p) and _is_image_file(fn, exts):
                out.append(p)

    out.sort()
    return out


def _rpc_split_to_folder(split: str) -> str:
    s = (split or "train").strip().lower()
    if s in {"train", "train2019"}:
        return "train2019"
    if s in {"val", "valid", "validation", "val2019"}:
        return "val2019"
    if s in {"test", "test2019"}:
        return "test2019"
    return split


class RPCSSL(Dataset):
    """RPC SSL dataset (image-only)."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        recursive: bool = True,
        extensions: Sequence[str] = tuple(_DEFAULT_EXTS),
    ):
        super().__init__()
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        self.recursive = bool(recursive)
        self.exts = _norm_exts(extensions)

        split_folder = _rpc_split_to_folder(split)
        candidates = [
            os.path.join(self.root, split_folder),
            os.path.join(self.root, "images", split_folder),
        ]

        img_dir = None
        for c in candidates:
            if os.path.isdir(c):
                img_dir = c
                break

        if img_dir is None:
            raise FileNotFoundError(
                f"[RPCSSL] image folder not found. root={self.root!r}, split={split!r}\n"
                f"Tried: {candidates}\n"
                "Please point cfg.data.root to retail_product_checkout containing train2019/val2019/test2019."
            )

        self.img_dir = img_dir
        self.samples = _walk_images(self.img_dir, recursive=self.recursive, exts=self.exts)
        if len(self.samples) == 0:
            raise RuntimeError(
                f"[RPCSSL] found 0 images under {self.img_dir}. Check dataset path and image extensions."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path = self.samples[idx]
        with Image.open(path) as im:
            img = im.convert("RGB")

        if self.transform is None:
            return {"img": img, "path": path, "index": idx}

        out = self.transform(img)
        if not isinstance(out, dict):
            raise TypeError(f"transform must return dict, got {type(out)}")

        out["path"] = path
        out["index"] = idx
        return out


class D2SSSL(Dataset):
    """D2S / MVTec D2S SSL dataset (image-only)."""

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        split: Optional[str] = None,
        recursive: bool = True,
        extensions: Sequence[str] = tuple(_DEFAULT_EXTS),
    ):
        super().__init__()
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.split = split
        self.recursive = bool(recursive)
        self.exts = _norm_exts(extensions)

        cand_dirs: List[str] = []
        if split is not None and str(split).strip() != "":
            s = str(split).strip()
            cand_dirs += [
                os.path.join(self.root, s),
                os.path.join(self.root, f"{s}2019"),
                os.path.join(self.root, "images", s),
                os.path.join(self.root, "images", f"{s}2019"),
            ]

        cand_dirs += [
            os.path.join(self.root, "images"),
            os.path.join(self.root, "imgs"),
            self.root,
        ]

        skip_kw = [
            "annotation", "annotations", "mask", "masks", "label", "labels",
            "gt", "ground_truth", "seg", "segmentation", "instance", "instances",
        ]

        chosen_dir = None
        samples: List[str] = []
        for d in cand_dirs:
            if not os.path.isdir(d):
                continue
            imgs = _walk_images(d, recursive=self.recursive, exts=self.exts, skip_dir_keywords=skip_kw)
            if len(imgs) > 0:
                chosen_dir = d
                samples = imgs
                break

        if chosen_dir is None:
            raise FileNotFoundError(
                f"[D2SSSL] image folder not found under root={self.root!r}.\n"
                f"Tried: {cand_dirs}\n"
                "Please point cfg.data.root to a directory containing images/ or image files."
            )

        self.img_dir = chosen_dir
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path = self.samples[idx]
        with Image.open(path) as im:
            img = im.convert("RGB")

        if self.transform is None:
            return {"img": img, "path": path, "index": idx}

        out = self.transform(img)
        if not isinstance(out, dict):
            raise TypeError(f"transform must return dict, got {type(out)}")

        out["path"] = path
        out["index"] = idx
        return out
