# -*- coding: utf-8 -*-
"""
Transforms for SSL pretraining (two-view) that strictly match paper preprocessing:
- Resize keep aspect: short side -> 768; if long side > 1536 cap to 1536 (bilinear)
- Zero pad ONLY on right and bottom so H,W divisible by 16
- View A: appearance perturbations (no geometry by default)
- View B: geometry perturbations (crop/affine/perspective/flip) and return homography H_B2A
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any

import math
import numpy as np
import cv2
from PIL import Image, ImageFilter

import torch
import torchvision.transforms.functional as TF


# -----------------------------
# Basic preprocessing: resize + pad
# -----------------------------

@dataclass
class ResizeKeepARShortLong:
    short_side: int = 768
    long_side_max: int = 1536
    interpolation: int = Image.BILINEAR

    def __call__(self, img: Image.Image) -> Tuple[Image.Image, Dict[str, Any]]:
        w, h = img.size
        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid image size: {(w, h)}")

        # scale to make short side = short_side
        scale = self.short_side / float(min(h, w))
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))

        # cap long side to long_side_max if needed
        if max(new_h, new_w) > self.long_side_max:
            scale = self.long_side_max / float(max(h, w))
            new_h = int(round(h * scale))
            new_w = int(round(w * scale))

        resized = img.resize((new_w, new_h), resample=self.interpolation)
        meta = {
            "orig_size_hw": (h, w),
            "resized_size_hw": (new_h, new_w),
            "scale": scale,
        }
        return resized, meta


@dataclass
class PadRightBottomToDivisor:
    divisor: int = 16
    pad_value: int = 0  # zero padding

    def __call__(self, img: Image.Image) -> Tuple[Image.Image, Dict[str, Any]]:
        w, h = img.size
        pad_w = (self.divisor - (w % self.divisor)) % self.divisor
        pad_h = (self.divisor - (h % self.divisor)) % self.divisor

        if pad_w == 0 and pad_h == 0:
            meta = {"pad_right": 0, "pad_bottom": 0, "padded_size_hw": (h, w)}
            return img, meta

        new_w = w + pad_w
        new_h = h + pad_h

        canvas = Image.new(img.mode, (new_w, new_h), color=(self.pad_value,) * (3 if img.mode == "RGB" else 1))
        canvas.paste(img, (0, 0))
        meta = {"pad_right": pad_w, "pad_bottom": pad_h, "padded_size_hw": (new_h, new_w)}
        return canvas, meta


# -----------------------------
# Appearance (View A)
# -----------------------------

@dataclass
class AppearanceAugment:
    # Color jitter params
    brightness: float = 0.4
    contrast: float = 0.4
    saturation: float = 0.4
    hue: float = 0.1
    jitter_prob: float = 1.0

    grayscale_prob: float = 0.2
    blur_prob: float = 0.5
    blur_sigma: Tuple[float, float] = (0.1, 2.0)

    # Optional horizontal flip for view A (default off to keep it "appearance only")
    hflip_prob: float = 0.0

    def __call__(self, img: Image.Image) -> Tuple[Image.Image, np.ndarray]:
        """
        Returns:
          img_aug: PIL RGB
          H_A: homography mapping from original (A pre-aug) coords -> A coords
               (only non-identity if hflip_prob > 0)
        """
        if img.mode != "RGB":
            img = img.convert("RGB")

        w, h = img.size
        H = np.eye(3, dtype=np.float32)

        # Optional hflip
        if self.hflip_prob > 0 and torch.rand(()) < self.hflip_prob:
            img = TF.hflip(img)
            # x' = (w-1) - x
            H_flip = np.array([[-1.0, 0.0, float(w - 1)],
                               [0.0,  1.0, 0.0],
                               [0.0,  0.0, 1.0]], dtype=np.float32)
            H = H_flip @ H

        # Color jitter
        if self.jitter_prob > 0 and torch.rand(()) < self.jitter_prob:
            # random factors
            b = float(torch.empty(1).uniform_(max(0.0, 1 - self.brightness), 1 + self.brightness))
            c = float(torch.empty(1).uniform_(max(0.0, 1 - self.contrast),   1 + self.contrast))
            s = float(torch.empty(1).uniform_(max(0.0, 1 - self.saturation), 1 + self.saturation))
            h_delta = float(torch.empty(1).uniform_(-self.hue, self.hue))
            img = TF.adjust_brightness(img, b)
            img = TF.adjust_contrast(img, c)
            img = TF.adjust_saturation(img, s)
            img = TF.adjust_hue(img, h_delta)

        # Grayscale
        if self.grayscale_prob > 0 and torch.rand(()) < self.grayscale_prob:
            img = TF.rgb_to_grayscale(img, num_output_channels=3)

        # Gaussian blur
        if self.blur_prob > 0 and torch.rand(()) < self.blur_prob:
            sigma = float(torch.empty(1).uniform_(self.blur_sigma[0], self.blur_sigma[1]))
            img = img.filter(ImageFilter.GaussianBlur(radius=sigma))

        return img, H


# -----------------------------
# Geometry (View B) with homography
# -----------------------------

@dataclass
class RandomResizedCropSameSize:
    scale: Tuple[float, float] = (0.5, 1.0)
    ratio: Tuple[float, float] = (0.75, 1.333)

    def sample_crop(self, h: int, w: int) -> Tuple[int, int, int, int]:
        area = float(h * w)
        log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))

        for _ in range(10):
            target_area = float(torch.empty(1).uniform_(self.scale[0], self.scale[1])) * area
            aspect = math.exp(float(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])))

            crop_w = int(round(math.sqrt(target_area * aspect)))
            crop_h = int(round(math.sqrt(target_area / aspect)))

            if 0 < crop_w <= w and 0 < crop_h <= h:
                top = int(torch.randint(0, h - crop_h + 1, (1,)).item())
                left = int(torch.randint(0, w - crop_w + 1, (1,)).item())
                return top, left, crop_h, crop_w

        # fallback: center crop
        in_ratio = w / h
        if in_ratio < self.ratio[0]:
            crop_w = w
            crop_h = int(round(w / self.ratio[0]))
        elif in_ratio > self.ratio[1]:
            crop_h = h
            crop_w = int(round(h * self.ratio[1]))
        else:
            crop_h = h
            crop_w = w

        top = (h - crop_h) // 2
        left = (w - crop_w) // 2
        return top, left, crop_h, crop_w

    def __call__(self, img_rgb: np.ndarray, H_total: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crop then resize back to original size (same H,W). Update H_total (A->B).
        """
        h, w = img_rgb.shape[:2]
        out_h, out_w = h, w

        top, left, ch, cw = self.sample_crop(h, w)

        # crop
        cropped = img_rgb[top:top + ch, left:left + cw, :]

        # resize back
        resized = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

        sx = out_w / float(cw)
        sy = out_h / float(ch)

        # x' = (x - left) * sx
        # y' = (y - top)  * sy
        H_crop = np.array([[sx, 0.0, -left * sx],
                           [0.0, sy, -top * sy],
                           [0.0, 0.0, 1.0]], dtype=np.float32)

        return resized, H_crop @ H_total


@dataclass
class RandomAffineSameSize:
    degrees: float = 10.0
    translate: Tuple[float, float] = (0.1, 0.1)  # fraction of w/h
    scale: Tuple[float, float] = (0.9, 1.1)
    prob: float = 0.7

    def __call__(self, img_rgb: np.ndarray, H_total: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.prob > 0 and torch.rand(()) >= self.prob:
            return img_rgb, H_total

        h, w = img_rgb.shape[:2]
        angle = float(torch.empty(1).uniform_(-self.degrees, self.degrees))
        sc = float(torch.empty(1).uniform_(self.scale[0], self.scale[1]))
        max_dx = self.translate[0] * w
        max_dy = self.translate[1] * h
        tx = float(torch.empty(1).uniform_(-max_dx, max_dx))
        ty = float(torch.empty(1).uniform_(-max_dy, max_dy))

        center = (w / 2.0, h / 2.0)
        M2 = cv2.getRotationMatrix2D(center, angle, sc)  # 2x3
        M2[0, 2] += tx
        M2[1, 2] += ty

        warped = cv2.warpAffine(
            img_rgb, M2, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        H_aff = np.eye(3, dtype=np.float32)
        H_aff[:2, :] = M2.astype(np.float32)
        return warped, H_aff @ H_total


@dataclass
class RandomPerspectiveSameSize:
    distortion_scale: float = 0.3
    prob: float = 0.7

    def __call__(self, img_rgb: np.ndarray, H_total: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.prob > 0 and torch.rand(()) >= self.prob:
            return img_rgb, H_total

        h, w = img_rgb.shape[:2]
        d = self.distortion_scale * min(h, w)

        # src corners
        src = np.array([[0.0, 0.0],
                        [w - 1.0, 0.0],
                        [w - 1.0, h - 1.0],
                        [0.0, h - 1.0]], dtype=np.float32)

        # random dst corners within distortion
        def jitter(xy):
            jx = float(torch.empty(1).uniform_(-d, d))
            jy = float(torch.empty(1).uniform_(-d, d))
            return [xy[0] + jx, xy[1] + jy]

        dst = np.array([jitter(src[0]), jitter(src[1]), jitter(src[2]), jitter(src[3])], dtype=np.float32)

        H_p = cv2.getPerspectiveTransform(src, dst).astype(np.float32)  # src->dst

        warped = cv2.warpPerspective(
            img_rgb, H_p, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        return warped, H_p @ H_total


@dataclass
class RandomHFlipSameSize:
    prob: float = 0.5

    def __call__(self, img_rgb: np.ndarray, H_total: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.prob > 0 and torch.rand(()) >= self.prob:
            return img_rgb, H_total
        h, w = img_rgb.shape[:2]
        flipped = np.ascontiguousarray(img_rgb[:, ::-1, :])
        H_flip = np.array([[-1.0, 0.0, float(w - 1)],
                           [0.0,  1.0, 0.0],
                           [0.0,  0.0, 1.0]], dtype=np.float32)
        return flipped, H_flip @ H_total


@dataclass
class GeometryAugmentWithHomography:
    """
    Geometry pipeline for View B:
      - (optional) random resized crop -> resize back to same size
      - (optional) random affine
      - (optional) random perspective
      - (optional) random hflip
    It returns:
      img_b (numpy RGB)
      H_A2B mapping A coords -> B coords
      and you can invert to get H_B2A.
    """
    use_rrc: bool = True
    rrc_scale: Tuple[float, float] = (0.5, 1.0)
    rrc_ratio: Tuple[float, float] = (0.75, 1.333)

    use_affine: bool = False
    affine_degrees: float = 10.0
    affine_translate: Tuple[float, float] = (0.1, 0.1)
    affine_scale: Tuple[float, float] = (0.9, 1.1)
    affine_prob: float = 0.7

    use_perspective: bool = True
    perspective_distortion: float = 0.3
    perspective_prob: float = 0.7

    hflip_prob: float = 0.5

    def __call__(self, img: Image.Image) -> Tuple[Image.Image, np.ndarray]:
        if img.mode != "RGB":
            img = img.convert("RGB")

        img_rgb = np.array(img, dtype=np.uint8)  # RGB
        H_total = np.eye(3, dtype=np.float32)  # A->B

        if self.use_rrc:
            rrc = RandomResizedCropSameSize(scale=self.rrc_scale, ratio=self.rrc_ratio)
            img_rgb, H_total = rrc(img_rgb, H_total)

        if self.use_affine:
            aff = RandomAffineSameSize(
                degrees=self.affine_degrees,
                translate=self.affine_translate,
                scale=self.affine_scale,
                prob=self.affine_prob,
            )
            img_rgb, H_total = aff(img_rgb, H_total)

        if self.use_perspective:
            persp = RandomPerspectiveSameSize(
                distortion_scale=self.perspective_distortion,
                prob=self.perspective_prob,
            )
            img_rgb, H_total = persp(img_rgb, H_total)

        if self.hflip_prob > 0:
            hf = RandomHFlipSameSize(prob=self.hflip_prob)
            img_rgb, H_total = hf(img_rgb, H_total)

        img_b = Image.fromarray(img_rgb, mode="RGB")
        return img_b, H_total


# -----------------------------
# Two-view wrapper (full pipeline)
# -----------------------------

@dataclass
class TwoViewSSLTransform:
    """
    Full transform:
      1) ResizeKeepARShortLong
      2) PadRightBottomToDivisor(16)
      3) Create View A: appearance (PIL)
      4) Create View B: geometry (PIL->cv2) + return homography
      5) Normalize to tensors
      6) Return H_B2A for mask warping
    """
    resize: ResizeKeepARShortLong
    pad: PadRightBottomToDivisor
    view_a: AppearanceAugment
    view_b: GeometryAugmentWithHomography
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    def _to_tensor_norm(self, img: Image.Image) -> torch.Tensor:
        t = TF.to_tensor(img)  # [0,1], CxHxW
        t = TF.normalize(t, mean=list(self.mean), std=list(self.std))
        return t

    @staticmethod
    def _safe_inv(H: np.ndarray) -> np.ndarray:
        H = np.asarray(H, dtype=np.float32)
        if not np.isfinite(H).all():
            return np.eye(3, dtype=np.float32)
        det = float(np.linalg.det(H))
        if (not np.isfinite(det)) or abs(det) < 1e-8:
            return np.eye(3, dtype=np.float32)
        inv = np.linalg.inv(H).astype(np.float32)
        if not np.isfinite(inv).all():
            return np.eye(3, dtype=np.float32)
        return inv

    @staticmethod
    def _valid_mask_from_resize_pad(resized_hw: Tuple[int, int], padded_hw: Tuple[int, int]) -> np.ndarray:
        rh, rw = resized_hw
        ph, pw = padded_hw
        mask = np.zeros((ph, pw), dtype=np.uint8)
        mask[:rh, :rw] = 1
        return mask

    def __call__(self, img: Image.Image) -> Dict[str, Any]:
        # 1) Resize + pad (paper preprocessing)
        img_r, meta_r = self.resize(img)
        img_p, meta_p = self.pad(img_r)

        # base valid region (before padding is valid, padding is invalid)
        valid_a = self._valid_mask_from_resize_pad(meta_r["resized_size_hw"], meta_p["padded_size_hw"])

        # 2) View A (appearance)
        img_a_pil, H_a = self.view_a(img_p)

        # 3) View B (geometry) + homography A->B
        img_b_pil, H_a2b = self.view_b(img_p)

        # If View A includes optional flip, its H_a is included. We want mapping from A coords -> B coords:
        # A coords are after H_a; however appearance typically doesn't change coords.
        # If H_a is non-identity, we interpret:
        #   x_a = H_a x_base, x_b = H_a2b x_base  => x_b = (H_a2b @ inv(H_a)) x_a
        H_base2a = H_a
        H_base2b = H_a2b
        H_a2b_final = H_base2b @ self._safe_inv(H_base2a)
        if (not np.isfinite(H_a2b_final).all()) or abs(float(np.linalg.det(H_a2b_final))) < 1e-8:
            H_a2b_final = np.eye(3, dtype=np.float32)
        H_b2a = self._safe_inv(H_a2b_final)

        # 4) Warp valid mask to B to get valid_b (use perspective; mask is uint8)
        h, w = valid_a.shape
        valid_b = cv2.warpPerspective(
            valid_a, H_a2b_final.astype(np.float32), (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        # 5) To tensor + normalize
        img_a = self._to_tensor_norm(img_a_pil)
        img_b = self._to_tensor_norm(img_b_pil)

        out = {
            "img_a": img_a,  # torch.FloatTensor [3,H,W]
            "img_b": img_b,  # torch.FloatTensor [3,H,W]
            "H_b2a": torch.from_numpy(H_b2a),  # [3,3] float32
            "H_a2b": torch.from_numpy(H_a2b_final.astype(np.float32)),  # [3,3] float32
            "valid_mask_a": torch.from_numpy(valid_a.astype(np.uint8)),  # [H,W] 0/1
            "valid_mask_b": torch.from_numpy(valid_b.astype(np.uint8)),  # [H,W] 0/1
            "meta": {
                **meta_r,
                **meta_p,
            },
        }
        return out
