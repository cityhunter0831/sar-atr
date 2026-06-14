"""
Clutter-transfer boundary blending (권승주 담당 — Exp A / +SSIM).

Pipeline:
    1. feather_blend : alpha-feathering at chip boundary (soft mask)
    2. poisson_blend : seamless Poisson cloning via cv2 (requires OpenCV)
    3. compute_ssim  : structural similarity for boundary quality metric

Reference: gengzhe2015/SAR-target-recognition (Table 4 image generation)
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Float [0,1] or raw amplitude → uint8 [0,255]."""
    img = np.clip(img, 0, None)
    if img.max() > 1.0:
        img = img / img.max()
    return (img * 255).astype(np.uint8)


def _soft_mask(h: int, w: int, border: int = 8) -> np.ndarray:
    """Gaussian-feathered mask: 1 inside, 0 at border."""
    mask = np.ones((h, w), dtype=np.float32)
    for i in range(border):
        val = (i + 1) / (border + 1)
        mask[i, :] = np.minimum(mask[i, :], val)
        mask[-(i + 1), :] = np.minimum(mask[-(i + 1), :], val)
        mask[:, i] = np.minimum(mask[:, i], val)
        mask[:, -(i + 1)] = np.minimum(mask[:, -(i + 1)], val)
    return mask


def feather_blend(target: Tensor, background: Tensor, border: int = 8) -> Tensor:
    """
    Alpha-feathering: paste target chip onto background with soft mask.

    Args:
        target:     [1, H, W] float32 — SAR target chip.
        background: [1, H, W] float32 — clutter background (same size or larger).
        border:     Width (pixels) of feathering transition.

    Returns:
        [1, H, W] float32 blended image.
    """
    _, h, w = target.shape
    # Crop/resize background to match target size
    bg = background[:, :h, :w]
    if bg.shape[-1] < w or bg.shape[-2] < h:
        import torch.nn.functional as F
        bg = F.interpolate(bg.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False).squeeze(0)

    mask = torch.from_numpy(_soft_mask(h, w, border)).unsqueeze(0)
    return mask * target + (1 - mask) * bg


def poisson_blend(target: Tensor, background: Tensor) -> Tensor:
    """
    Seamless Poisson cloning via OpenCV.
    Falls back to feather_blend if cv2 is unavailable.

    Args:
        target:     [1, H, W] float32 — SAR target chip (foreground).
        background: [1, H, W] float32 — clutter background (same size).

    Returns:
        [1, H, W] float32 blended image.
    """
    try:
        import cv2
    except ImportError:
        return feather_blend(target, background)

    _, h, w = target.shape

    # cv2.seamlessClone requires 3-channel uint8
    def _to_bgr(t: Tensor) -> np.ndarray:
        arr = _to_uint8(t.squeeze(0).numpy())
        return np.stack([arr, arr, arr], axis=-1)

    src = _to_bgr(target)
    dst_img = _to_bgr(background[:, :h, :w])

    # Elliptical mask centered on chip
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (w // 2, h // 2), (w // 2 - 4, h // 2 - 4), 0, 0, 360, 255, -1)

    center = (w // 2, h // 2)
    blended = cv2.seamlessClone(src, dst_img, mask, center, cv2.NORMAL_CLONE)

    result = torch.from_numpy(blended[:, :, 0].astype(np.float32) / 255.0).unsqueeze(0)
    return result


def compute_ssim(img_a: Tensor, img_b: Tensor, window_size: int = 11) -> float:
    """
    Structural Similarity Index (SSIM) between two [1, H, W] float32 tensors.
    Uses a Gaussian window approximation.
    """
    a = img_a.squeeze(0).numpy().astype(np.float64)
    b = img_b.squeeze(0).numpy().astype(np.float64)

    try:
        from skimage.metrics import structural_similarity
        return float(structural_similarity(a, b, data_range=1.0))
    except ImportError:
        pass

    # Fallback: simple patch-based approximation
    C1, C2 = (0.01) ** 2, (0.03) ** 2
    mu_a, mu_b = a.mean(), b.mean()
    sig_a = a.std()
    sig_b = b.std()
    sig_ab = ((a - mu_a) * (b - mu_b)).mean()
    ssim = ((2 * mu_a * mu_b + C1) * (2 * sig_ab + C2)) / (
        (mu_a**2 + mu_b**2 + C1) * (sig_a**2 + sig_b**2 + C2)
    )
    return float(ssim)


def clutter_transfer(
    target: Tensor,
    clutter_bg: Tensor,
    method: str = "feather",
    border: int = 8,
) -> Tensor:
    """
    Top-level clutter transfer function used by Exp A dataset builder.

    Args:
        target:     [1, H, W] original MSTAR target chip.
        clutter_bg: [1, H, W] background clutter chip.
        method:     'feather' | 'poisson'.
        border:     Feathering border width (pixels).
    """
    if method == "poisson":
        return poisson_blend(target, clutter_bg)
    return feather_blend(target, clutter_bg, border=border)
