"""IoU 계산 및 시각화 유틸리티 (Exp B 우리 팀 개선 #3)."""
from __future__ import annotations

import numpy as np


def centers_to_mask(
    centers: list[tuple[float, float]], h: int, w: int, radius: int = 5
) -> np.ndarray:
    """산란점 좌표 목록 → 이진 마스크 [H, W]."""
    mask = np.zeros((h, w), dtype=np.float32)
    for cy, cx in centers:
        iy = int(np.clip(cy, 0, h - 1))
        ix = int(np.clip(cx, 0, w - 1))
        y0, y1 = max(0, iy - radius), min(h, iy + radius)
        x0, x1 = max(0, ix - radius), min(w, ix + radius)
        mask[y0:y1, x0:x1] = 1.0
    return mask


def iou(mask_a: np.ndarray, mask_b: np.ndarray, threshold: float = 0.5) -> float:
    """두 마스크의 IoU 계산."""
    a = mask_a > threshold
    b = mask_b > threshold
    inter = (a & b).sum()
    union = (a | b).sum()
    return float(inter / union) if union > 0 else 0.0
