"""
픽셀 단위 XAI 어트리뷰션 (Exp B 우리 팀 개선 #3 확장).

Grad-CAM은 SMPL의 마지막 conv 특징맵(8×8) 해상도에 묶여 얇은 점 산란체를
정밀하게 국소화하지 못한다. 아래 두 방법은 입력 픽셀 공간(64×64)에서 직접
어트리뷰션을 계산하므로 CAM의 해상도 천장을 원천적으로 우회한다.

- occlusion_sensitivity: 인과적. 패치를 가리고 타깃 클래스 확률 하락을 측정.
  "이 영역을 지우면 분류가 망가지는가" → 산란점의 물리적 기여를 직접 검증.
- integrated_gradients / smoothgrad_ig: 공리적 그래디언트 어트리뷰션.
  baseline→입력 경로 적분(IG) + 노이즈 평균(SmoothGrad)으로 SAR 스페클 억제.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _target_index(model: torch.nn.Module, x: torch.Tensor, class_idx: int | None) -> int:
    if class_idx is not None:
        return class_idx
    with torch.no_grad():
        return int(model(x).argmax(1).item())


def occlusion_sensitivity(
    model: torch.nn.Module,
    x: torch.Tensor,
    class_idx: int | None = None,
    patch: int = 8,
    stride: int = 4,
    baseline: float = 0.0,
) -> np.ndarray:
    """
    Occlusion sensitivity map [H, W] in [0,1].

    patch×patch 창을 stride 간격으로 밀며 baseline 값으로 가린 뒤, 타깃 클래스
    확률(softmax) 하락량을 그 창 영역에 누적. 하락이 클수록(=중요) 값이 높음.
    x: [1,1,H,W]. 모델과 같은 device 가정.
    """
    model.eval()
    device = next(model.parameters()).device
    x = x.to(device)
    _, _, h, w = x.shape
    with torch.no_grad():
        base_prob = F.softmax(model(x), dim=1)
    tgt = _target_index(model, x, class_idx)
    base_score = float(base_prob[0, tgt])

    heat = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)
    ys = list(range(0, h, stride))
    xs = list(range(0, w, stride))

    # 배치로 묶어 forward (CPU에서도 감당 가능한 규모)
    patches: list[tuple[int, int]] = [(yy, xx) for yy in ys for xx in xs]
    B = 64
    with torch.no_grad():
        for i in range(0, len(patches), B):
            chunk = patches[i:i + B]
            batch = x.repeat(len(chunk), 1, 1, 1).clone()
            for j, (yy, xx) in enumerate(chunk):
                batch[j, :, yy:min(yy + patch, h), xx:min(xx + patch, w)] = baseline
            probs = F.softmax(model(batch), dim=1)[:, tgt].cpu().numpy()
            for j, (yy, xx) in enumerate(chunk):
                drop = max(0.0, base_score - float(probs[j]))
                heat[yy:min(yy + patch, h), xx:min(xx + patch, w)] += drop
                counts[yy:min(yy + patch, h), xx:min(xx + patch, w)] += 1.0
    counts[counts == 0] = 1.0
    heat /= counts
    mn, mx = heat.min(), heat.max()
    return ((heat - mn) / (mx - mn + 1e-8)).astype(np.float32)


def integrated_gradients(
    model: torch.nn.Module,
    x: torch.Tensor,
    class_idx: int | None = None,
    steps: int = 32,
    baseline: torch.Tensor | float = 0.0,
) -> np.ndarray:
    """
    Integrated Gradients map [H, W] in [0,1] (절댓값 정규화).

    baseline→x 직선 경로를 steps개로 적분한 그래디언트×(x−baseline).
    x: [1,1,H,W].
    """
    model.eval()
    device = next(model.parameters()).device
    x = x.to(device)
    if isinstance(baseline, (int, float)):
        base = torch.full_like(x, float(baseline))
    else:
        base = baseline.to(device)
    tgt = _target_index(model, x, class_idx)

    grads = torch.zeros_like(x)
    for s in range(1, steps + 1):
        alpha = s / steps
        xi = (base + alpha * (x - base)).clone().requires_grad_(True)
        model.zero_grad()
        out = model(xi)
        out[0, tgt].backward()
        grads += xi.grad.detach()
    avg_grad = grads / steps
    attr = (avg_grad * (x - base)).squeeze().cpu().numpy()
    attr = np.abs(attr)
    mn, mx = attr.min(), attr.max()
    return ((attr - mn) / (mx - mn + 1e-8)).astype(np.float32)


def smoothgrad_ig(
    model: torch.nn.Module,
    x: torch.Tensor,
    class_idx: int | None = None,
    steps: int = 32,
    n_noise: int = 8,
    noise_std: float = 0.15,
    baseline: torch.Tensor | float = 0.0,
) -> np.ndarray:
    """
    SmoothGrad-적용 Integrated Gradients [H, W] in [0,1].

    입력에 가우시안 노이즈를 n_noise회 더해 IG를 평균 → SAR 스페클로 인한
    그래디언트 잡음을 억제해 산란점 어트리뷰션을 선명하게 함.
    """
    device = next(model.parameters()).device
    x = x.to(device)
    # class_idx를 노이즈 간 고정 (원본 예측 기준)
    tgt = _target_index(model, x, class_idx)
    acc = np.zeros(x.shape[-2:], dtype=np.float32)
    for _ in range(n_noise):
        noisy = x + torch.randn_like(x) * noise_std
        acc += integrated_gradients(model, noisy, class_idx=tgt,
                                    steps=steps, baseline=baseline)
    acc /= n_noise
    mn, mx = acc.min(), acc.max()
    return ((acc - mn) / (mx - mn + 1e-8)).astype(np.float32)
