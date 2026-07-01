"""
Grad-CAM implementation.

Hook 패턴은 jacobgil/pytorch-grad-cam 참조:
  output tensor에 register_hook() 등록 → register_full_backward_hook보다 gradient timing 안정적
  forward hook에서 .cpu().detach()로 메모리 누수 방지
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """
    Minimal Grad-CAM for any conv-based model.
    Targets the last Conv2d layer automatically.

    Usage:
        gcam = GradCAM(model)
        cam = gcam(image_tensor.unsqueeze(0))  # [H, W] float32 in [0,1]
        gcam.remove()
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self._fmaps: torch.Tensor | None = None
        self._grads: torch.Tensor | None = None
        self._handle_f = None
        self._hook_last_conv()

    def _hook_last_conv(self):
        last_conv = None
        for m in self.model.modules():
            if isinstance(m, torch.nn.Conv2d):
                last_conv = m
        assert last_conv is not None, "No Conv2d found in model."

        def fwd_hook(_, __, output):
            self._fmaps = output.cpu().detach()
            def _store_grad(grad):
                self._grads = grad.cpu().detach()
            output.register_hook(_store_grad)

        self._handle_f = last_conv.register_forward_hook(fwd_hook)

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        self.model.zero_grad()
        logits = self.model(x)
        target_idx = class_idx if class_idx is not None else int(logits.argmax(1).item())
        target = logits[0, target_idx]
        target.backward()

        if self._grads is None or self._fmaps is None:
            h, w = x.shape[-2:]
            return np.zeros((h, w), dtype=np.float32)

        weights = self._grads.mean(dim=[2, 3], keepdim=True)
        cam = F.relu((weights * self._fmaps).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        return ((cam - cam_min) / (cam_max - cam_min + 1e-8)).astype(np.float32)

    def remove(self):
        if self._handle_f:
            self._handle_f.remove()
