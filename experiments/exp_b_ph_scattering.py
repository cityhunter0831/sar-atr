"""
Exp B — Phase History / Scattering Center Analysis (박승준)

Validates that the top-K scattering centres extracted via 2D FFT
are spatially coherent with the Grad-CAM activation heatmap from
a trained SMPL / ResNet18 model.

Success criterion: ≥3 test images show ≥X% overlap (IoU / correlation)
between scattering centres and Grad-CAM heatmap.

Requires:
    - MSTAR raw files under  data/mstar/targets/  (or mixed_targets/)
    - A trained model checkpoint  results/exp_a/smpl_seed0.pth
      (or pass model directly via CLI flag --mock to use random weights)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from augmentation.ph_extraction import (
    amplitude_to_tensor,
    extract_scattering_centers,
    extract_spatial_scattering_centers,
    read_mstar_raw,
    visualize_scattering,
)
from core.models import get_model

MSTAR_RAW_DIR = Path("data/mstar/MSTAR_PUBLIC_TARGETS_CHIPS_T72_BMP2_BTR70_SLICY/TARGETS/TRAIN/17_DEG")
RESULTS_DIR = Path("results/exp_b")
CLASSES = ["BMP2", "BTR70", "T72"]


# ─── Grad-CAM (minimal, no external dependency) ───────────────────────────────

class GradCAM:
    """
    Minimal Grad-CAM for any conv-based model.
    Targets the last Conv2d layer automatically.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self._fmaps: torch.Tensor | None = None
        self._grads: torch.Tensor | None = None
        self._handle_f = None
        self._handle_g = None
        self._hook_last_conv()

    def _hook_last_conv(self):
        last_conv = None
        for m in self.model.modules():
            if isinstance(m, torch.nn.Conv2d):
                last_conv = m
        assert last_conv is not None, "No Conv2d found in model."

        def fwd_hook(_, __, output):
            self._fmaps = output

        def bwd_hook(_, __, grad_out):
            self._grads = grad_out[0]

        self._handle_f = last_conv.register_forward_hook(fwd_hook)
        self._handle_g = last_conv.register_full_backward_hook(bwd_hook)

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        self.model.zero_grad()
        logits = self.model(x)
        target = logits[0, class_idx if class_idx is not None else logits.argmax(1).item()]
        target.backward()

        weights = self._grads.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self._fmaps).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def remove(self):
        if self._handle_f:
            self._handle_f.remove()
        if self._handle_g:
            self._handle_g.remove()


# ─── Scatter centre → binary mask ────────────────────────────────────────────

def _centers_to_mask(
    centers: list[tuple[float, float]], h: int, w: int, radius: int = 5
) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.float32)
    for cy, cx in centers:
        # Map FFT coordinates (centred) to image coordinates
        iy = int(np.clip(cy, 0, h - 1))
        ix = int(np.clip(cx, 0, w - 1))
        y0, y1 = max(0, iy - radius), min(h, iy + radius)
        x0, x1 = max(0, ix - radius), min(w, ix + radius)
        mask[y0:y1, x0:x1] = 1.0
    return mask


def _iou(mask_a: np.ndarray, mask_b: np.ndarray, threshold: float = 0.5) -> float:
    a = mask_a > threshold
    b = mask_b > threshold
    inter = (a & b).sum()
    union = (a | b).sum()
    return float(inter / union) if union > 0 else 0.0


# ─── Per-sample analysis ──────────────────────────────────────────────────────

def analyse_sample(
    raw_path: Path,
    model: torch.nn.Module,
    k: int = 5,
    save_dir: Path | None = None,
) -> dict:
    amplitude = read_mstar_raw(raw_path)
    image_t = amplitude_to_tensor(amplitude)

    ph_map = extract_scattering_centers(amplitude, k=k)
    # BUG-1 수정: Grad-CAM과 IoU 비교는 공간 도메인 좌표 사용
    spatial_centers = extract_spatial_scattering_centers(amplitude, k=k)

    # Grad-CAM
    gcam = GradCAM(model)
    cam = gcam(image_t.unsqueeze(0))
    gcam.remove()

    h, w = amplitude.shape
    scatter_mask = _centers_to_mask(spatial_centers, h, w)
    iou = _iou(scatter_mask, cam)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(amplitude, cmap="gray")
        axes[0].set_title("Amplitude")
        axes[0].axis("off")
        axes[1].imshow(ph_map.spectrum, cmap="hot")
        for cy, cx in ph_map.scattering_centers:
            axes[1].plot(cx, cy, "c+", markersize=8, markeredgewidth=2)
        axes[1].set_title(f"PH Spectrum (top-{k})")
        axes[1].axis("off")
        axes[2].imshow(amplitude, cmap="gray")
        axes[2].imshow(cam, cmap="jet", alpha=0.5)
        axes[2].set_title(f"Grad-CAM (IoU={iou:.2f})")
        axes[2].axis("off")
        plt.tight_layout()
        fig.savefig(save_dir / f"{raw_path.stem}_analysis.png", dpi=150)
        plt.close(fig)

    return {"file": raw_path.name, "iou": iou, "n_centers": len(spatial_centers)}


# ─── Runner ───────────────────────────────────────────────────────────────────

def run(
    model_name: str = "smpl",
    checkpoint: Path | None = None,
    use_mock_model: bool = False,
    k: int = 5,
    n_samples: int = 10,
    save_dir: Path = RESULTS_DIR,
) -> list[dict]:
    num_classes = len(CLASSES)
    model = get_model(model_name, num_classes)

    if not use_mock_model and checkpoint is not None and checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        print(f"Loaded checkpoint: {checkpoint}")
    else:
        print("Using random-weight model (pass --checkpoint to use trained model).")

    model.eval()

    # Gather raw files
    raw_files: list[Path] = []
    if MSTAR_RAW_DIR.exists():
        for cls in CLASSES:
            cls_dir = MSTAR_RAW_DIR / cls
            if cls_dir.exists():
                # MSTAR raw: 확장자가 3자리 숫자 (elevation 각도, e.g. .017)
                raw_files.extend(
                    p for p in cls_dir.rglob("*")
                    if p.is_file() and p.suffix.lstrip(".").isdigit() and len(p.suffix) == 4
                )
    if not raw_files:
        print("[Exp B] No MSTAR raw files found — skipping real analysis.")
        print("        Download MSTAR Targets package and place under data/mstar/targets/<class>/")
        return []

    records = []
    for p in raw_files[:n_samples]:
        rec = analyse_sample(p, model, k=k, save_dir=save_dir)
        print(f"  {rec['file']:40s}  IoU={rec['iou']:.3f}  centers={rec['n_centers']}")
        records.append(rec)

    valid = [r["iou"] for r in records if r["iou"] > 0]
    if valid:
        print(f"\nMean IoU = {np.mean(valid):.3f}  (n={len(valid)} samples)")
        print(f"Criterion (≥3 samples IoU>0): {'PASS' if len(valid) >= 3 else 'FAIL'}")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="smpl", choices=["smpl", "resnet18"])
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--mock", action="store_true", help="Use random weights")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--n-samples", type=int, default=10)
    args = parser.parse_args()
    run(
        model_name=args.model,
        checkpoint=args.checkpoint,
        use_mock_model=args.mock,
        k=args.k,
        n_samples=args.n_samples,
    )
