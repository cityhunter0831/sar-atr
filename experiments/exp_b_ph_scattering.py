"""
Exp B — Phase History Interpolation Augmentation (논문 Section 2.1 재현)

논문 Table 3 재현:
  조건                | SMPL 5-class acc
  원본만 학습          | ~56.6%
  PH 보간 증강 추가   | ~96.6%

방법:
  1. Mixed Targets CD2에서 클래스별 이미지 pair 구성
  2. IFFT → PH 도메인 선형 보간 → FFT → 합성 이미지 생성
  3. 원본 + 합성 이미지로 SMPL 학습 후 테스트

우리 팀 개선 #3 (Grad-CAM 산란점 일치도 검증):
  run_gradcam_analysis() 참고

데이터: MSTAR Mixed Targets CD2 (Phoenix 헤더 확인됨)
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset

from augmentation.ph_extraction import (
    amplitude_to_tensor,
    extract_scattering_centers,
    extract_spatial_scattering_centers,
    interpolate_phase_history,
    read_mstar_complex,
    read_mstar_raw,
    visualize_scattering,
)
from core.interfaces import EvalResult, SARDataset, SARSample, TrainConfig
from core.mock_data import MockSARDataset
from core.models import get_model
from core.train import train_model
from gradcam.cam import GradCAM
from gradcam.scatter_overlap import centers_to_mask, iou as compute_iou

MSTAR_RAW_DIR = Path("data/mstar/MSTAR_PUBLIC_MIXED_TARGETS_CD2")
RESULTS_DIR = Path("results/exp_b")
CLASSES = ["2S1", "BRDM_2", "BTR_60", "D7", "T62", "ZIL131", "ZSU_23_4"]


# ─── MSTAR raw file dataset ───────────────────────────────────────────────────

def _has_phoenix_header(path: Path) -> bool:
    """파일 앞 4KB만 읽어 Phoenix 헤더 존재 여부를 빠르게 확인."""
    try:
        with open(path, "rb") as f:
            chunk = f.read(4096)
        return b"EndofPhoenixHeader" in chunk
    except Exception:
        return False


def _collect_raw_files(root: Path, classes: list[str]) -> dict[str, list[Path]]:
    """Mixed Targets 디렉토리에서 클래스별 raw 파일 목록 수집.
    Phoenix 헤더가 없는 파일은 제외해 zeros 학습 방지."""
    result: dict[str, list[Path]] = {}
    for cls in classes:
        candidates: list[Path] = []
        for p in root.rglob(f"*{cls}*/*"):
            if p.is_file() and p.suffix.lstrip(".").isdigit() and len(p.suffix) >= 3:
                candidates.append(p)
        if not candidates:
            for p in root.rglob("*"):
                if p.is_file() and cls.lower() in str(p).lower():
                    if p.suffix.lstrip(".").isdigit():
                        candidates.append(p)
        files = [p for p in candidates if _has_phoenix_header(p)]
        result[cls] = files
    return result


class MSTARRawDataset(SARDataset):
    """MSTAR Mixed Targets raw binary 파일 → 이미지 Dataset."""

    def __init__(self, files_by_class: dict[str, list[Path]], class_names: list[str]):
        self._class_names = class_names
        self._samples: list[tuple[Path, int]] = []
        for idx, cls in enumerate(class_names):
            for p in files_by_class.get(cls, []):
                self._samples.append((p, idx))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> SARSample:
        path, label = self._samples[idx]
        try:
            amp = read_mstar_raw(path)
        except Exception:
            amp = np.zeros((128, 128), dtype=np.float32)
        t = amplitude_to_tensor(amp)
        return SARSample(image=t, label=label, meta={"source": str(path), "class_name": self._class_names[label]})

    @property
    def class_names(self) -> list[str]:
        return self._class_names


# ─── PH 보간 증강 데이터셋 ────────────────────────────────────────────────────

class PHAugmentedDataset(SARDataset):
    """
    원본 MSTAR 이미지 + PH 보간 합성 이미지를 합친 Dataset.

    각 클래스에서 이미지 pair를 구성하고, n_alphas개의 alpha 값으로
    합성 이미지를 생성해서 원본에 추가합니다.

    lazy loading: __init__에서 경로/메타 튜플만 저장, I/O는 __getitem__에서 수행.
    """

    def __init__(
        self,
        files_by_class: dict[str, list[Path]],
        class_names: list[str],
        n_alphas: int = 5,
        seed: int = 0,
    ):
        self._class_names = class_names
        # (kind, data, label) — kind: "original" | "synth"
        self._samples: list[tuple[str, object, int]] = []

        alphas = [i / (n_alphas + 1) for i in range(1, n_alphas + 1)]

        for idx, cls in enumerate(class_names):
            files = files_by_class.get(cls, [])
            if not files:
                continue

            for p in files:
                self._samples.append(("original", p, idx))

            if len(files) >= 2:
                pairs = list(zip(files, files[1:] + files[:1]))
                for p_a, p_b in pairs:
                    for alpha in alphas:
                        self._samples.append(("synth", (p_a, p_b, alpha), idx))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> SARSample:
        kind, data, label = self._samples[idx]
        if kind == "original":
            try:
                amp = read_mstar_raw(data)
            except Exception:
                amp = np.zeros((128, 128), dtype=np.float32)
        else:
            p_a, p_b, alpha = data
            try:
                amp = interpolate_phase_history(
                    read_mstar_complex(p_a), read_mstar_complex(p_b), alpha
                )
            except Exception:
                amp = np.zeros((128, 128), dtype=np.float32)
        return SARSample(image=amplitude_to_tensor(amp), label=label,
                         meta={"class_name": self._class_names[label]})

    @property
    def class_names(self) -> list[str]:
        return self._class_names


# ─── Main runners ─────────────────────────────────────────────────────────────

def _data_available() -> bool:
    return MSTAR_RAW_DIR.exists() and any(MSTAR_RAW_DIR.rglob("*"))


def run(
    model_name: str = "smpl",
    epochs: int = 60,
    n_interp: int = 5,
    seed: int = 0,
    save_dir: Path = RESULTS_DIR,
    use_mock: bool = False,
) -> dict:
    """
    논문 Table 3 재현: 원본 학습 vs PH 보간 증강 학습 정확도 비교.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    num_classes = len(CLASSES)

    if use_mock or not _data_available():
        if not use_mock:
            print(f"[Exp B] MSTAR Mixed Targets not found at {MSTAR_RAW_DIR} — using mock data.")
        n = 60 * num_classes
        base_ds = MockSARDataset(n=n, num_classes=num_classes, seed=seed)
        aug_ds  = MockSARDataset(n=n * (n_interp + 1), num_classes=num_classes, seed=seed + 1)
        test_ds = MockSARDataset(n=20 * num_classes, num_classes=num_classes, seed=seed + 2)
    else:
        print(f"[Exp B] Loading MSTAR Mixed Targets from {MSTAR_RAW_DIR} ...")
        files_by_class = _collect_raw_files(MSTAR_RAW_DIR, CLASSES)
        for cls, files in files_by_class.items():
            print(f"  {cls}: {len(files)} files")

        # 80/20 train/test split
        train_files: dict[str, list[Path]] = {}
        test_files:  dict[str, list[Path]] = {}
        rng = random.Random(seed)
        for cls, files in files_by_class.items():
            shuffled = files[:]
            rng.shuffle(shuffled)
            n_train = max(1, int(len(shuffled) * 0.8))
            train_files[cls] = shuffled[:n_train]
            test_files[cls]  = shuffled[n_train:]

        base_ds = MSTARRawDataset(train_files, CLASSES)
        aug_ds  = PHAugmentedDataset(train_files, CLASSES, n_alphas=n_interp, seed=seed)
        test_ds = MSTARRawDataset(test_files, CLASSES)

    results = {}

    # Condition 1: 원본만 학습
    print("\n── Condition 1: 원본 이미지만 학습 ────────────────────────────────")
    model1 = get_model(model_name, num_classes)
    cfg = TrainConfig(model_name=model_name, num_classes=num_classes, epochs=epochs, seed=seed)
    model1, r1 = train_model(model1, base_ds, test_ds, cfg)
    results["no_aug_acc"] = r1.accuracy
    print(f"  → 테스트 정확도: {r1.accuracy * 100:.1f}%")

    # Condition 2: PH 보간 증강 추가
    print("\n── Condition 2: PH 보간 증강 추가 학습 ────────────────────────────")
    model2 = get_model(model_name, num_classes)
    model2, r2 = train_model(model2, aug_ds, test_ds, cfg)
    results["ph_aug_acc"] = r2.accuracy
    print(f"  → 테스트 정확도: {r2.accuracy * 100:.1f}%")

    print("\n── Exp B 결과 (Table 3 재현) ───────────────────────────────────────")
    print(f"  원본만:      {results['no_aug_acc'] * 100:.1f}%  (논문: 56.6%)")
    print(f"  PH 보간 추가: {results['ph_aug_acc'] * 100:.1f}%  (논문: 96.6%)")

    import json
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def run_gradcam_analysis(
    model_name: str = "smpl",
    checkpoint: Path | None = None,
    k: int = 5,
    n_samples: int = 10,
    save_dir: Path = RESULTS_DIR / "gradcam",
) -> list[dict]:
    """
    우리 팀 개선 #3: Grad-CAM 히트맵과 공간 산란점 좌표 IoU 비교.
    모델이 실제 산란점 위치를 보고 분류하는지 물리적 신뢰도 검증.
    """
    num_classes = len(CLASSES)
    model = get_model(model_name, num_classes)
    if checkpoint is not None and checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        print(f"Loaded checkpoint: {checkpoint}")
    else:
        print("Using random-weight model.")
    model.eval()

    raw_files: list[Path] = []
    if _data_available():
        files_by_class = _collect_raw_files(MSTAR_RAW_DIR, CLASSES)
        for files in files_by_class.values():
            raw_files.extend(files)
    if not raw_files:
        print("[Grad-CAM] No raw files found — skipping.")
        return []

    save_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for p in raw_files[:n_samples]:
        try:
            amp = read_mstar_raw(p)
        except Exception as e:
            print(f"  Skip {p.name}: {e}")
            continue

        image_t = amplitude_to_tensor(amp)
        spatial_centers = extract_spatial_scattering_centers(amp, k=k)
        ph_map = extract_scattering_centers(amp, k=k)

        gcam = GradCAM(model)
        cam = gcam(image_t.unsqueeze(0))
        gcam.remove()

        h, w = amp.shape
        scatter_mask = centers_to_mask(spatial_centers, h, w)
        iou = compute_iou(scatter_mask, cam)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(amp, cmap="gray"); axes[0].set_title("Amplitude"); axes[0].axis("off")
        axes[1].imshow(ph_map.spectrum, cmap="hot")
        for cy, cx in ph_map.scattering_centers:
            axes[1].plot(cx, cy, "c+", markersize=8, markeredgewidth=2)
        axes[1].set_title(f"PH Spectrum (top-{k})"); axes[1].axis("off")
        axes[2].imshow(amp, cmap="gray")
        axes[2].imshow(cam, cmap="jet", alpha=0.5)
        axes[2].set_title(f"Grad-CAM (IoU={iou:.2f})"); axes[2].axis("off")
        plt.tight_layout()
        fig.savefig(save_dir / f"{p.stem}_analysis.png", dpi=150)
        plt.close(fig)

        rec = {"file": p.name, "iou": iou, "n_centers": len(spatial_centers)}
        print(f"  {p.name:40s}  IoU={iou:.3f}  centers={len(spatial_centers)}")
        records.append(rec)

    valid = [r["iou"] for r in records if r["iou"] > 0]
    if valid:
        print(f"\nMean IoU = {np.mean(valid):.3f}  (n={len(valid)})")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="smpl", choices=["smpl", "resnet18"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--n-interp", type=int, default=5)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--gradcam", action="store_true", help="Grad-CAM 분석 실행")
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()

    if args.gradcam:
        run_gradcam_analysis(
            model_name=args.model,
            checkpoint=args.checkpoint,
        )
    else:
        run(
            model_name=args.model,
            epochs=args.epochs,
            n_interp=args.n_interp,
            use_mock=args.mock,
        )
