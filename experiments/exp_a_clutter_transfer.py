"""
Exp A — Clutter Transfer (권승주, Table 4 재현)

4 conditions × 2 models × 3 seeds → 8 rows of Table 4.

Conditions:
    MSTAROR          : train=original, test=original
    TrainOR+TestCT   : train=original, test=clutter-transferred
    TrainCT+TestCT   : train=CT×1,    test=clutter-transferred
    TrainCTx2+TestCT : train=CT×2,    test=clutter-transferred

Data expected at:
    data/clutter_gengzhe/original/<class>/*.png   (MSTAR original chips)
    data/clutter_gengzhe/clutter_bg/*.png         (background clutter chips)

Falls back to MockSARDataset when real data is absent.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import ConcatDataset, Dataset

from augmentation.boundary_blend import clutter_transfer, compute_ssim
from core.interfaces import EvalResult, SARDataset, SARSample, TrainConfig
from core.mock_data import MockSARDataset
from core.models import get_model
from core.train import train_model
from core.evaluate import evaluate

RESULTS_DIR = Path("results/exp_a")
DATA_ROOT = Path("data/clutter_gengzhe")

# gengzhe2015 repo folder names → condition mapping
GENG_FOLDERS = {
    "MSTAROR":          "Original MSTAR Images",
    "TrainOR+TestCT":   "Train_OR_Test_CT",
    "TrainCT+TestCT":   "Train_CT_Test_CT",
    "TrainCTx2+TestCT": "Train_CTx2_Test_CT",
}

# All 5 classes present in gengzhe2015 (lowercase folder names)
TABLE4_CLASSES = ["2s1", "bmp2", "btr70", "t72", "zsu23"]

Condition = Literal["MSTAROR", "TrainOR+TestCT", "TrainCT+TestCT", "TrainCTx2+TestCT"]


# ─── Dataset implementation ───────────────────────────────────────────────────

class MSTARImageFolder(SARDataset):
    """
    Loads PNG/JPG chips from  <root>/<class_name>/*.png
    Returns [1, H, W] float32 normalized to [0, 1].
    """

    def __init__(
        self,
        root: Path,
        class_names: list[str],
        transform=None,
        augmentation=None,
    ):
        self._class_names = class_names
        self._transform = transform
        self._augmentation = augmentation
        self._samples: list[tuple[Path, int]] = []

        for idx, cls in enumerate(class_names):
            cls_dir = root / cls
            if not cls_dir.exists():
                continue
            for p in sorted(cls_dir.iterdir()):
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif"}:
                    self._samples.append((p, idx))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> SARSample:
        path, label = self._samples[idx]
        # Convert via RGB first to avoid palette-mode artifacts
        img = Image.open(path).convert("RGB").convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).unsqueeze(0)  # [1, H, W]

        meta = {
            "class_name": self._class_names[label],
            "source": str(path),
            "elevation": _infer_elevation(path),
        }

        if self._augmentation is not None:
            t = self._augmentation(t, meta)
        return SARSample(image=t, label=label, meta=meta)

    @property
    def class_names(self) -> list[str]:
        return self._class_names


def _infer_elevation(path: Path) -> int:
    """Try to infer elevation from path stem (e.g. 'HB03344.017' → 17)."""
    stem = path.stem
    for part in stem.split("."):
        try:
            val = int(part)
            if val in (15, 17, 30, 45):
                return val
        except ValueError:
            continue
    return 17


class ClutterBgPool:
    """Random background clutter chip sampler."""

    def __init__(self, bg_dir: Path, image_size: int = 128, seed: int = 0):
        self._paths = sorted(bg_dir.glob("*.png")) + sorted(bg_dir.glob("*.jpg"))
        self._size = image_size
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._paths)

    def sample(self) -> Tensor:
        if not self._paths:
            return torch.zeros(1, self._size, self._size)
        p = self._paths[int(self._rng.integers(len(self._paths)))]
        img = Image.open(p).convert("L").resize((self._size, self._size))
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)


class ClutterTransferDataset(SARDataset):
    """
    Wraps a MSTARImageFolder and applies clutter_transfer on-the-fly.
    augment_factor=2 doubles the dataset (one original + one CT per sample).
    """

    def __init__(
        self,
        source_ds: MSTARImageFolder,
        bg_pool: ClutterBgPool,
        method: str = "feather",
        augment_factor: int = 1,
    ):
        self._ds = source_ds
        self._bg = bg_pool
        self._method = method
        self._factor = augment_factor

    def __len__(self) -> int:
        return len(self._ds) * self._factor

    def __getitem__(self, idx: int) -> SARSample:
        sample = self._ds[idx % len(self._ds)]
        bg = self._bg.sample()
        blended = clutter_transfer(sample.image, bg, method=self._method)
        return SARSample(image=blended, label=sample.label, meta={**sample.meta, "aug": "ct"})

    @property
    def class_names(self) -> list[str]:
        return self._ds.class_names


# ─── Condition builder ────────────────────────────────────────────────────────

def _data_available() -> bool:
    orig_dir = DATA_ROOT / "Original MSTAR Images"
    return orig_dir.exists() and any(orig_dir.iterdir())


def _split_dataset(ds: MSTARImageFolder, train_ratio: float = 0.8, seed: int = 0):
    """Deterministic 80/20 split into train/test MSTARImageFolder-compatible datasets."""
    import random as _random
    rng = _random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    n_train = int(len(indices) * train_ratio)

    class _SubsetDataset(SARDataset):
        def __init__(self, base, idxs):
            self._base = base
            self._idxs = idxs
        def __len__(self): return len(self._idxs)
        def __getitem__(self, i): return self._base[self._idxs[i]]
        @property
        def class_names(self): return self._base.class_names

    return _SubsetDataset(ds, indices[:n_train]), _SubsetDataset(ds, indices[n_train:])


def load_condition(
    condition: Condition,
    class_names: list[str] = TABLE4_CLASSES,
    seed: int = 0,
) -> tuple[SARDataset, SARDataset]:
    """
    Returns (train_ds, test_ds) for the given condition.

    gengzhe2015 folder mapping:
      MSTAROR          → train+test split from 'Original MSTAR Images'
      TrainOR+TestCT   → train from original, test from 'Train_OR_Test_CT'
      TrainCT+TestCT   → train from 'Train_CT_Test_CT', test from same (split)
      TrainCTx2+TestCT → train from 'Train_CTx2_Test_CT' (×2 aug), test from CT

    Falls back to MockSARDataset if real data is not found.
    """
    if not _data_available():
        print(f"[Exp A] Real data not found at {DATA_ROOT} — using MockSARDataset.")
        return (
            MockSARDataset(n=200, num_classes=len(class_names), seed=seed),
            MockSARDataset(n=50,  num_classes=len(class_names), seed=seed + 1),
        )

    orig_ds = MSTARImageFolder(DATA_ROOT / "Original MSTAR Images", class_names)
    ct_or_ds = MSTARImageFolder(DATA_ROOT / "Train_OR_Test_CT",   class_names)
    ct1_ds   = MSTARImageFolder(DATA_ROOT / "Train_CT_Test_CT",   class_names)
    ct2_ds   = MSTARImageFolder(DATA_ROOT / "Train_CTx2_Test_CT", class_names)

    orig_train, orig_test = _split_dataset(orig_ds, train_ratio=0.8, seed=seed)
    _, ct_test            = _split_dataset(ct_or_ds, train_ratio=0.8, seed=seed)
    ct1_train, _          = _split_dataset(ct1_ds,   train_ratio=0.8, seed=seed)
    ct2_train, _          = _split_dataset(ct2_ds,   train_ratio=0.8, seed=seed)

    if condition == "MSTAROR":
        return orig_train, orig_test
    elif condition == "TrainOR+TestCT":
        return orig_train, ct_test
    elif condition == "TrainCT+TestCT":
        return ct1_train, ct_test
    elif condition == "TrainCTx2+TestCT":
        return ct2_train, ct_test
    else:
        raise ValueError(f"Unknown condition: {condition!r}")


# ─── Runner ───────────────────────────────────────────────────────────────────

SEEDS = [0, 1, 2]
CONDITIONS: list[Condition] = ["MSTAROR", "TrainOR+TestCT", "TrainCT+TestCT", "TrainCTx2+TestCT"]
MODEL_NAMES = ["smpl", "resnet18"]


class _SyntheticClutterPool:
    """clutter_bg 폴더가 없을 때 쓰는 합성 SAR 클러터 배경 풀.

    SAR amplitude 클러터는 Rayleigh 분포를 따름 → |N(0,1)+jN(0,1)| 로 생성.
    경계 아티팩트(feather 이음새) 정량화에는 실제 배경이 아니어도 무방 —
    측정 대상은 '블렌딩이 타겟 경계를 얼마나 바꾸는가'이지 배경의 정체가 아님.
    """

    def __init__(self, image_size: int = 128, seed: int = 0):
        self._size = image_size
        self._rng = np.random.default_rng(seed)

    def sample(self) -> "Tensor":
        re = self._rng.standard_normal((self._size, self._size))
        im = self._rng.standard_normal((self._size, self._size))
        amp = np.hypot(re, im).astype(np.float32)   # Rayleigh amplitude
        amp = amp / (amp.max() + 1e-8)
        return torch.from_numpy(amp).unsqueeze(0)


def run_boundary_ssim_analysis(
    n_samples: int = 20,
    seed: int = 0,
    save_dir: Path = RESULTS_DIR,
) -> dict:
    """
    우리 팀 개선 #1: clutter transfer 경계 아티팩트를 SSIM으로 정량화.
    원본 chip vs feather_blend 결과의 SSIM을 측정해 경계 품질 검증.

    clutter_bg/ 폴더가 있으면 실제 배경 사용, 없으면 합성 SAR 클러터로 폴백.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    if not _data_available():
        print("[SSIM] Real data not available — skipping.")
        return {}

    bg_dir = DATA_ROOT / "clutter_bg"
    if bg_dir.exists() and (sorted(bg_dir.glob("*.png")) or sorted(bg_dir.glob("*.jpg"))):
        bg_pool: object = ClutterBgPool(bg_dir, seed=seed)
        bg_source = "real"
        print(f"[SSIM] Using real clutter backgrounds from {bg_dir}")
    else:
        bg_pool = _SyntheticClutterPool(seed=seed)
        bg_source = "synthetic"
        print(f"[SSIM] No clutter_bg dir — using synthetic SAR-speckle backgrounds.")

    orig_ds = MSTARImageFolder(DATA_ROOT / "Original MSTAR Images", TABLE4_CLASSES)
    if len(orig_ds) == 0:
        print("[SSIM] 'Original MSTAR Images' 비어있음 — skipping.")
        return {}

    import random as _rng
    rng = _rng.Random(seed)
    indices = rng.sample(range(len(orig_ds)), min(n_samples, len(orig_ds)))

    ssim_scores = []
    for i in indices:
        sample = orig_ds[i]
        bg = bg_pool.sample()
        blended = clutter_transfer(sample.image, bg, method="feather")
        score = compute_ssim(sample.image, blended)
        ssim_scores.append(score)

    mean_ssim = float(np.mean(ssim_scores)) if ssim_scores else 0.0
    std_ssim  = float(np.std(ssim_scores))  if ssim_scores else 0.0
    print(f"[SSIM] Boundary SSIM (original vs feather-blended, bg={bg_source}): "
          f"{mean_ssim:.4f} ± {std_ssim:.4f}  (n={len(ssim_scores)})")

    result = {"mean_ssim": mean_ssim, "std_ssim": std_ssim,
              "n": len(ssim_scores), "bg_source": bg_source}
    import json as _json
    with open(save_dir / "boundary_ssim.json", "w") as f:
        _json.dump(result, f, indent=2)
    return result


def run_all(epochs: int = 60, save_dir: Path = RESULTS_DIR) -> dict:
    """
    Run 4 conditions × 2 models × 3 seeds and return aggregated results.
    Saves metrics.json and a markdown summary to save_dir.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {}

    for model_name in MODEL_NAMES:
        results[model_name] = {}
        for condition in CONDITIONS:
            accs = []
            for seed in SEEDS:
                train_ds, test_ds = load_condition(condition, seed=seed)
                model = get_model(model_name, num_classes=len(TABLE4_CLASSES))
                config = TrainConfig(
                    model_name=model_name,
                    num_classes=len(TABLE4_CLASSES),
                    epochs=epochs,
                    seed=seed,
                )
                model, _ = train_model(model, train_ds, test_ds, config)
                result = evaluate(model, test_ds)
                accs.append(result.accuracy * 100)
                print(
                    f"[{model_name}] {condition} seed={seed} → {result.accuracy * 100:.1f}%"
                )

            mean_acc = float(np.mean(accs))
            std_acc = float(np.std(accs))
            results[model_name][condition] = {"mean": mean_acc, "std": std_acc, "runs": accs}
            print(f"  ↳ {mean_acc:.1f} ± {std_acc:.1f}%\n")

    # Save
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    _print_table(results)
    return results


def _print_table(results: dict):
    header = f"{'Condition':<22}" + "".join(f"  {m:<18}" for m in MODEL_NAMES)
    print("\n" + "=" * len(header))
    print("Exp A — Clutter Transfer (Table 4 re-production)")
    print(header)
    print("-" * len(header))
    for cond in CONDITIONS:
        row = f"{cond:<22}"
        for m in MODEL_NAMES:
            d = results[m].get(cond, {})
            row += f"  {d.get('mean', 0):.1f}±{d.get('std', 0):.1f}%       "
        print(row)
    print("=" * len(header))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--ssim", action="store_true", help="SSIM 경계 아티팩트 분석만 실행")
    args = parser.parse_args()
    if args.ssim:
        run_boundary_ssim_analysis()
    else:
        run_all(epochs=args.epochs)
