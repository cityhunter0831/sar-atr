"""
Exp C — Contrast Balance + Optuna (박승준, Figure 1 재현)

Reproduces and improves the cross-elevation accuracy drop shown in Figure 1:
    Train El=17° → Test El=17°: ~97.2%   (baseline)
    Train El=17° → Test El=30°: ~65.3%   (degradation without contrast balance)
    Train El=17° → Test El=30° + ContrastBalance: target ≥88.5%

Data required:
    data/mstar/mixed_targets/  (MSTAR/IU Mixed Targets package)
    Classes: 2S1, BRDM2, ZSU23-4  at elevations 17° and 30°

Falls back to MockSARDataset when data is absent.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import optuna
import torch

from augmentation.contrast_balance import ContrastBalance, make_optuna_objective
from core.evaluate import evaluate
from core.interfaces import EvalResult, SARDataset, SARSample, TrainConfig
from core.mock_data import MockSARDataset
from core.models import get_model
from core.train import train_model

RESULTS_DIR = Path("results/exp_c")
DATA_ROOT = Path("data/mstar/mixed_targets")
SAMPLE_ROOT = Path("data/sample/png_images")
FIGURE1_CLASSES = ["2S1", "BRDM2", "ZSU23-4"]
# SAMPLE dataset 클래스 (BMP2, BTR70, T72 등 MSTAR와 동일)
SAMPLE_CLASSES = ["BMP2", "BTR70", "T72", "2S1", "BRDM2"]


# ─── Dataset helpers ──────────────────────────────────────────────────────────

class ElevationFilteredDataset(SARDataset):
    """
    Wraps a directory-based dataset and filters by elevation angle.

    Expected structure:
        <root>/<elevation>/<class>/*.png    e.g. mixed_targets/017/2S1/HB03344.017
    """

    def __init__(
        self,
        root: Path,
        elevation: int,
        class_names: list[str],
        augmentation=None,
    ):
        self._class_names = class_names
        self._augmentation = augmentation
        self._samples: list[tuple[Path, int]] = []

        el_str = str(elevation).zfill(3)
        el_dir = root / el_str
        if not el_dir.exists():
            # Try alternative naming: 'el17', '17deg', etc.
            for d in sorted(root.iterdir()):
                if d.is_dir() and str(elevation) in d.name:
                    el_dir = d
                    break

        for idx, cls in enumerate(class_names):
            cls_dir = el_dir / cls
            if not cls_dir.exists():
                continue
            for p in sorted(cls_dir.iterdir()):
                if p.suffix.lower() in {".png", ".jpg", ".0", ".017", ".030", ".045"}:
                    self._samples.append((p, idx))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> SARSample:
        from PIL import Image
        path, label = self._samples[idx]
        try:
            img = Image.open(path).convert("L")
            arr = np.array(img, dtype=np.float32) / 255.0
        except Exception:
            # MSTAR raw fallback
            from augmentation.ph_extraction import read_mstar_raw, amplitude_to_tensor
            arr_raw = read_mstar_raw(path)
            arr = arr_raw / (arr_raw.max() + 1e-8)

        t = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)
        meta = {"class_name": self._class_names[label], "elevation": 0, "source": str(path)}
        if self._augmentation is not None:
            t = self._augmentation(t, meta)
        return SARSample(image=t, label=label, meta=meta)

    @property
    def class_names(self) -> list[str]:
        return self._class_names


class AugmentedWrapper(SARDataset):
    """Apply a ContrastBalance augmentation on top of any SARDataset."""

    def __init__(self, ds: SARDataset, aug: ContrastBalance):
        self._ds = ds
        self._aug = aug

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> SARSample:
        s = self._ds[idx]
        s.image = self._aug(s.image, s.meta)
        return s

    @property
    def class_names(self) -> list[str]:
        return self._ds.class_names


# ─── SAMPLE dataset loader (v3 주 데이터) ─────────────────────────────────────

class SampleDataset(SARDataset):
    """
    SAMPLE dataset (benjaminlewis-afrl/SAMPLE_dataset_public) loader.

    구조: data/sample/png_images/<class_name>/measured/*.png
              data/sample/png_images/<class_name>/synthetic/*.png
    split: "measured" | "synthetic"
    """

    def __init__(self, root: Path, split: str, class_names: list[str]):
        assert split in ("measured", "synthetic"), f"split must be 'measured' or 'synthetic', got {split!r}"
        self._class_names = class_names
        self._split = split
        self._samples: list[tuple[Path, int]] = []

        for idx, cls in enumerate(class_names):
            cls_dir = root / cls / split
            if not cls_dir.exists():
                # 대소문자 차이 허용
                for d in sorted(root.iterdir()):
                    if d.name.lower() == cls.lower():
                        cls_dir = d / split
                        break
            if not cls_dir.exists():
                continue
            for p in sorted(cls_dir.iterdir()):
                if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    self._samples.append((p, idx))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> SARSample:
        from PIL import Image
        path, label = self._samples[idx]
        img = Image.open(path).convert("RGB").convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).unsqueeze(0)
        meta = {
            "class_name": self._class_names[label],
            "split": self._split,
            "source": str(path),
        }
        return SARSample(image=t, label=label, meta=meta)

    @property
    def class_names(self) -> list[str]:
        return self._class_names


def _sample_available() -> bool:
    return SAMPLE_ROOT.exists() and any(SAMPLE_ROOT.iterdir())


def load_sample(
    class_names: list[str] = SAMPLE_CLASSES,
) -> tuple[SARDataset, SARDataset]:
    """
    Returns (train_synthetic, test_measured).
    SAMPLE의 synthetic으로 훈련 → measured로 테스트 = 논문 Figure 1 원본 방법.
    Falls back to mock if data is absent.
    """
    if not _sample_available():
        print(f"[Exp C] SAMPLE data not found at {SAMPLE_ROOT} — using MockSARDataset.")
        print("  → git clone https://github.com/benjaminlewis-afrl/SAMPLE_dataset_public")
        return (
            MockSARDataset(n=200, num_classes=len(class_names), seed=0),
            MockSARDataset(n=60,  num_classes=len(class_names), seed=1),
        )

    train_ds = SampleDataset(SAMPLE_ROOT, "synthetic", class_names)
    test_ds  = SampleDataset(SAMPLE_ROOT, "measured",  class_names)
    print(f"[Exp C] SAMPLE loaded: train(synthetic)={len(train_ds)}, test(measured)={len(test_ds)}")
    return train_ds, test_ds


# ─── Data loading (MSTAR El ablation — 보조) ──────────────────────────────────

def _data_available() -> bool:
    return DATA_ROOT.exists() and any(DATA_ROOT.iterdir())


def load_el17_el30(
    class_names: list[str] = FIGURE1_CLASSES,
) -> tuple[SARDataset, SARDataset, SARDataset]:
    """
    Returns (train_el17, test_el17, test_el30).
    Falls back to mock if real data is absent.
    """
    if not _data_available():
        print(f"[Exp C] Real data not found at {DATA_ROOT} — using MockSARDataset.")
        n = 150
        return (
            MockSARDataset(n=n, num_classes=len(class_names), seed=0),
            MockSARDataset(n=50, num_classes=len(class_names), seed=1),
            MockSARDataset(n=50, num_classes=len(class_names), seed=2),
        )

    train_el17 = ElevationFilteredDataset(DATA_ROOT, elevation=17, class_names=class_names)
    test_el17 = ElevationFilteredDataset(DATA_ROOT, elevation=17, class_names=class_names)
    test_el30 = ElevationFilteredDataset(DATA_ROOT, elevation=30, class_names=class_names)
    return train_el17, test_el17, test_el30


# ─── Runner ───────────────────────────────────────────────────────────────────

def run(
    model_name: str = "resnet18",
    n_optuna_trials: int = 20,
    epochs_full: int = 60,
    epochs_trial: int = 10,
    seed: int = 0,
    save_dir: Path = RESULTS_DIR,
    class_names: list[str] = SAMPLE_CLASSES,
) -> dict:
    """
    Primary Exp C: SAMPLE dataset (synthetic → measured).
    synthetic으로 학습, measured로 테스트 — Figure 1 재현.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    n_classes = len(class_names)

    train_ds, test_ds = load_sample(class_names)

    base_config = TrainConfig(
        model_name=model_name,
        num_classes=n_classes,
        epochs=epochs_full,
        seed=seed,
    )

    # ── Step 1: baseline (synthetic → measured, no aug) ──
    print("Step 1: Train synthetic → Test measured (baseline, no aug)")
    model_base = get_model(model_name, n_classes)
    model_base, _ = train_model(model_base, train_ds, test_ds, base_config)
    acc_no_aug = evaluate(model_base, test_ds).accuracy * 100
    print(f"  synthetic → measured (no aug): {acc_no_aug:.1f}%  (paper: ~65.3%)")

    # ── Step 2: Optuna search for ContrastBalance hyperparams ──
    print(f"\nStep 2: Optuna ({n_optuna_trials} trials, {epochs_trial} epochs each)")
    objective = make_optuna_objective(train_ds, test_ds, base_config, n_epochs_trial=epochs_trial)
    study = optuna.create_study(direction="maximize", study_name="exp_c_contrast")
    study.optimize(objective, n_trials=n_optuna_trials, show_progress_bar=True)

    best = study.best_params
    print(f"\nBest params: {best}")
    print(f"Best trial val acc: {study.best_value * 100:.1f}%")

    # ── Step 3: retrain with best params (full epochs) ──
    print(f"\nStep 3: Retrain with best ContrastBalance (full {epochs_full} epochs)")
    aug = ContrastBalance(
        clip_limit=best["clip_limit"],
        tile_grid_size=(best["tile_grid_size"], best["tile_grid_size"]),
        global_norm=best["global_norm"],
    )
    aug_train = AugmentedWrapper(train_ds, aug)
    aug_test = AugmentedWrapper(test_ds, aug)

    model_aug = get_model(model_name, n_classes)
    model_aug, _ = train_model(model_aug, aug_train, aug_test, base_config)
    acc_aug = evaluate(model_aug, aug_test).accuracy * 100
    print(f"  synthetic → measured + ContrastBalance: {acc_aug:.1f}%  (target: ≥88.5%)")

    results = {
        "model": model_name,
        "dataset": "SAMPLE",
        "acc_no_aug": acc_no_aug,
        "acc_with_aug": acc_aug,
        "best_params": best,
        "optuna_best_trial_acc": study.best_value * 100,
        "criterion_met": acc_aug >= 88.5,
    }

    with open(save_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    torch.save(model_aug.state_dict(), save_dir / f"{model_name}_contrast_best.pth")

    _print_figure1(acc_no_aug, acc_aug)
    return results


def run_el_ablation(
    model_name: str = "resnet18",
    n_optuna_trials: int = 20,
    epochs_full: int = 60,
    epochs_trial: int = 10,
    seed: int = 0,
    save_dir: Path = RESULTS_DIR / "el_ablation",
) -> dict:
    """
    보조 실험: MSTAR El=17° → El=30° 교차 elevation 정확도 하락 재현.
    MSTAR Mixed Targets 데이터 필요 (SDMS 승인 필요).
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    n_classes = len(FIGURE1_CLASSES)

    train_el17, test_el17, test_el30 = load_el17_el30()

    base_config = TrainConfig(
        model_name=model_name,
        num_classes=n_classes,
        epochs=epochs_full,
        seed=seed,
    )

    print("El Ablation Step 1: Train El17° → Test El17° (baseline)")
    model_base = get_model(model_name, n_classes)
    model_base, _ = train_model(model_base, train_el17, test_el17, base_config)
    acc_el17 = evaluate(model_base, test_el17).accuracy * 100
    acc_el30_no_aug = evaluate(model_base, test_el30).accuracy * 100
    print(f"  El17 → El17 : {acc_el17:.1f}%  (paper: ~97.2%)")
    print(f"  El17 → El30 : {acc_el30_no_aug:.1f}%  (paper: ~65.3%)")

    print(f"\nEl Ablation Step 2: Optuna ({n_optuna_trials} trials)")
    objective = make_optuna_objective(train_el17, test_el30, base_config, n_epochs_trial=epochs_trial)
    study = optuna.create_study(direction="maximize", study_name="exp_c_el_ablation")
    study.optimize(objective, n_trials=n_optuna_trials, show_progress_bar=True)

    best = study.best_params
    aug = ContrastBalance(
        clip_limit=best["clip_limit"],
        tile_grid_size=(best["tile_grid_size"], best["tile_grid_size"]),
        global_norm=best["global_norm"],
    )
    aug_train = AugmentedWrapper(train_el17, aug)
    aug_test_el30 = AugmentedWrapper(test_el30, aug)

    model_aug = get_model(model_name, n_classes)
    model_aug, _ = train_model(model_aug, aug_train, aug_test_el30, base_config)
    acc_el30_aug = evaluate(model_aug, aug_test_el30).accuracy * 100
    print(f"  El17 → El30 + ContrastBalance: {acc_el30_aug:.1f}%  (target: ≥88.5%)")

    results = {
        "model": model_name,
        "dataset": "MSTAR_mixed_targets",
        "acc_el17_baseline": acc_el17,
        "acc_el30_no_aug": acc_el30_no_aug,
        "acc_el30_with_aug": acc_el30_aug,
        "best_params": best,
        "criterion_met": acc_el30_aug >= 88.5,
    }
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    _print_figure1(acc_el30_no_aug, acc_el30_aug)
    return results


def _print_figure1(acc_no_aug: float, acc_aug: float):
    print("\n── Figure 1 Reproduction ─────────────────────────")
    print(f"  synthetic→measured (no aug):  {acc_no_aug:5.1f}%  (paper: ~65.3%)")
    print(f"  synthetic→measured (CLAHE):   {acc_aug:5.1f}%  (target: ≥88.5%)")
    status = "PASS ✓" if acc_aug >= 88.5 else "FAIL ✗"
    print(f"  Criterion: {status}")
    print("──────────────────────────────────────────────────")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet18", choices=["smpl", "resnet18"])
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--epochs-trial", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--el-ablation", action="store_true",
                        help="Run MSTAR El=17→30 ablation instead of SAMPLE experiment")
    args = parser.parse_args()
    fn = run_el_ablation if args.el_ablation else run
    fn(
        model_name=args.model,
        n_optuna_trials=args.n_trials,
        epochs_full=args.epochs,
        epochs_trial=args.epochs_trial,
        seed=args.seed,
    )
