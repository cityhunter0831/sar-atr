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

from augmentation.boundary_blend import clutter_transfer
from core.interfaces import EvalResult, SARDataset, SARSample, TrainConfig
from core.mock_data import MockSARDataset
from core.models import get_model
from core.train import train_model
from core.evaluate import evaluate

RESULTS_DIR = Path("results/exp_a")
DATA_ROOT = Path("data/clutter_gengzhe")

# Classes used in Table 4 (subset available in MSTAR Targets package)
TABLE4_CLASSES = ["BMP2", "BTR70", "T72"]

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
        img = Image.open(path).convert("L")
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
    orig_dir = DATA_ROOT / "original"
    return orig_dir.exists() and any(True for _ in orig_dir.iterdir())


def load_condition(
    condition: Condition,
    class_names: list[str] = TABLE4_CLASSES,
    seed: int = 0,
) -> tuple[SARDataset, SARDataset]:
    """
    Returns (train_ds, test_ds) for the given condition.
    Falls back to MockSARDataset if real data is not found.
    """
    if not _data_available():
        print(f"[Exp A] Real data not found at {DATA_ROOT} — using MockSARDataset.")
        n = 200
        return (
            MockSARDataset(n=n, num_classes=len(class_names), seed=seed),
            MockSARDataset(n=50, num_classes=len(class_names), seed=seed + 1),
        )

    orig_train = MSTARImageFolder(DATA_ROOT / "original" / "train", class_names)
    orig_test = MSTARImageFolder(DATA_ROOT / "original" / "test", class_names)
    bg_pool = ClutterBgPool(DATA_ROOT / "clutter_bg", seed=seed)
    ct_test = ClutterTransferDataset(orig_test, bg_pool, augment_factor=1)

    if condition == "MSTAROR":
        return orig_train, orig_test
    elif condition == "TrainOR+TestCT":
        return orig_train, ct_test
    elif condition == "TrainCT+TestCT":
        ct_train = ClutterTransferDataset(orig_train, bg_pool, augment_factor=1)
        return ct_train, ct_test
    elif condition == "TrainCTx2+TestCT":
        ct_train_x2 = ClutterTransferDataset(orig_train, bg_pool, augment_factor=2)
        return ct_train_x2, ct_test
    else:
        raise ValueError(f"Unknown condition: {condition!r}")


# ─── Runner ───────────────────────────────────────────────────────────────────

SEEDS = [0, 1, 2]
CONDITIONS: list[Condition] = ["MSTAROR", "TrainOR+TestCT", "TrainCT+TestCT", "TrainCTx2+TestCT"]
MODEL_NAMES = ["smpl", "resnet18"]


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
    args = parser.parse_args()
    run_all(epochs=args.epochs)
