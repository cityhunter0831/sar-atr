"""
Contrast balance / preservation for cross-elevation generalization (박승준 — Exp C).

SAR image brightness changes significantly between elevation angles (El 17° → El 30°).
CLAHE normalizes local contrast while preserving the structural information
that is meaningful for ATR.

Reference: Figure 1 in Geng et al. 2023 (97.2% → 65.3% degradation; target: ≥88.5%)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from core.interfaces import Augmentation


class ContrastBalance(nn.Module):
    """
    CLAHE-based contrast normalizer — implements the Augmentation protocol.

    Args:
        clip_limit:      CLAHE clip limit (Optuna search range: [0.5, 4.0]).
        tile_grid_size:  CLAHE tile grid (e.g. (4,4), (8,8)).
        global_norm:     If True, additionally z-score normalize the whole chip.
    """

    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: tuple[int, int] = (8, 8),
        global_norm: bool = False,
    ):
        super().__init__()
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.global_norm = global_norm

    def forward(self, image: Tensor, meta: dict | None = None) -> Tensor:
        return clahe_equalize(image, self.clip_limit, self.tile_grid_size, self.global_norm)

    # Support Augmentation protocol as a plain callable
    def __call__(self, image: Tensor, meta: dict) -> Tensor:
        return self.forward(image, meta)


def clahe_equalize(
    image: Tensor,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
    global_norm: bool = False,
) -> Tensor:
    """
    Apply CLAHE to a single SAR image tensor.

    Args:
        image:          [1, H, W] float32 in [0, 1].
        clip_limit:     CLAHE clip threshold.
        tile_grid_size: Tile grid dimensions.
        global_norm:    Optionally z-score normalize after CLAHE.

    Returns:
        [1, H, W] float32 in [0, 1].
    """
    try:
        import cv2
        return _clahe_cv2(image, clip_limit, tile_grid_size, global_norm)
    except ImportError:
        return _clahe_numpy(image, global_norm)


def _clahe_cv2(
    image: Tensor,
    clip_limit: float,
    tile_grid_size: tuple[int, int],
    global_norm: bool,
) -> Tensor:
    import cv2

    arr = (image.squeeze(0).numpy() * 255).clip(0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    equalized = clahe.apply(arr).astype(np.float32) / 255.0

    if global_norm:
        mu, sigma = equalized.mean(), equalized.std()
        equalized = (equalized - mu) / (sigma + 1e-8)
        equalized = (equalized - equalized.min()) / (equalized.max() - equalized.min() + 1e-8)

    return torch.from_numpy(equalized).unsqueeze(0)


def _clahe_numpy(image: Tensor, global_norm: bool) -> Tensor:
    """Histogram equalization fallback (no OpenCV)."""
    arr = image.squeeze(0).numpy().astype(np.float32)
    flat = (arr * 255).clip(0, 255).astype(np.uint8).flatten()
    hist, _ = np.histogram(flat, bins=256, range=(0, 255))
    cdf = hist.cumsum()
    cdf_min = cdf[cdf > 0].min()
    n = flat.size
    lut = np.round(((cdf - cdf_min) / (n - cdf_min)) * 255).astype(np.uint8)
    equalized = lut[(arr * 255).clip(0, 255).astype(np.uint8)].astype(np.float32) / 255.0

    if global_norm:
        mu, sigma = equalized.mean(), equalized.std()
        equalized = (equalized - mu) / (sigma + 1e-8)
        equalized = (equalized - equalized.min()) / (equalized.max() - equalized.min() + 1e-8)

    return torch.from_numpy(equalized).unsqueeze(0)


# ─── Optuna objective helper ──────────────────────────────────────────────────

def make_optuna_objective(train_ds, val_ds, base_config, n_epochs_trial: int = 10):
    """
    Returns an Optuna objective function that searches CLAHE hyperparameters.

    Searches:
        clip_limit     : float in [0.5, 4.0]
        tile_grid_size : categorical [4, 8, 16]
        global_norm    : bool

    Usage:
        study = optuna.create_study(direction='maximize')
        study.optimize(make_optuna_objective(...), n_trials=20)
    """
    from core.train import train_model
    from core.models import get_model
    from core.interfaces import TrainConfig
    import copy

    class _AugmentedDataset:
        def __init__(self, ds, aug):
            self._ds = ds
            self._aug = aug

        def __len__(self):
            return len(self._ds)

        def __getitem__(self, idx):
            sample = self._ds[idx]
            sample.image = self._aug(sample.image, sample.meta)
            return sample

        @property
        def class_names(self):
            return self._ds.class_names

    def objective(trial):
        clip = trial.suggest_float("clip_limit", 0.5, 4.0)
        tile = trial.suggest_categorical("tile_grid_size", [4, 8, 16])
        gnorm = trial.suggest_categorical("global_norm", [True, False])

        aug = ContrastBalance(clip_limit=clip, tile_grid_size=(tile, tile), global_norm=gnorm)

        aug_train = _AugmentedDataset(train_ds, aug)
        aug_val = _AugmentedDataset(val_ds, aug)

        cfg = copy.deepcopy(base_config)
        cfg.epochs = n_epochs_trial

        model = get_model(cfg.model_name, cfg.num_classes)
        _, result = train_model(model, aug_train, aug_val, cfg)
        return result.accuracy

    return objective
