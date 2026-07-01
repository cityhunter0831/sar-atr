from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from .interfaces import EvalResult, SARDataset, SARSample, TrainConfig


# ─── Shared helpers ───────────────────────────────────────────────────────────

def _collate(batch: list[SARSample]) -> SimpleNamespace:
    images = torch.stack([s.image for s in batch])
    labels = torch.tensor([s.label for s in batch], dtype=torch.long)
    return SimpleNamespace(image=images, label=labels)


def _make_loader(ds: SARDataset, batch_size: int = 128) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=_collate)


def _device_of(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


# ─── Standard evaluation ──────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_ds: SARDataset,
    config: Optional[TrainConfig] = None,
) -> EvalResult:
    """Accuracy + confusion matrix + per-class accuracy."""
    device = _device_of(model)
    model.eval()

    n_classes = len(test_ds.class_names)
    cm = np.zeros((n_classes, n_classes), dtype=int)

    for batch in _make_loader(test_ds):
        imgs = batch.image.to(device)
        lbls = batch.label.numpy()
        preds = model(imgs).argmax(dim=1).cpu().numpy()
        for true, pred in zip(lbls, preds):
            cm[true][pred] += 1

    per_class: dict[str, float] = {}
    for i, name in enumerate(test_ds.class_names):
        total = cm[i].sum()
        per_class[name] = float(cm[i, i] / total) if total > 0 else 0.0

    accuracy = float(np.diag(cm).sum() / cm.sum())
    return EvalResult(
        accuracy=accuracy,
        confusion_matrix=cm.tolist(),
        per_class_accuracy=per_class,
    )


# ─── Feature extraction ───────────────────────────────────────────────────────

@torch.no_grad()
def _extract_features(model: nn.Module, ds: SARDataset, batch_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Return (features [N, D], labels [N])."""
    device = _device_of(model)
    model.eval()
    all_feats, all_labels = [], []
    for batch in _make_loader(ds, batch_size):
        imgs = batch.image.to(device)
        if hasattr(model, "get_features"):
            feats = model.get_features(imgs)
        else:
            feats = imgs.flatten(1)
        all_feats.append(feats.cpu().numpy())
        all_labels.append(batch.label.numpy())
    return np.concatenate(all_feats), np.concatenate(all_labels)


# ─── OOD scoring methods ──────────────────────────────────────────────────────

def _mahalanobis_scores(
    model: nn.Module, train_ds: SARDataset, test_ds: SARDataset, ood_ds: SARDataset
) -> tuple[np.ndarray, np.ndarray]:
    """Return (id_scores, ood_scores); higher = more in-distribution."""
    train_feats, train_labels = _extract_features(model, train_ds)
    n_classes = len(train_ds.class_names)

    # Class-conditional means and shared (tied) covariance
    means = []
    centered = []
    for c in range(n_classes):
        mask = train_labels == c
        mu = train_feats[mask].mean(0) if mask.any() else np.zeros(train_feats.shape[1])
        means.append(mu)
        if mask.any():
            centered.append(train_feats[mask] - mu)
    centered_all = np.concatenate(centered, 0)
    cov = np.cov(centered_all.T) + 1e-6 * np.eye(train_feats.shape[1])
    cov_inv = np.linalg.pinv(cov)

    def _score(ds: SARDataset) -> np.ndarray:
        feats, _ = _extract_features(model, ds)
        scores = []
        for f in feats:
            dists = [float((f - mu) @ cov_inv @ (f - mu)) for mu in means]
            scores.append(-0.5 * min(dists))  # negative Mahalanobis distance → higher = more ID
        return np.array(scores)

    return _score(test_ds), _score(ood_ds)


def _odin_scores(
    model: nn.Module,
    test_ds: SARDataset,
    ood_ds: SARDataset,
    temperature: float = 1000.0,
    epsilon: float = 0.0014,
) -> tuple[np.ndarray, np.ndarray]:
    """ODIN: temperature scaling + input perturbation (Liang et al. 2018)."""
    device = _device_of(model)
    criterion = nn.CrossEntropyLoss()

    def _score(ds: SARDataset) -> np.ndarray:
        scores = []
        for batch in _make_loader(ds):
            imgs = batch.image.to(device).requires_grad_(True)
            logits = model(imgs) / temperature
            pseudo = logits.argmax(1)
            loss = criterion(logits, pseudo)
            loss.backward()
            perturbed = (imgs - epsilon * imgs.grad.sign()).detach().clamp(0.0, 1.0)
            with torch.no_grad():
                s = F.softmax(model(perturbed), dim=1).max(1).values  # perturbation 후 unscaled
            scores.append(s.cpu().numpy())
        return np.concatenate(scores)

    model.eval()
    return _score(test_ds), _score(ood_ds)


# ─── Metrics ─────────────────────────────────────────────────────────────────

def _auroc_tnr95(id_scores: np.ndarray, ood_scores: np.ndarray) -> tuple[float, float]:
    labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])
    auroc = float(roc_auc_score(labels, scores))

    # TNR @ TPR=0.95: threshold where 95% of ID samples are correctly accepted
    threshold = float(np.percentile(id_scores, 5))
    tnr = float((ood_scores < threshold).mean())
    return auroc, tnr


# ─── Public OOD API ──────────────────────────────────────────────────────────

def evaluate_ood(
    model: nn.Module,
    id_ds: SARDataset,
    ood_ds: SARDataset,
    method: str = "mahalanobis",
    train_ds: Optional[SARDataset] = None,
) -> EvalResult:
    """
    OOD detection evaluation.

    Args:
        id_ds:     In-distribution test set.
        ood_ds:    Out-of-distribution set (e.g. SAR-ship, holdout classes).
        method:    'mahalanobis' | 'odin'.
        train_ds:  Required for Mahalanobis (to fit class means/covariance).

    Returns:
        EvalResult with auroc and tnr_at_95tpr filled; accuracy=-1 sentinel.
    """
    model.eval()
    if method == "mahalanobis":
        assert train_ds is not None, "Mahalanobis requires train_ds to fit class statistics."
        id_scores, ood_scores = _mahalanobis_scores(model, train_ds, id_ds, ood_ds)
    elif method == "odin":
        id_scores, ood_scores = _odin_scores(model, id_ds, ood_ds)
    else:
        raise ValueError(f"Unknown OOD method: {method!r}. Choose 'mahalanobis' or 'odin'.")

    auroc, tnr = _auroc_tnr95(id_scores, ood_scores)
    return EvalResult(
        accuracy=-1.0,
        confusion_matrix=[],
        per_class_accuracy={},
        auroc=auroc,
        tnr_at_95tpr=tnr,
    )
