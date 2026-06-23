import random
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .interfaces import EvalResult, SARDataset, SARSample, TrainConfig


# ─── Collate ─────────────────────────────────────────────────────────────────

def _collate(batch: list[SARSample]) -> SimpleNamespace:
    """Stack SARSample list into batched tensors."""
    images = torch.stack([s.image for s in batch])
    labels = torch.tensor([s.label for s in batch], dtype=torch.long)
    metas = [s.meta for s in batch]
    return SimpleNamespace(image=images, label=labels, meta=metas)


# ─── Loss functions ───────────────────────────────────────────────────────────

class _LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n = logits.size(-1)
        log_p = torch.log_softmax(logits, dim=-1)
        nll = -log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
        smooth = -log_p.mean(dim=-1)
        return ((1 - self.smoothing) * nll + self.smoothing * smooth).mean()


def _fgsm_perturb(model, images, labels, criterion, eps: float = 2 / 255):
    """Single-step FGSM perturbation for adversarial training."""
    images = images.clone().requires_grad_(True)
    loss = criterion(model(images), labels)
    loss.backward()
    return (images + eps * images.grad.sign()).detach().clamp(0.0, 1.0)


# ─── Seed ────────────────────────────────────────────────────────────────────

def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─── Public API ──────────────────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    train_ds: SARDataset,
    val_ds: SARDataset,
    config: TrainConfig,
) -> tuple[nn.Module, EvalResult]:
    """
    Train model and return (trained_model, val_EvalResult).

    Follows TrainConfig defaults from spec:
      - SGD + momentum=0.9, weight_decay=1e-4
      - LR drops by lr_decay_factor at lr_decay_epoch
      - loss_type='lsm' → label smoothing CE; 'at' → FGSM adversarial training
    """
    _set_seed(config.seed)

    device = torch.device(
        config.device if config.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = model.to(device)

    loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=_collate,
        drop_last=False,
    )

    if config.loss_type == "lsm":
        criterion = _LabelSmoothingCE(config.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    if getattr(config, "optimizer", "adam") == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=1e-4)
    else:
        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-4)

    for epoch in tqdm(range(config.epochs), desc=f"[{config.model_name}] train", leave=False):
        # Step LR (Adam에서도 decay 적용)
        lr = config.lr if epoch < config.lr_decay_epoch else config.lr * config.lr_decay_factor
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        model.train()
        for batch in loader:
            imgs = batch.image.to(device)
            lbls = batch.label.to(device)

            if config.loss_type == "at":
                imgs = _fgsm_perturb(model, imgs, lbls, criterion)

            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()

    from .evaluate import evaluate
    result = evaluate(model, val_ds, config=config)
    return model, result
