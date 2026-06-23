from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable
from torch import Tensor
from torch.utils.data import Dataset


@dataclass
class SARSample:
    image: Tensor   # [1, H, W], float32, [0, 1] normalized
    label: int
    meta: dict = field(default_factory=dict)  # azimuth, elevation, class_name, source, ...


class SARDataset(Dataset):
    """All datasets follow this interface."""

    def __getitem__(self, idx) -> SARSample:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    @property
    def class_names(self) -> list[str]:
        raise NotImplementedError


@runtime_checkable
class Augmentation(Protocol):
    """clutter transfer / contrast balance / boundary blending — unified signature."""

    def __call__(self, image: Tensor, meta: dict) -> Tensor:
        ...


@dataclass
class TrainConfig:
    model_name: str          # "smpl" | "resnet18"
    num_classes: int
    epochs: int = 60
    batch_size: int = 128
    lr: float = 1e-3
    lr_decay_epoch: int = 50
    lr_decay_factor: float = 0.1
    loss_type: str = "lsm"   # "lsm" (label smoothing) | "at" (adversarial training)
    label_smoothing: float = 0.1
    optimizer: str = "adam"  # "adam" | "sgd" — Adam이 SAR 데이터에서 더 빠른 수렴
    seed: int = 0
    num_workers: int = 0
    device: Optional[str] = None  # None → auto-detect


@dataclass
class EvalResult:
    accuracy: float
    confusion_matrix: list[list[int]]
    per_class_accuracy: dict[str, float]
    auroc: Optional[float] = None
    tnr_at_95tpr: Optional[float] = None
