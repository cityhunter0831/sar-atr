import torch
from .interfaces import SARSample, SARDataset


class MockSARDataset(SARDataset):
    """64×64 synthetic SAR data — pipeline / model validation without real MSTAR."""

    def __init__(self, n: int = 200, num_classes: int = 10, image_size: int = 64, seed: int = 0):
        self.n = n
        self.num_classes = num_classes
        self.image_size = image_size
        self._rng = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> SARSample:
        label = idx % self.num_classes
        g = torch.Generator().manual_seed(idx)
        img = torch.randn(1, self.image_size, self.image_size, generator=g) * 0.3 + 0.5
        # Fake "bright scattering" region — discriminative by class
        cx = 16 + (label % 4) * 8
        cy = 16 + (label // 4) * 8
        r = 4
        img[:, cx : cx + r, cy : cy + r] += 1.0
        return SARSample(
            image=img.clamp(0.0, 1.0),
            label=label,
            meta={"class_name": f"class{label}", "source": "mock", "elevation": 17},
        )

    @property
    def class_names(self) -> list[str]:
        return [f"class{i}" for i in range(self.num_classes)]
