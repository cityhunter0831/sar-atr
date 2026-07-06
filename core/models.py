import torch.nn as nn
import torchvision.models as tv_models


class SMPL(nn.Module):
    """Lightweight CNN from Geng et al. 2023 (Table 3 headline model)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5, padding=2), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 5, padding=2), nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

    def get_features(self, x):
        """Penultimate embedding for OOD / Grad-CAM."""
        feats = self.features(x)
        pooled = nn.functional.adaptive_avg_pool2d(feats, 1)
        return pooled.flatten(1)


class ResNet18SAR(nn.Module):
    """ResNet-18 for 1-channel SAR input. Wrapper class for pickle safety."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        base = tv_models.resnet18(weights=None)
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        base.fc = nn.Linear(base.fc.in_features, num_classes)
        self._base = base

    def forward(self, x):
        return self._base(x)

    def get_features(self, x):
        b = self._base
        x = b.relu(b.bn1(b.conv1(x)))
        x = b.maxpool(x)
        x = b.layer1(x)
        x = b.layer2(x)
        x = b.layer3(x)
        x = b.layer4(x)
        return b.avgpool(x).flatten(1)


def get_resnet18(num_classes: int = 10) -> nn.Module:
    """ResNet-18 adapted for 1-channel SAR grayscale input."""
    return ResNet18SAR(num_classes)


def get_model(name: str, num_classes: int = 10) -> nn.Module:
    if name == "smpl":
        return SMPL(num_classes)
    if name == "resnet18":
        return get_resnet18(num_classes)
    raise ValueError(f"Unknown model: {name!r}. Choose 'smpl' or 'resnet18'.")
