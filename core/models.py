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


def get_resnet18(num_classes: int = 10) -> nn.Module:
    """ResNet-18 adapted for 1-channel SAR grayscale input."""
    model = tv_models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    def get_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x.flatten(1)

    import types
    model.get_features = types.MethodType(get_features, model)
    return model


def get_model(name: str, num_classes: int = 10) -> nn.Module:
    if name == "smpl":
        return SMPL(num_classes)
    if name == "resnet18":
        return get_resnet18(num_classes)
    raise ValueError(f"Unknown model: {name!r}. Choose 'smpl' or 'resnet18'.")
