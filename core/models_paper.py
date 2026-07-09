"""
Exp C 논문 충실 재현 전용 모델 4종 — core/models.py(SMPL/ResNet18SAR)는 건드리지 않는다.

Exp A/B/D는 core/models.py의 기존 SMPL/ResNet18SAR로 이미 검증된 체크포인트·수치가
쌓여 있어, 여기에 Dropout 등 구조 변경을 섞으면 그 결과들이 흔들린다. 그래서 논문
Table 6(Exp C)이 요구하는 "Gaus=Drop, lblsm" 레시피가 필요한 4개 모델은 별도
모듈로 분리했다.

참고 구현체 (논문이 "네트워크 모델 구현 참고"로 직접 지목):
    https://github.com/inkawhich/synthetic-to-measured-sar/tree/master/models
    - sample_model.py   → SMPLPaper
    - a_convnet.py       → AConvNetPaper
    - resnet.py          → ResNet18Paper
    - heiligers_model.py → HeiligersPaper

레시피 (원문 Section 4.3, 페이지 이미지로 확정):
    SMPL, AConvNet   : Gaus=Drop=0.3, lblsm=0.08
    ResNet18, Heiligers CNN : Gaus=Drop=0.4, lblsm=0.1

⚠️ AConvNet의 정확한 conv 커널/패딩과 Heiligers의 Dropout 유무는 참고 구현체에서
100% 확인하지 못했다(웹 조사 요약 기반) — 표준 A-ConvNet 구조와 논문 레시피를
최대한 준수해 근사했다. Colab에서 실제 학습이 도는지(shape 에러 여부)로 검증 필요.
"""
from __future__ import annotations

import torch
import torch.nn as nn


PAPER_RECIPE: dict[str, dict[str, float]] = {
    "smpl_paper":      {"gaus": 0.3, "drop": 0.3, "lblsm": 0.08},
    "aconv_paper":      {"gaus": 0.3, "drop": 0.3, "lblsm": 0.08},
    "resnet18_paper":  {"gaus": 0.4, "drop": 0.4, "lblsm": 0.1},
    "heiligers_paper": {"gaus": 0.4, "drop": 0.4, "lblsm": 0.1},
}


class SMPLPaper(nn.Module):
    """참고: inkawhich/synthetic-to-measured-sar/models/sample_model.py.
    GAP 없이 flatten → FC(→1000→500→250→classes), FC 사이마다 Dropout(drop_prob).

    FC 입력 크기가 해상도에 따라 달라지므로 __init__에서 더미 forward로 즉시
    materialize한다 — core/train.py가 optimizer를 첫 forward 이전에 생성하기
    때문에, forward 시점까지 classifier를 안 만들면 그 파라미터가 optimizer에서
    누락돼 전혀 학습되지 않는다."""

    def __init__(self, num_classes: int = 10, drop_prob: float = 0.3, input_size: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_size, input_size)
            in_features = self.features(dummy).flatten(1).shape[1]
        self.classifier = nn.Sequential(
            nn.Dropout(drop_prob), nn.Linear(in_features, 1000), nn.ReLU(),
            nn.Dropout(drop_prob), nn.Linear(1000, 500), nn.ReLU(),
            nn.Dropout(drop_prob), nn.Linear(500, 250), nn.ReLU(),
            nn.Dropout(drop_prob), nn.Linear(250, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))

    def get_features(self, x):
        feats = self.features(x)
        return nn.functional.adaptive_avg_pool2d(feats, 1).flatten(1)


class AConvNetPaper(nn.Module):
    """참고: a_convnet.py — 순수 컨볼루션, 마지막은 GAP 없이 1x1 conv로 분류.
    표준 A-ConvNet(Chen et al. 2016) 채널 진행을 따름."""

    def __init__(self, num_classes: int = 10, drop_prob: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 6, padding=0), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 5, padding=0), nn.ReLU(),
            nn.Dropout2d(drop_prob),
        )
        self.classifier = nn.AdaptiveAvgPool2d(1)  # 입력 해상도 무관하게 안전한 폴백
        self.out = nn.Conv2d(128, num_classes, 1)

    def forward(self, x):
        feats = self.features(x)
        pooled = self.classifier(feats)
        logits = self.out(pooled)
        return logits.flatten(1)

    def get_features(self, x):
        feats = self.features(x)
        return nn.functional.adaptive_avg_pool2d(feats, 1).flatten(1)


class HeiligersPaper(nn.Module):
    """참고: heiligers_model.py. 원 구현엔 Dropout이 없으나, 논문이 Heiligers에도
    Gaus=Drop=0.4를 명시하므로 마지막 FC 앞에 Dropout을 추가해 레시피를 준수한다."""

    def __init__(self, num_classes: int = 10, drop_prob: float = 0.4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 18, 5, padding=0), nn.ReLU(),
            nn.Conv2d(18, 18, 5, padding=0), nn.ReLU(),
            nn.MaxPool2d(6, 6),
            nn.Conv2d(18, 36, 5, padding=0), nn.ReLU(),
            nn.MaxPool2d(4, 4),
        )
        self.conv_final = nn.Conv2d(36, 120, 4, padding=0)  # in_channels=36 고정, 입력 해상도와 무관
        self.gap = nn.AdaptiveAvgPool2d(1)  # 원 구현은 flatten(120*1*1)이지만 GAP로 해상도 의존성 제거
        self.drop = nn.Dropout(drop_prob)
        self.fc = nn.Linear(120, num_classes)

    def forward(self, x):
        feats = torch.relu(self.conv_final(self.features(x)))
        pooled = self.gap(feats).flatten(1)
        return self.fc(self.drop(pooled))

    def get_features(self, x):
        feats = torch.relu(self.conv_final(self.features(x)))
        return self.gap(feats).flatten(1)


class ResNet18Paper(nn.Module):
    """참고: resnet.py — torchvision resnet18 backbone + layer1~3 사이·GAP 이후
    Dropout2d/Dropout 삽입. 1채널 SAR 입력에 맞게 conv1 교체."""

    def __init__(self, num_classes: int = 10, drop_prob: float = 0.4):
        super().__init__()
        import torchvision.models as tv_models
        base = tv_models.resnet18(weights=None)
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        base.fc = nn.Linear(base.fc.in_features, num_classes)
        self._base = base
        self.drop_prob = drop_prob
        self.drop2d = nn.Dropout2d(drop_prob)
        self.drop1d = nn.Dropout(drop_prob)

    def forward(self, x):
        b = self._base
        x = b.relu(b.bn1(b.conv1(x)))
        x = b.maxpool(x)
        x = self.drop2d(b.layer1(x))
        x = self.drop2d(b.layer2(x))
        x = self.drop2d(b.layer3(x))
        x = b.layer4(x)
        x = b.avgpool(x).flatten(1)
        x = self.drop1d(x)
        return b.fc(x)

    def get_features(self, x):
        b = self._base
        x = b.relu(b.bn1(b.conv1(x)))
        x = b.maxpool(x)
        x = b.layer1(x)
        x = b.layer2(x)
        x = b.layer3(x)
        x = b.layer4(x)
        return b.avgpool(x).flatten(1)


def get_paper_model(name: str, num_classes: int = 10, drop_prob: float | None = None) -> nn.Module:
    """name: 'smpl_paper' | 'aconv_paper' | 'resnet18_paper' | 'heiligers_paper'.
    drop_prob 생략 시 PAPER_RECIPE의 해당 모델 기본값 사용."""
    if drop_prob is None:
        drop_prob = PAPER_RECIPE[name]["drop"]
    if name == "smpl_paper":
        return SMPLPaper(num_classes, drop_prob)
    if name == "aconv_paper":
        return AConvNetPaper(num_classes, drop_prob)
    if name == "resnet18_paper":
        return ResNet18Paper(num_classes, drop_prob)
    if name == "heiligers_paper":
        return HeiligersPaper(num_classes, drop_prob)
    raise ValueError(f"Unknown paper model: {name!r}")
