# CLAUDE.md — SAR-ATR 프로젝트 컨텍스트

> 이 파일은 Claude Code가 자동으로 읽습니다.
> 모든 구현은 이 문서의 인터페이스·경로·버그 목록을 따르세요.

---

## 프로젝트 개요

Geng et al. 2023 ("Target Recognition in SAR Images by Deep Learning with Training Data
Augmentation") 재현 및 개선 과제.

**4개 실험:**
- Exp A: 클러터 전이 (Table 4 재현) — gengzhe2015 데이터
- Exp B: 위상 기록(PH)/산란점 추출 — MSTAR raw binary
- Exp C: 대비 보정 + Optuna (Figure 1) — SAMPLE dataset
- Exp D: OOD 탐지 — MSTAR 10클래스 + SAR-ship OE

---

## 디렉토리 구조

```
D:\workspace\software\sar-atr\          ← 프로젝트 루트 (이 파일 위치)
├── core/
│   ├── interfaces.py    ← SARSample, SARDataset, TrainConfig, EvalResult (변경 금지)
│   ├── models.py        ← SMPL, get_resnet18(), get_model()
│   ├── train.py         ← train_model()
│   ├── evaluate.py      ← evaluate(), evaluate_ood()
│   └── mock_data.py     ← MockSARDataset
├── augmentation/
│   ├── boundary_blend.py
│   ├── contrast_balance.py
│   └── ph_extraction.py
├── experiments/
│   ├── exp_a_clutter_transfer.py
│   ├── exp_b_ph_scattering.py
│   ├── exp_c_contrast_optuna.py
│   └── exp_d_ood.py
├── gradcam/             ← 아직 없음, 구현 필요
├── notebooks/
└── results/
```

---

## 데이터 경로

### 로컬 (개발·디버깅용, mock 전용)
```
D:\workspace\software\sar-atr\data\     ← .gitignore에 포함
```

### Google Drive (실데이터 — Colab에서 마운트)
```
MyDrive/SAR_ATR_Project/data/
├── clutter_gengzhe/                 ← Exp A
│   ├── Original MSTAR Images/       (gengzhe2015 ZIP 해제)
│   ├── Train_OR_Test_CT/
│   ├── Train_CT_Test_CT/
│   └── Train_CTx2_Test_CT/
│
├── sample/                          ← Exp C (SAMPLE dataset)
│   └── png_images/
│       ├── BMP2/
│       │   ├── measured/
│       │   └── synthetic/
│       └── ...
│
├── mstar/                           ← Exp B + Exp D (SDMS 승인 후)
│   ├── targets/                     (MSTAR Targets 패키지)
│   │   ├── BMP2/
│   │   │   ├── train/               El=15° raw binary 파일들 (.015 확장자)
│   │   │   └── test/                El=17° raw binary 파일들 (.017 확장자)
│   │   ├── BTR70/
│   │   └── T72/
│   └── mixed_targets/               (MSTAR/IU Mixed Targets 패키지)
│       ├── 2S1/
│       │   ├── train/
│       │   └── test/
│       ├── BRDM2/
│       ├── BTR60/
│       ├── D7/
│       ├── T62/
│       ├── ZIL131/
│       └── ZSU23-4/
│
└── sarship/                         ← Exp D OE (Kaggle 다운로드)
    └── images/                      PNG 파일들 (크기 무관)
```

### Colab에서 마운트하는 법
```python
from google.colab import drive
drive.mount('/content/drive')
DATA_ROOT = '/content/drive/MyDrive/SAR_ATR_Project/data'
# 심볼릭 링크로 코드 경로와 연결
import os, subprocess
subprocess.run(['ln', '-sf', DATA_ROOT, '/content/repo/data'])
```

---

## 참조 레포 (~/refs/ 에 clone)

```bash
git clone https://github.com/jacobgil/pytorch-grad-cam       ~/refs/pytorch-grad-cam
git clone https://github.com/kkirchheim/pytorch-ood           ~/refs/pytorch-ood
git clone https://github.com/SENSE-Lab-OSU/mstar_data_aug     ~/refs/mstar_data_aug
```

| 참조 레포 | 참고할 파일 | 용도 |
|---|---|---|
| `~/refs/pytorch-grad-cam` | `pytorch_grad_cam/grad_cam.py` | Grad-CAM hook 구현 |
| `~/refs/pytorch-ood` | `src/pytorch_ood/detector/odin.py`, `mahalanobis.py` | OOD 탐지기 로직 검증 |
| `~/refs/mstar_data_aug` | `*.m` 파일들 | MATLAB PH 추출 로직 → Python 번역 참고 |

---

## 핵심 인터페이스 계약 (변경 금지)

```python
# core/interfaces.py 요약
SARSample: image[1,H,W] float32, label:int, meta:dict
SARDataset: __getitem__→SARSample, __len__, class_names
Augmentation: (image:Tensor, meta:dict) → Tensor
TrainConfig: model_name, num_classes, epochs=60, batch_size=128,
             lr=1e-3, lr_decay_epoch=50, loss_type="lsm"|"at"
EvalResult: accuracy, confusion_matrix, per_class_accuracy,
            auroc=None, tnr_at_95tpr=None
```

모든 Dataset 클래스는 **반드시 SARDataset을 상속**해야 합니다.
모델은 항상 `get_model(name, num_classes)` 를 통해 생성하세요.

---

## 🔴 수정 필요한 버그 목록

### BUG-1: Exp B 산란점 도메인 불일치 [치명]

**파일:** `augmentation/ph_extraction.py`, `experiments/exp_b_ph_scattering.py`

**문제:** `extract_scattering_centers()`가 FFT 주파수 도메인 좌표를 반환함.
Grad-CAM은 공간(spatial) 도메인 → IoU 비교가 물리적으로 무의미.

**수정:**
1. `ph_extraction.py`에 `extract_spatial_scattering_centers(amplitude, k=5, min_distance=5)` 추가:
   - `amplitude` 이미지(공간 도메인)에서 `scipy.ndimage.maximum_filter`로 local maxima 추출
   - DC 제거 불필요 (FFT 아니므로), 단순히 밝기 상위 K개 픽셀 좌표 반환
2. `exp_b_ph_scattering.py`의 `analyse_sample()`에서 `extract_scattering_centers` → `extract_spatial_scattering_centers`로 교체

```python
# 추가할 함수 skeleton
def extract_spatial_scattering_centers(
    amplitude: np.ndarray, k: int = 5, min_distance: int = 5
) -> list[tuple[float, float]]:
    """공간 도메인 amplitude 이미지에서 직접 산란점(밝은 점) 추출."""
    from scipy.ndimage import maximum_filter
    local_max = maximum_filter(amplitude, size=min_distance * 2 + 1)
    peaks_mask = (amplitude == local_max) & (amplitude > amplitude.mean())
    ys, xs = np.where(peaks_mask)
    vals = amplitude[ys, xs]
    order = np.argsort(vals)[::-1][:k]
    return [(float(ys[i]), float(xs[i])) for i in order]
```

---

### BUG-2: Exp D 데이터 누수 [치명]

**파일:** `experiments/exp_d_ood.py`

**문제:** `load_id_holdout()`의 real data 분기에서 `train_ds`와 `test_id_ds`가
동일 폴더를 그대로 로드 → train/test 중복.

**수정:** `exp_a_clutter_transfer.py`의 `_split_dataset()`과 `MSTARImageFolder`를
import해서 80/20 분리 적용:

```python
from experiments.exp_a_clutter_transfer import _split_dataset, MSTARImageFolder

# load_id_holdout() real data 분기 수정
full_ds = MSTARImageFolder(MSTAR_DIR, known)
train_ds, test_id_ds = _split_dataset(full_ds, train_ratio=0.8, seed=seed)
```

---

### BUG-3: Exp A gengzhe2015 split 로직 오류 [중간]

**파일:** `experiments/exp_a_clutter_transfer.py`

**문제:** 현재 코드는 전체 데이터를 랜덤 80/20 분할함.
gengzhe2015 레포 README에 따르면 El=15°(train) / El=17°(test)로 이미 분리된 이미지.

**수정:**
- `_split_dataset()` 호출 제거
- 각 ZIP 폴더 내에서 `train/` 과 `test/` (또는 El=15° / El=17°) 서브폴더를 직접 로드
- ZIP 해제 후 실제 폴더 구조 확인 필수: `ls data/clutter_gengzhe/` 로 확인 후 경로 수정

---

### BUG-4: Exp C SAMPLE 로더 없음 [중간]

**파일:** `experiments/exp_c_contrast_optuna.py`

**문제:** MSTAR El=17°/30° 전용 로더만 있음. SAMPLE은 `png_images/<class>/measured/` +
`synthetic/` 구조.

**수정:** `SampleDataset` 클래스 추가:

```python
class SampleDataset(SARDataset):
    """
    SAMPLE dataset loader.
    구조: data/sample/png_images/<class_name>/measured/*.png
          data/sample/png_images/<class_name>/synthetic/*.png
    split: "measured" | "synthetic"
    """
    def __init__(self, root: Path, split: str, class_names: list[str]):
        # split = "measured" or "synthetic"
        ...
```

`load_el17_el30()` 대신 `load_sample()` 함수 추가:
```python
def load_sample(class_names=SAMPLE_CLASSES):
    """Returns (train_synthetic, test_measured)"""
    train = SampleDataset(DATA_ROOT / "sample/png_images", "synthetic", class_names)
    test  = SampleDataset(DATA_ROOT / "sample/png_images", "measured",  class_names)
    return train, test
```

MSTAR El=17°/30° 실험은 `run_el_ablation()` 함수로 분리해서 보조 ablation으로 유지.

---

### BUG-5: contrast_balance.py _AugmentedDataset 미상속 [경미]

**파일:** `augmentation/contrast_balance.py`

**문제:** `make_optuna_objective()` 내 `_AugmentedDataset`이 `SARDataset` 미상속.

**수정:**
```python
from core.interfaces import SARDataset
class _AugmentedDataset(SARDataset):   # SARDataset 상속 추가
    ...
```

---

### BUG-6: models.py get_features 바인딩 불안정 [경미]

**파일:** `core/models.py`

**문제:** `get_resnet18()`에서 `types.MethodType`으로 `get_features`를 인스턴스에
직접 바인딩 → `torch.save()` 시 pickle 실패 가능.

**수정:** ResNet18 wrapper 클래스로 대체:
```python
class ResNet18SAR(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        base = tv_models.resnet18(weights=None)
        base.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        base.fc = nn.Linear(base.fc.in_features, num_classes)
        self._base = base

    def forward(self, x):
        return self._base(x)

    def get_features(self, x):
        b = self._base
        x = b.relu(b.bn1(b.conv1(x)))
        x = b.maxpool(x)
        x = b.layer1(x); x = b.layer2(x)
        x = b.layer3(x); x = b.layer4(x)
        return b.avgpool(x).flatten(1)
```

---

## 추가 구현 필요 항목

### IMPL-1: 시각화 함수 (core/evaluate.py에 추가)

```python
def plot_confusion_matrix(result: EvalResult, class_names: list[str],
                           title: str = "", save_path=None) -> None:
    """seaborn heatmap, annotated, saved to PNG."""

def plot_accuracy_bar(results_dict: dict, title: str = "",
                       save_path=None) -> None:
    """조건별 정확도 막대그래프 + 오차막대 (mean ± std)."""
```

의존성: `pip install seaborn matplotlib`

### IMPL-2: gradcam/ 모듈 신설

```
gradcam/
├── __init__.py
├── cam.py          ← GradCAM 클래스 (~/refs/pytorch-grad-cam 참고)
└── scatter_overlap.py  ← IoU 계산, 시각화
```

`~/refs/pytorch-grad-cam/pytorch_grad_cam/grad_cam.py` 읽고 우리 모델 인터페이스에 맞게 구현.

### IMPL-3: notebooks/visualize.ipynb 생성

metrics.json → confusion matrix PNG, 정확도 비교표 자동 생성 노트북.

---

## 테스트 방법 (실데이터 없이)

```bash
cd D:\workspace\software\sar-atr

# 파이프라인 전체 smoke test
python -c "
from core.mock_data import MockSARDataset
from core.models import get_model
from core.train import train_model
from core.interfaces import TrainConfig

ds = MockSARDataset(n=50, num_classes=5)
model = get_model('smpl', 5)
cfg = TrainConfig('smpl', 5, epochs=2)
model, result = train_model(model, ds, ds, cfg)
print('OK:', result.accuracy)
"
```

---

## 커밋 규칙

- `feat:` 새 기능
- `fix:` 버그 수정 (BUG-N 번호 명시 권장, 예: `fix: BUG-1 spatial scattering centers`)
- `exp:` 실험 결과
- `docs:` 문서
