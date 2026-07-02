# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

Geng et al. 2023 ("Target Recognition in SAR Images by Deep Learning with Training Data Augmentation") 재현 및 개선 과제.

**4개 실험:**
- Exp A: 클러터 전이 Table 4 재현 — gengzhe2015 데이터
- Exp B: PH 보간 증강 Table 3 재현 — MSTAR Mixed Targets raw binary
- Exp C: 대비 보정 + Optuna Figure 1 재현 — SAMPLE dataset
- Exp D: OOD 탐지 (ODIN vs Mahalanobis) — MSTAR 10클래스 + SAR-ship

**우리 팀 개선 3가지:**
1. SSIM 경계 아티팩트 정량화 (`run_boundary_ssim_analysis()` in exp_a)
2. Adam 옵티마이저 비교
3. Grad-CAM × 산란점 IoU 검증 (`run_gradcam_analysis()` in exp_b)

---

## 실행 명령

```bash
# 전체 파이프라인 smoke test (실데이터 없이)
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

# Exp B smoke test (mock 모드)
python -c "from experiments.exp_b_ph_scattering import run; print(run(use_mock=True, epochs=2, n_interp=2))"

# Exp A 직접 실행
python experiments/exp_a_clutter_transfer.py --mock
python experiments/exp_a_clutter_transfer.py --ssim  # SSIM 분석

# Exp B 직접 실행
python experiments/exp_b_ph_scattering.py --mock
python experiments/exp_b_ph_scattering.py --gradcam --checkpoint results/exp_b/model.pth

# Exp C 직접 실행
python experiments/exp_c_contrast_optuna.py --mock

# Exp D 직접 실행
python experiments/exp_d_ood.py --model smpl

# MSTAR 파일 포맷 진단
python scripts/diag_mstar_format.py
```

---

## 핵심 인터페이스 계약 (변경 금지)

`core/interfaces.py` 에 정의된 타입들:

```python
SARSample: image[1,H,W] float32, label:int, meta:dict
SARDataset: __getitem__→SARSample, __len__, class_names:list[str]
TrainConfig: model_name, num_classes, epochs=60, batch_size=128,
             lr=1e-3, lr_decay_epoch=50, loss_type="lsm"|"at"
EvalResult: accuracy, confusion_matrix, per_class_accuracy,
            auroc=None, tnr_at_95tpr=None
```

- 모든 Dataset 클래스는 반드시 `SARDataset`을 상속해야 합니다.
- 모델은 항상 `get_model(name, num_classes)`로 생성하세요 (`core/models.py`).
- `get_features(x)` 메서드: SMPL과 ResNet18SAR 모두 구현됨 — OOD/Grad-CAM에서 사용.

---

## 아키텍처

### 데이터 흐름
```
raw MSTAR binary → read_mstar_complex() → interpolate_phase_history() → amplitude_to_tensor()
PNG/JPEG images  → PIL.Image → numpy → torch.Tensor [1,128,128]
```

### 학습 파이프라인
`train_model(model, train_ds, test_ds, config)` → `(model, EvalResult)`  
내부: DataLoader → CrossEntropyLoss (또는 LSM/AT loss) → Adam → tqdm 진행 표시

### OOD 탐지 (`core/evaluate.py:evaluate_ood`)
- ODIN: temperature scaling(T=1000) + input perturbation(ε=0.0014) → softmax score
- Mahalanobis: 훈련셋으로 클래스별 평균/공분산 추정 → `-0.5 * min_class_dist` score
- AUROC, TNR@95TPR 계산

### Grad-CAM (`gradcam/`)
- `GradCAM(model)`: forward hook으로 feature map 저장, output tensor hook으로 gradient 저장
- `cam(image_t.unsqueeze(0))` → [H,W] heat map (0~1 normalized)
- `scatter_overlap.centers_to_mask()` + `iou()` → 산란점과 cam 일치도

---

## MSTAR 파일 형식 (중요)

Phoenix binary format (`augmentation/ph_extraction.py`):
- 파일 = [ASCII 헤더: `PhoenixHeaderLength` 바이트] + [SAR complex float32 데이터]
- **`PhoenixSigSize` = 파일 전체 크기** (extra block이 아님 — 오해하기 쉬운 필드)
- SAR 데이터 오프셋 = `PhoenixHeaderLength` 값만 사용
- 데이터: big-endian float32, real+imag 교차 저장
- **부각(앙각)은 헤더의 `DesiredDepression`/`MeasuredDepression` 필드에서 읽어야 함** — Mixed Targets의 파일 확장자(`.000`/`.001` 등)는 앙각이 아니라 단순 일련번호. (Targets chips는 `.017`/`.015` 확장자가 앙각이지만 Mixed Targets는 다름)
- Mixed Targets CD2는 158×158 이미지 → `amplitude_to_tensor()`에서 128×128로 리사이즈

---

## 데이터 경로 (Google Drive — Colab 전용)

```
MyDrive/SAR_ATR_Project/data/
├── clutter_gengzhe/          ← Exp A
├── sample/png_images/
│   ├── real/<class>/         ← Exp C 테스트 (measured)
│   └── synth/<class>/        ← Exp C 학습 (synthetic)
├── mstar/
│   ├── MSTAR_PUBLIC_TARGETS_CHIPS_T72_BMP2_BTR70_SLICY/  ← Exp D
│   ├── MSTAR_PUBLIC_MIXED_TARGETS_CD1/                   ← Exp B + D
│   └── MSTAR_PUBLIC_MIXED_TARGETS_CD2/                   ← Exp B (주)
└── sarship/                  ← Exp D OE
```

Colab 마운트:
```python
from google.colab import drive
drive.mount('/content/drive')
import subprocess
subprocess.run(['ln', '-sf', '/content/drive/MyDrive/SAR_ATR_Project/data', '/content/repo/data'])
```

SAMPLE 클래스 10개 (소문자): `2s1 bmp2 btr70 m1 m2 m35 m60 m548 t72 zsu23`

---

## 알려진 설계 결정 및 주의사항

**Exp B 데이터 = CD1 + CD2 둘 다 로드** (`MSTAR_RAW_DIRS`). 표준 MSTAR SOC(train 17° / test 15°) 재현을 위해 필수 — **15°는 CD1에만, 17°는 CD2에만** 있음. 7개 클래스 모두 두 부각을 다 가짐 (진단 교차표로 확인).

**Exp B train/test split:** `_split_train_test()` 사용 — **헤더 부각 기반 cross-depression split**.
헤더 `DesiredDepression` 필드에서 부각을 읽어 전역 상위 2개 부각을 train/test로 분리 (`_depression_angle()`).
- CD1+CD2 합산 시 7개 클래스 기준 17°(2049) / 15°(1838)가 상위 2개 → **train=17°, test=15°** (표준 SOC).
- 특정 클래스가 두 부각 중 하나만 가지면 그 클래스만 클래스 내 랜덤 80/20, 부각을 아예 못 읽으면 전체 stratified 폴백 (0% 방지).
- 같은 부각 내 랜덤 분할 시 99%+ 정확도 (trivial) — 논문 재현 불가.
- ⚠️ **파일 확장자로 앙각 판별 금지** (Mixed Targets 확장자 `.001`/`.015`는 앙각이 아니라 일련번호; `.015` 파일의 실제 부각이 16°인 경우도 있음).

**Taylor 윈도우:** `scipy.signal.windows.taylor(sll=35)` — **양수** 값 사용.
`sll=-35` 시 `arccosh` 정의역 위반 → NaN → 합성 이미지 전부 zeros.

**`PHAugmentedDataset`:** lazy loading 설계 — `__init__`에서 경로 튜플만 저장,
실제 I/O는 `__getitem__`에서만 수행. eager loading 복귀 시 `num_samples=0` 발생.

**`FolderDataset` (exp_d):** Phoenix 헤더 확인 (`b"PhoenixHeaderVer"` in 첫 100바이트)
후 파일 필터링. 헤더 없는 파일 학습 시 zeros 데이터 문제.

---

## 커밋 규칙

- `feat:` 새 기능
- `fix:` 버그 수정 (예: `fix: BUG-X1 taylor sll sign`)
- `exp:` 실험 결과
- `docs:` 문서
