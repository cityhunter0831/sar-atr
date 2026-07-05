# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> 📌 **이 문서는 "코어 구현 방향·이론적 배경·진척도"의 살아있는 기록이다.**
> 핵심 구현 방향이나 진행 상태가 바뀌면 (지시 없어도) 이 문서를 갱신할 것.
> 참조 문서: `docs/PAPER_SPEC.md`(Geng 논문 표·수치) · `docs/DATASET_METHOD.md`(데이터셋·정당성) · `docs/THEORY_REFERENCES.md`(이론배경·참조논문·구현난점·검증기준, 보고서용) · `docs/PRESENTATION_CHECKLIST.md`(발표 슬라이드별 필요 항목) · `docs/2b626a7d-sensors2300941.pdf`(Geng 원문) · `docs/9eac9493-2012.09284v2.pdf`(Agarwal 위상보간 원전)

## 프로젝트 개요

Geng et al. 2023 ("Target Recognition in SAR Images by Deep Learning with Training Data Augmentation") 재현 및 개선 과제.

> ⭐ **참조 문서**: `docs/PAPER_SPEC.md`(논문 원문 표·수치, 권위 문서) · `docs/DATASET_METHOD.md`(실험별 데이터셋·구현방법·대체 정당성)
> 코드가 논문과 다르면 논문이 정답. 아래는 요약이며, 정확한 표/샘플수/클래스는 PAPER_SPEC.md에 있음.

**4개 실험 (논문 원문 기준):**
- Exp A: 클러터 전이 Table 4 — 5클래스(2S1,BMP2,BTR70,T72,ZSU23), train El15°/test El17°
- Exp B: PH 보간 증강 Table 3 — 5클래스, **few-shot**(baseline 136장→Aug 1088장), train El17°/test El15°, **64×64 crop**
- Exp C: 대비 보정 Figure1/Table6 — SAMPLE 10클래스, K=0(100% synth)→measured, 목표 RN18 94.5%
- Exp D: OOD 탐지 — **ID=SAMPLE 10클래스**, OE=SAR-ship+MiniSAR, OOD=Holdout+MSTAR-O/P

⚠️ **현재 코드는 논문과 여러 곳이 다름** (특히 Exp B는 few-shot이 아니라 전체 데이터로 학습 중 → 98% 나옴). PAPER_SPEC.md의 "현재 코드와의 차이" 표 참조.

**우리 팀 개선 3가지 (논문에 없는 추가 기여):**
1. SSIM 경계 아티팩트 정량화 (`run_boundary_ssim_analysis()` in exp_a) — 클러터 전이 품질 정량화
2. 옵티마이저 비교 (ADAM vs SGD) — 논문은 ADAM 고정
3. XAI × 산란점 IoU 검증 (`run_gradcam_analysis()` + `run_xai_analysis()` in exp_b) — 모델이 물리적 산란점을 보는지 검증.
   Grad-CAM은 SMPL 8×8 특징맵 해상도 한계(1.05×) → **픽셀 단위 XAI로 확장**: Occlusion(인과, 3.21×) + SmoothGrad-IG(공리적, **11.95×**). 구현 `gradcam/attributions.py`.
   (부가: Exp C에 Optuna 하이퍼파라미터 자동 탐색 — 논문은 대비 레벨 3개 고정)

---

## 🎯 프로젝트 지향점 (항상 참고)

**우리가 하려는 것 = ① 논문 4개 실험 충실 재현 + ② 우리 팀 개선 3가지 추가.**

원칙:
- **논문이 정답이다.** 코드가 논문과 다르면 논문에 맞춘다. 정확한 표/샘플수/클래스/입력크기는 `docs/PAPER_SPEC.md`(권위 문서) 참조.
- 논문과 **의도적으로 다르게 가는 경우** = 데이터 가용성 등 불가피한 이유가 있을 때만. 그 이유를 아래 표에 남긴다.
- 논문과 **의도치 않게 어긋난 것** = 버그/오해 → 트러블슈팅 목록에서 수정한다.

### 논문 실험 요약 (headline, 상세는 PAPER_SPEC.md)
| Exp | 논문 실험 | 데이터 | headline 수치 |
|---|---|---|---|
| A | Clutter transfer (Table 4) | MSTAR 5클래스, train El15/test El17 | 원본→CT 급락(SMPL7 38.6%) → CTx2 회복(96.0%) |
| B | PH 보간 few-shot (Table 3) | MSTAR 5클래스, train El17/test El15, **136장→1088장** | SMPL/AT **56.6%→96.4%** |
| C | Contrast 증강 (Table 6) | SAMPLE 10클래스, K=0 synth→measured | RN18 91.9%→**94.5%** |
| D | OOD 탐지 (Figure 9) | ID=SAMPLE, OE=SAR-ship+MiniSAR, OOD=Holdout+MSTAR-O/P | Maha가 대체로 최고 |

### 의도적 설계 변경 (근거 있음 — 유지)
| 항목 | 논문 | 우리 | 이유 |
|---|---|---|---|
| Exp D OE 데이터 | MiniSAR + SAR-ship | **SAR-ship만** | MiniSAR은 비공개(논문 저자 자체 개발) → 공개 SAR-ship로 대체 |
| Exp C 하이퍼파라미터 | 대비 레벨 3개 고정 | **Optuna 자동 탐색** | 우리 팀 개선 — 재현성·객관성 강화 |
| 개선 #1 SSIM | (없음) | 추가 | 클러터 전이 경계 품질 정량화 (우리 기여) |
| 개선 #2 옵티마이저 | ADAM 고정 | ADAM vs SGD | 증강 효과의 옵티마이저 의존성 분석 (우리 기여) |
| 개선 #3 Grad-CAM | (없음) | 추가 | 물리적 해석가능성 검증 (우리 기여) |

### 🔧 트러블슈팅 (논문과 어긋남 = 수정 대상)
| # | 실험 | 문제 → 논문 정답 | 상태 |
|---|---|---|---|
| T1 | Exp B | 7클래스 전체(2049장) → 5클래스 **few-shot 136장** (56.6%) | ✅ 코드 반영 (`few_shot=True`, 5클래스, `FEWSHOT_COUNTS`) |
| T2 | Exp B | 128 resize → **64×64 center-crop** | ✅ `amplitude_to_tensor(center_crop=64)` |
| T3 | Exp B | CE → **AT(ε=2)/LSM** | ✅ `run(loss_type='at'/'lsm')` |
| T4 | Exp B | 인접파일 보간 → **azimuth 이웃 PH 보간** | ✅ `read_azimuth` 정렬 pairing |
| T5 | Exp A | TrainCTx2 붕괴(39%) → 회복(**96.0%**) | 🟠 진단 로깅 추가 — **Colab 실행해 폴더 로드수 확인 필요** |
| T6 | Exp D | ID=MSTAR → **ID=SAMPLE 10클래스**, SAR-ship=far-OOD | ✅ `SampleDataset` 기반 재설계 |
| T7 | Exp C | 목표 불명확 → K=0 **RN18 94.5%** 명시 | ✅ `_print_figure1` 수정 |
| T8 | Exp B | σ_G 라인서치 제거 필요성 → **σ_G=1.0 고정 확정** | ✅ 2S1 대표이미지 잔차 차이 0.002% 정량 검증. 재복원 불필요. |
| ✅ | 공통 | ~~BUG-X1~X4 (Taylor 부호/오프셋/포맷)~~ | 완료 |

> **T1~T4,T6,T7은 코드 반영 완료. 실데이터 검증은 Colab 필요.** T5는 진단 로깅만 넣음(원인 확정에 Colab 실행 필요).
> Exp B 실행: `run(model_name='smpl', loss_type='at')` → 목표 SMPL/AT **56.6%→96.4%** (few-shot이 핵심).

---

## ⭐ Exp B 위상보간 — 완전판 구현 방향 (진행 중, 코어)

### 이론적 배경 (왜 이 방법인가)
- Geng 논문은 위상보간을 **직접 구현하지 않고 Agarwal et al. [19] 방법을 채택**("we adopt the method proposed in Agarwal et al."). 따라서 정답 구현체 = `docs/9eac9493-2012.09284v2.pdf` + `SENSE-Lab-OSU/mstar_data_aug`(MATLAB).
- 핵심 원리: SAR 타겟은 소수의 **산란점(scattering centers)에 에너지가 집중**(스파시티). 그 산란점의 위치·계수만 알면 **위상이력(PH) 도메인에서 임의 방위각의 이미지를 물리적으로 재합성** 가능.
- 수식: Eq.6(격자 point-scatterer forward 모델) → **Eq.7 그룹 희소 복원** `min_C(Σλ‖c_k‖₂ + ‖S−Ŝ‖_F)` → Eq.8(임의 θ 재합성). Fig.3이 전체 파이프라인.

### ❌ 기존 구현의 오류 (버려야 함)
- `interpolate_phase_history(img_a, img_b, alpha)` = **두 이미지 선형 평균**. 산란점을 전혀 안 씀 → 논문 방법 아님. few-shot에서 65% 정체의 원인.

### ✅ 완전판 3단계 (구현 계획)
| 단계 | 대응 MATLAB | 하는 일 | Python 구현 |
|---|---|---|---|
| 1. preprocess | `preprocess_raw_data.m` | 복소이미지 → FFT_shift → 2D_FFT → K-space → inverted Taylor → Cartesian→Polar 보간 → PH(극좌표) | scipy FFT + `griddata` |
| 2. sparse recovery | `sparse_recovery.m` (SPGL1) | Eq.7 그룹 희소 복원으로 산란점 계수 C 추정 + σ_G 라인서치 | **FISTA + 그룹 soft-threshold** (SPGL1 불필요) |
| 3. synthesize | `generate_aug_images.m` | Eq.8로 ±6° 방위각 외삽 + subpixel shift → Polar→Cartesian → Taylor → IFFT | numpy(mtimesx 대체) + `griddata` |

### 정당성 (논문과 다르게 가는 부분)
- **2단계 solver를 SPGL1 대신 FISTA+그룹prox로**: Eq.7은 표준 group-lasso(L2,1). FISTA 근접경사법으로 동일 문제를 풂 → MATLAB MEX 의존성 제거, Python 순정. **결과 동등, 구현 단순**.
- **mtimesx(MEX) → numpy matmul**, **scatteredInterpolant → scipy.griddata**: 언어 대체일 뿐 수식 동일.

### 데이터 규모 (혼동 금지)
- **136장 = Geng Table 2 (5클래스, few-shot baseline)** — 우리 재현 목표. 이 136장 각각을 위상보간으로 증강해 Aug1(1088장) 구성.
- Agarwal 원전은 10클래스를 R∈{2⁻⁵..2⁰} 비율로 줄임(136 고정 아님). 방법은 Agarwal, 규모는 Geng.

### 파라미터 (논문·MATLAB 확인값)
- 패치 L=30m, 해상도 0.3m → 격자 100×100. f_center=9.6GHz, BW=521MHz, 100 freq bins.
- 가우시안 기저 D=12(3° 서브개구), σ_G 이미지별 라인서치. Taylor(100,4,−35). 입력 64×64 crop. 외삽 ±6°(η=3).

### ⭐ 현재 노선 = 하이브리드 (로컬 MATLAB + Colab Python) — 이게 실제 파이프라인
OSU Box precomputed 데이터가 삭제(404)돼 지름길이 막힌 뒤, **원본 MATLAB(Agarwal repo)을 로컬에서 직접 실행**하는 방식으로 전환. Python 포팅(`ph_sparse.py`)은 원리 이해·검증용으로 남기되, **실제 증강 데이터는 MATLAB이 생성**한다.

- **로컬 MATLAB (다른 채팅=Antigravity/Gemini 담당, Agarwal repo)**: stage1(PH)→stage2(희소복원)→stage3(generate)→merge. 가속 완료(장당 26분→18초): 익명함수 슬라이싱 제거, `pagemtimes`, `maxNumCompThreads(1)`, **`gaussWidth=1.0` 고정 ✅ 검증 완료** — 2S1 대표이미지에서 σ_G=1(잔차 9419.6) vs σ_G=2(잔차 9419.4) 차이 **0.002%**. 민감도 극히 낮음 정량 확인 → 재복원 불필요, 발표 방어 가능.
- **Colab Python (이 repo=sar-atr, 나 담당)**: MATLAB이 만든 `.mat`을 로드해 SMPL/AT 학습.
- **데이터 핸드오프 = Google Drive** (두 채팅은 메모리 공유 안 함, `.mat` 파일이 인터페이스).

### 데이터 규격 (MATLAB→Colab 계약)
- `<class>_aug_images.mat`: `imgTrain`(N×64×64 복소, 샘플당 196장), `aziTrain`, `elev` — El17° 증강 학습셋
- `<class>_baseline.mat`: 동일 규격 — El17° 원본 few-shot 136장 (증강 전 비교용)
- `<class>_test.mat`: `imgTest`(M×64×64) — El15° 실측 1913장 (평가용)
- 파일명: aug는 시리얼명(9개, BMP2_SN_9563 등), baseline/test는 병합명(5개). ZSU_23_4=ZSU23.

### Python 로더 (완료, `augmentation/precomputed_aug.py`)
- `AugImagesDataset` / `BaselineDataset` / `TestImagesDataset` (모두 `MatImagesDataset(suffix=...)` 래퍼) — `SARDataset` 호환, `train_model()`에 바로 투입.
- `inspect_mat(path)` — .mat 변수·shape 검사. `_resolve_class()` — 파일명→5클래스 매핑(시리얼/병합/ZSU 혼용 대응, 검증됨).

### 진척도
- [x] 데이터 설계 확정: 5클래스(2S1,BMP2,BTR70,T72,ZSU23), few-shot **136장 정밀분포**(24/32/24/24/32), El17train/El15test, 64×64 crop, AT손실
- [x] 로컬 MATLAB stage1(PH) 완료, stage2(희소복원) 구동 중(막바지)
- [x] El15° test 1913장 export 완료 (장수 논문과 일치: 274/587/196/582/274)
- [x] El17° baseline 136장 export 완료
- [x] Python 로더 3종 완료·검증
- [ ] **stage3(generate_aug_images) + merge → `<class>_aug_images.mat` 생성** ← 지금 대기 중
- [ ] Drive 업로드 → Colab 학습: baseline(목표 56.6%) vs aug(목표 96.4%)
- (참고) Python 포팅 `ph_sparse.py`: stage1 corr0.99✅, stage2 연산자 자기일관성5/5✅ + λ_max스케일링·sigma_n캘리브(`calibrate_sparse`) — 검증용, 실파이프라인은 MATLAB

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
raw MSTAR binary → read_mstar_raw()([진폭 블록]) / read_mstar_complex()(진폭·exp(i·위상))
                 → interpolate_phase_history() → amplitude_to_tensor()
PNG/JPEG images  → PIL.Image → numpy → torch.Tensor [1,H,W] (리사이즈)
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
- 파일 = [ASCII 헤더: `PhoenixHeaderLength` 바이트] + [SAR 데이터]
- **`PhoenixSigSize` = 파일 전체 크기** (extra block이 아님 — 오해하기 쉬운 필드)
- SAR 데이터 오프셋 = `PhoenixHeaderLength` 값만 사용
- ⚠️ **데이터 = [진폭 블록][위상 블록]** (BUG-X4): big-endian float32, 픽셀당 2개.
  real+imag 교차가 **아님**. 교차로 읽으면 이미지가 상하로 갈림(위=진폭 오독, 아래=위상 노이즈).
  진폭 = 첫 블록, complex = 진폭·exp(i·위상).
- **부각(앙각)은 헤더의 `DesiredDepression`/`MeasuredDepression` 필드에서 읽어야 함** — Mixed Targets의 파일 확장자(`.000`/`.001` 등)는 앙각이 아니라 단순 일련번호. (Targets chips는 `.017`/`.015` 확장자가 앙각이지만 Mixed Targets는 다름)
- 이미지 크기 다양(128×129, 158×158 등) → `amplitude_to_tensor()`에서 리사이즈.
  ⚠️ 단 **Exp B 논문 재현 시 입력은 64×64 center-crop** (PAPER_SPEC 참조)

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

**⚠️ Exp B 재설계 필요 (논문과 불일치):** 현재 코드는 7개 Mixed Targets 클래스 전체 데이터(2049장)로 학습해 98%가 나오지만, **논문 Table 3은 5클래스(2S1,BMP2,BTR70,T72,ZSU23)를 클래스당 24~32장(총 136장)만 학습하는 few-shot 실험**이다 (baseline 56.6% → PH 증강 96.4%). BMP2/BTR70/T72는 Targets 패키지, 2S1/ZSU23는 Mixed Targets에 있음. 자세한 것은 `docs/PAPER_SPEC.md` Exp B 절 참조.

**Exp B 데이터 = CD1 + CD2 둘 다 로드** (`MSTAR_RAW_DIRS`). train 17° / test 15° — **15°는 CD1에만, 17°는 CD2에만** 있음.

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
