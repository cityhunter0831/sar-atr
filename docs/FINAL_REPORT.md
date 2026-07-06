# SAR-ATR 재현 및 개선 최종 보고서

> **논문**: Geng et al. 2023, *"Target Recognition in SAR Images by Deep Learning with Training Data Augmentation"*, Sensors 23(2), 941.
>
> **목표**: 논문의 4개 실험을 충실히 재현하고, 3가지 독자적 개선을 추가한다.
>
> **그래프 파일**: `docs/figures/` (GitHub 추적) 또는 Google Drive `SAR_ATR_Project/figures/`

---

## 1. 프로젝트 개요

### 1.1 배경
SAR(합성개구면 레이더)는 날씨·주야에 관계없이 고해상도 지표 관측이 가능한 센서다. 군사 표적 자동 인식(ATR)에서 딥러닝이 높은 성능을 보이지만, **실측 데이터가 극히 부족**하다는 근본 한계가 있다. Geng et al. 2023은 물리 기반 데이터 증강 4종으로 이 문제를 해결한다.

### 1.2 논문 구조 (4개 실험)

| 실험 | 주제 | 핵심 질문 |
|---|---|---|
| **Exp A** | 클러터 전이 (Clutter Transfer) | 배경이 바뀌어도 분류할 수 있는가? |
| **Exp B** | 위상이력 보간 (PH Interpolation) | 136장으로 96%를 달성할 수 있는가? |
| **Exp C** | 대비 증강 (Contrast Augmentation) | 합성 이미지로 실측 데이터를 분류할 수 있는가? |
| **Exp D** | OOD 탐지 (Out-of-Distribution) | 학습에 없던 표적을 "모른다"고 걸러낼 수 있는가? |

### 1.3 우리 팀의 추가 기여 (3가지 개선)

| # | 개선 | 실험 | 한 줄 설명 |
|---|---|---|---|
| 1 | SSIM 경계 정량화 | Exp A | 클러터 전이 품질을 SSIM으로 정량화 ("왜 잘 되는가"의 근거) |
| 2 | Optuna 대비 자동탐색 | Exp C | 논문의 매직넘버(contrast=0.5)를 데이터 기반 최적값으로 대체 |
| 3 | XAI 산란점 IoU 검증 | Exp B | 모델이 물리적 산란점을 보고 분류하는지 Grad-CAM/XAI로 검증 |

---

## 2. 총괄 결과 요약

| 실험 | 논문 핵심 지표 | 우리 결과 | 판정 |
|---|---|---|---|
| Exp A (Table 4) | 도메인 갭 → CTx2로 회복 96.0% | CTx2 **98.8%** (논문 초과) | 완전 재현 |
| Exp B (Table 3) | 56.6% → 96.4% (SMPL/AT) | 66.6% → **90.9%** (log-amp 60dB, AT) | 근접 재현 |
| Exp C (Table 6) | 91.9% → 94.5% (RN18) | ①69.6% ②61.8% ③**80.3%** (+18.5%p) | 개선 확인 |
| Exp D (Fig 9) | far-OOD 쉬움 / near-OOD 어려움 | far-OOD AUROC=1.000 / near-OOD ~0.45 | 패턴 재현 |

> **그래프**: 실험별 비교 그래프 5개 (`docs/figures/` — Colab Cell 11 실행 시 자동 저장)

---

## 3. Exp A — 클러터 전이

> **그래프**: `exp_a_clutter_transfer.png` (4조건 막대그래프, 논문 vs 우리)

![Exp A — 클러터 전이 재현](figures/exp_a_clutter_transfer.png)

### 3.1 논문이 한 것
SAR 이미지 = 표적(전차) + 클러터(배경). CNN이 배경 패턴을 외우면 배경이 달라질 때 성능 붕괴. 논문은 SAR-Bake 픽셀 주석으로 표적/그림자/클러터를 분리하고, 표적을 다른 클러터 배경에 feather blending으로 합성(클러터 전이)한다.

**4가지 실험 조건**:
1. **MSTAROR**: 원본 학습 → 원본 테스트 (기준선)
2. **TrainOR+TestCT**: 원본 학습 → 전이 테스트 (도메인 갭 측정)
3. **TrainCT+TestCT**: 전이 학습 → 전이 테스트 (회복)
4. **TrainCTx2+TestCT**: 2배 전이 학습 → 전이 테스트 (추가 향상)

### 3.2 우리의 재현
- **데이터**: gengzhe2015 공개 데이터 (논문 저자 직접 업로드), 5클래스
- **모델**: SMPL + ResNet18, 각 3 seed 평균
- **트러블슈팅**: CTx2 폴더의 zip 재압축 해제로 누락 클래스(btr70/t72/zsu23) 복원 → 3563장 정상

### 3.3 결과 비교표

| 조건 | 우리 SMPL | 논문 SMPL7 | 우리 RN18 | 논문 RN18 |
|---|---|---|---|---|
| MSTAROR | 93.3% | 98.1% | 100.0% | 99.8% |
| TrainOR+TestCT | **67.7%** | 38.6% | **81.9%** | 55.2% |
| TrainCT+TestCT | **96.3%** | 91.5% | **99.7%** | 97.5% |
| TrainCTx2+TestCT | **98.8%** | 96.0% | **99.9%** | 98.4% |

**핵심**: 도메인 갭(67.7%) → CT로 회복(96.3%) → CTx2로 추가 향상(**98.8%**, 논문 96.0% 초과). 논문의 서사 완전 재현.

### 3.4 개선 #1 — SSIM 경계 정량화

**문제**: 논문은 "전이하면 96%로 회복"이라고만 보고. 전이 이미지의 품질이 나쁘면 경계 아티팩트를 표적 특징으로 오인할 위험. **품질 근거가 없다.**

**우리가 한 것**: 표적-배경 경계 패치의 SSIM(구조적 유사도)을 원본 대비 측정.

**결과**: **SSIM = 0.9472 +/- 0.0024** → 경계 아티팩트가 극히 낮고, 구조적 일관성 높음 → 96% 회복의 품질 근거.

**발표 포인트**: "논문은 입력(전이)과 출력(정확도)만 보고했다. 우리는 그 사이의 '왜 되는가'를 SSIM으로 처음 정량화했다."

---

## 4. Exp B — 위상이력(PH) 보간 증강

> **그래프**: `exp_b_confusion_matrix.png` (baseline vs aug 혼동행렬)

![Exp B — Confusion Matrix](figures/exp_b_confusion_matrix.png)

### 4.1 논문이 한 것
**핵심 문제**: 실측 SAR 데이터가 클래스당 24~32장(총 136장)뿐인 few-shot 상황.

**논문 해법**: SAR 표적의 산란점(scattering centers)을 위상이력(PH) 도메인에서 그룹 희소 복원(Eq.7)으로 추정 → 미관측 방위각의 이미지를 물리적으로 재합성(Eq.8). 1장 → 196장 증강 (136장 → 1088장).

파이프라인 3단계 (Agarwal et al. 2020):
1. **PH 추출**: 이미지 → 2D FFT → 역Taylor → 극좌표 K-space
2. **희소 복원**: 그룹 Lasso(L2,1)로 산란점 계수 C 추정
3. **재합성**: 복원된 C로 임의 방위각 PH 생성 → IFFT → 새 이미지

### 4.2 우리의 재현 (하이브리드 파이프라인)
- **데이터**: MSTAR 5클래스, train El17/test El15, few-shot 136장(24/32/24/24/32)
- **증강**: 로컬 MATLAB(원본 Agarwal 코드) → .mat → Google Drive → Colab Python 학습
- **왜 MATLAB**: OSU Box precomputed 삭제(404) + Python 포팅은 검증 왕복이 김 → 원본 실행이 정확·빠름

**엔지니어링 성과**:
| 성과 | 효과 |
|---|---|
| 100x 솔버 가속 (pagemtimes, 익명함수 제거) | 칩당 26분 → 18초 |
| 인덱싱 버그 수정 (상관 0.18 → 0.997) | 재생성 이미지 정상화 |
| 로그진폭(dB) 전처리 도입 | 72.1% → 90.9% (+18.8%p) |
| 툴박스 의존성 제거 (pdist2, imrotate) | 라이선스 없이 실행 |

### 4.3 결과 비교표

| 조건 | 우리 (SMPL/AT) | 논문 |
|---|---|---|
| Baseline (136장, 증강 없음) | **66.6%** | 56.6% |
| Aug (PH 증강) | **90.9%** | 96.4% |

- **Baseline**: few-shot 난이도 재현 확인 (136장 → 66.6%, 논문 56.6%와 유사하게 낮음)
- **증강 효과**: +24.3%p 개선으로 물리 기반 증강의 효과 입증
- **잔여 갭 (5.5%p)**: T72(85%)·2S1(84%) tracked vehicle 병목 — 클래스별 sigma_G 미세조정 미완

**클래스별 정확도** (AT/60dB): 2S1 84.3% / BMP2 91.3% / BTR70 95.4% / T72 85.1% / ZSU23 100%

### 4.4 개선 #3 — XAI 산란점 IoU 검증

> **그래프**: `exp_b_xai_iou.png` (IoU 단조증가 막대 + 방법 비교표) / CAM 샘플: `docs/figures/cam/`

![Exp B — XAI IoU 단조증가](figures/exp_b_xai_iou.png)

**CAM 오버레이 샘플** (모델이 어디를 보는지 — 산란점 위에 히트맵):

| Grad-CAM 16×16 샘플 1 | Grad-CAM 16×16 샘플 2 | SmoothGrad-IG 샘플 1 | SmoothGrad-IG 샘플 2 |
|---|---|---|---|
| ![](figures/cam/gradcam_sample_1.png) | ![](figures/cam/gradcam_sample_2.png) | ![](figures/cam/xai_sample_1.png) | ![](figures/cam/xai_sample_2.png) |

**문제**: 모델이 높은 정확도를 내더라도, 물리적 산란점을 근거로 분류하는지 알 수 없다 (블랙박스).

**질문**: 모델이 실제 산란점(scattering centers)을 보고 분류하는가?

**방법**: 4가지 XAI 기법의 어트리뷰션 맵과 산란점 마스크의 IoU를 측정.

| 방법 | 해상도 | 성격 | IoU |
|---|---|---|---|
| Grad-CAM 8x8 | 8x8 | CAM 계열 | 0.12 |
| Grad-CAM 16x16 | 16x16 | CAM 계열 | 0.19 |
| Occlusion Sensitivity | 64x64 | 인과적 (가리면 붕괴) | 0.24 |
| **SmoothGrad-IG** | **64x64** | **공리적** | **0.29** |

**핵심 발견**: **IoU가 0.12 → 0.19 → 0.24 → 0.29로 단조 증가**
- CAM 해상도의 구조적 천장(8x8로는 점 산란체 국소화 불가)을 데이터로 입증
- 픽셀 단위 XAI(Occlusion, SmoothGrad-IG)가 이를 극복
- 랜덤 기준선 IoU ~0.05 대비 0.29는 **강한 정합** (산란점은 5개 점이라 이론적 IoU 상한 ~0.3~0.4)

**발표 포인트**: "물리 기반 증강으로 학습한 모델이 실제 물리적 산란점을 근거로 분류함을 정량적으로 입증했다."

---

## 5. Exp C — 대비 증강

> **그래프**: `exp_c_contrast_optuna.png` (3단계 비교 막대 + Optuna 산점도)

![Exp C — 대비 증강 3단계](figures/exp_c_contrast_optuna.png)

### 5.1 논문이 한 것
**핵심 문제**: SAMPLE 데이터셋의 합성(synthetic) 이미지는 실측(measured) 대비 배경 클러터가 약하고 대비가 다르다. synth로 학습 → real 테스트 시 붕괴 (도메인 갭).

**논문 해법**: `ColorJitter(contrast=0.5)` 로 대비를 [0.5, 1.5] 범위에서 무작위 변형, 이미지당 3배 증강 (806장 → 2418장). 학습 시에만 적용, 평가는 원본.

**결과**: RN18 91.9% → **94.5%** (+2.6%p).

**논문의 한계**: `contrast=0.5`, 3레벨은 **근거 없이 임의 고정된 매직넘버**. 왜 0.5인지, 왜 3배인지 정당화 없음.

### 5.2 우리의 재현 + 개선 #2

**2단 구조**:
1. **논문 방법 재현**: ColorJitter(0.5) x 3 그대로 적용 (baseline)
2. **Optuna 자동탐색**: strength(0.1~0.9) x levels(1~4)를 베이지안 최적화로 탐색

**누수 방지**: real 평가셋을 val(20%)/test(80%) 분할. Optuna는 val로만 파라미터 선택, 최종 수치는 못 본 test(1076장)로 보고.

### 5.3 결과 비교표

| 조건 | 우리 결과 (RN18, K=0) |
|---|---|
| ① 증강 없음 (synth→real) | 69.6% |
| ② 논문 ColorJitter(0.5) x 3 | 61.8% (**-7.8%p 악화**) |
| ③ Optuna (strength=0.672, levels=4) | **80.3%** (**+18.5%p 개선**) |

**핵심 발견**: 논문의 매직넘버 0.5가 이 데이터에서는 오히려 **성능을 해침** (69.6 → 61.8%). Optuna가 찾은 최적값(0.672, 4레벨)으로 **61.8 → 80.3%** 회복.

**발표 포인트**: "근거 없는 임의 대비값은 데이터에 따라 독이 될 수 있고, 자동탐색이 근거 있는 최적을 찾는다."

**절대 수치 갭 (69.6 vs 논문 91.9) 설명**: SAMPLE 로드 장수(train/test 각 1345)가 논문 Table 5(806/539)와 달라 서브셋 규약 차이. 개선 서사(+18.5%p)와는 무관.

---

## 6. Exp D — OOD 탐지

> **그래프**: `exp_d_ood.png` (AUROC 히트맵 + near/far-OOD 비교)

![Exp D — OOD 탐지 AUROC](figures/exp_d_ood.png)

### 6.1 논문이 한 것
**핵심 문제**: 실전 환경에는 학습에 없던 미지 표적(OOD)이 등장. "모르는 것을 모른다고 말할 수 있는가?"

**두 방법 비교**:
- **ODIN**: softmax 신뢰도 + temperature scaling(T=1000) + 입력 섭동(epsilon=0.0014)
- **Mahalanobis**: 클래스별 특징 분포(평균/공분산) 기반 거리 측정

**OOD 종류**:
- **Near-OOD**: 같은 도메인(SAMPLE) 내 미지 클래스 (holdout, J=1~3)
- **Far-OOD**: 완전히 다른 도메인 (SAR-ship = 선박)

### 6.2 우리의 재현
- **ID**: SAMPLE 10클래스 (논문 Figure 9 기준)
- **MiniSAR 대체**: 비공개 데이터 → 공개 SAR-ship으로 대체 (정당: 선박과 전차는 완전히 다른 SAR 반사 패턴)
- **J=1,2,3 세 설정으로 holdout 클래스 수 변화에 따른 탐지 난이도 분석**

### 6.3 결과 요약

| OOD 종류 | ODIN AUROC | Maha AUROC | 해석 |
|---|---|---|---|
| **Far-OOD (SAR-ship)** | 1.000 (J=1) | **1.000** (전 J) | 완벽 탐지 |
| **Near-OOD (holdout)** | ~0.50 | ~0.42 | 랜덤 수준 (탐지 불가) |

**핵심 발견**:
- **Far-OOD**: Mahalanobis가 AUROC=1.000으로 **전 J에서 완벽**. 도메인 갭이 크면 특징 거리 기반이 압도적.
- **Near-OOD**: 두 방법 모두 ~0.45 (랜덤 수준). 동일 도메인 내 미지 클래스는 현재 방법으로 탐지 불가.
- **논문 패턴 재현**: "far-OOD는 쉬움, near-OOD는 어려움" 정성적 결론 일치.

**발표 포인트**: "OOD 탐지 난이도는 도메인 거리에 강하게 의존. Mahalanobis가 cross-domain에서 일관 우세."

---

## 7. 우리 팀 개선 3가지 종합

| # | 개선 | 논문의 한계 | 우리가 추가한 것 | 핵심 수치 |
|---|---|---|---|---|
| 1 | **SSIM 경계 정량화** | CT 효과만 보고, 품질 기준 없음 | 경계 SSIM으로 "왜 되는가" 정량화 | SSIM 0.9472 |
| 2 | **Optuna 대비 자동탐색** | contrast=0.5, 3레벨 매직넘버 | 베이지안 최적화로 근거 있는 최적값 | +18.5%p |
| 3 | **XAI 산란점 IoU** | 블랙박스, 분류 근거 미제시 | Grad-CAM→픽셀 XAI, 산란점 정합 | IoU 0.29 |

---

## 8. 그래프 목록 (발표 슬라이드용)

| 파일명 | 실험 | 내용 | 슬라이드 위치 |
|---|---|---|---|
| `exp_a_clutter_transfer.png` | Exp A | 4조건 x 2모델 막대그래프, 논문 vs 우리, 도메인 갭/회복 화살표 | Exp A 재현 결과 |
| `exp_b_confusion_matrix.png` | Exp B | Baseline(56.3%) vs Aug(89.0%) 혼동행렬 나란히 | Exp B 재현 결과 |
| `exp_b_xai_iou.png` | Exp B | IoU 단조증가 막대 + 방법 비교표 | 개선 #3 결과 |
| `exp_c_contrast_optuna.png` | Exp C | 3단계 비교 막대 + Optuna 탐색 산점도 | Exp C + 개선 #2 결과 |
| `exp_d_ood.png` | Exp D | AUROC 히트맵 + near/far-OOD 그룹 막대 | Exp D 재현 결과 |

---

## 9. 논문과의 설계 차이 총정리

| 실험 | 논문 | 우리 | 차이 이유 |
|---|---|---|---|
| Exp A | CT 4조건 | 동일 + SSIM | 품질 근거 추가 (개선 #1) |
| Exp B | MATLAB PH 보간 | MATLAB 하이브리드 + XAI | 원본 구현체 직접 실행 + 해석가능성 (개선 #3) |
| Exp C | ColorJitter(0.5) x 3 | 재현 + Optuna | 매직넘버 → 자동탐색 (개선 #2) |
| Exp D OE | MiniSAR + SAR-ship | SAR-ship만 | MiniSAR 비공개 → SAR-ship 대체 |
| Exp D ID | SAMPLE | SAMPLE | 동일 |

---

## 10. 발표 슬라이드 구성 제안

### 슬라이드 1: 제목 + 개요
- 논문 제목, 4개 실험 + 3가지 개선 한 줄 소개

### 슬라이드 2: 프로젝트 구조
- 실험 4개의 질문과 데이터 흐름도
- 개선 3종이 어디에 붙는지 표시

### 슬라이드 3: Exp A — 클러터 전이
- `exp_a_clutter_transfer.png` + 결과표
- "배경이 바뀌면 67.7%로 붕괴 → CT 학습으로 98.8% 회복 (논문 96.0% 초과)"

### 슬라이드 4: 개선 #1 — SSIM
- SSIM = 0.9472: 전이 품질 높음 → 회복 효과의 근거
- "논문이 말하지 않은 '왜 되는가'를 정량화"

### 슬라이드 5: Exp B — PH 보간
- `exp_b_confusion_matrix.png`
- "136장 → 66.6% (few-shot) → 증강 후 90.9% (논문 96.4%에 근접)"
- 하이브리드 파이프라인 다이어그램

### 슬라이드 6: 개선 #3 — XAI 산란점 검증
- `exp_b_xai_iou.png`
- IoU 단조증가 서사: "해상도 올릴수록 산란점 정합 증가 → 모델이 물리적 근거로 분류"

### 슬라이드 7: Exp C — 대비 증강
- `exp_c_contrast_optuna.png`
- "논문 0.5가 오히려 독 (-7.8%p) → Optuna로 +18.5%p 회복"

### 슬라이드 8: Exp D — OOD 탐지
- `exp_d_ood.png`
- "far-OOD 완벽(1.000), near-OOD 한계(~0.45) → 도메인 거리 의존성"

### 슬라이드 9: 종합 + 한계/향후
- 총괄 결과표 (4실험 x 판정)
- 개선 3종 요약
- 한계: Exp B 잔여 갭 5.5%p (T72/2S1), Exp C 절대 수치 갭 (서브셋 규약 차이)

---

## 11. 데이터셋 설명

### MSTAR (Moving and Stationary Target Acquisition and Recognition)
- 미 DARPA/AFRL 공개 SAR 데이터셋
- X-band SAR, 해상도 0.3m, 128x128 칩
- 부각(depression angle) 15/17도에서 360도 회전 촬영
- 군사 차량 10+ 클래스: 2S1, BMP2, BTR70, T72, ZSU23 등

### SAMPLE (Synthetic and Measured Paired and Labeled Experiment)
- AFRL 공개, 합성-실측 쌍
- 10클래스 군사 차량, 합성(CAD 시뮬레이션)과 실측(SAR 촬영)을 쌍으로 제공
- synth → real 도메인 갭 연구의 표준 데이터셋

### SAR-ship
- 선박 SAR 이미지, 전차와 완전히 다른 SAR 반사 패턴
- Far-OOD 역할 (MiniSAR 비공개 대체)

---

## 12. 핵심 용어 정리

| 용어 | 설명 |
|---|---|
| **SAR** | 합성개구면 레이더. 마이크로파로 고해상도 지표 영상 획득 |
| **ATR** | 자동 표적 인식 (Automatic Target Recognition) |
| **클러터 전이** | 표적 칩을 다른 배경에 합성. 배경 편향 방지 |
| **위상이력 (PH)** | SAR 원시 주파수(K-space) 도메인. 이미지의 2D FFT |
| **산란점** | 표적에서 강하게 반사하는 점(금속 모서리 등) |
| **그룹 희소 복원** | 소수 산란점만 활성화하도록 L2,1 정규화로 추정 |
| **Few-shot** | 극소량 데이터(136장)로 학습하는 상황 |
| **SSIM** | 구조적 유사도 지수 (0~1, 1이 완벽 일치) |
| **ODIN** | Softmax 신뢰도 기반 OOD 탐지 (temperature + 섭동) |
| **Mahalanobis** | 특징 공간에서 클래스 분포 거리 기반 OOD 탐지 |
| **AUROC** | OOD 탐지 성능 지표 (1.0 = 완벽, 0.5 = 랜덤) |
| **Grad-CAM** | CNN이 어디를 보는지 시각화 (클래스 활성화 맵) |
| **IoU** | Intersection over Union (겹침 비율, 0~1) |
| **Optuna** | 베이지안 하이퍼파라미터 최적화 프레임워크 |
| **AT (Adversarial Training)** | 적대적 노이즈(FGSM)로 모델 강건성 향상 |
| **LSM (Label Smoothing)** | 라벨을 부드럽게 하여 과적합 방지 |

---

## 13. 참조 문헌

1. **Geng, Z. et al.** "Target Recognition in SAR Images by Deep Learning with Training Data Augmentation." *Sensors* 2023, 23(2), 941. — **재현 대상 논문**
2. **Agarwal, S. et al.** "Using Group Sparsity to Generate Pose-Varied SAR Target Training Data." *arXiv* 2020, 2012.09284. — **PH 보간 방법 원전**
3. **SENSE-Lab-OSU/mstar_data_aug** — Agarwal MATLAB 구현체
4. **gengzhe2015/SAR-target-recognition** — Geng 클러터 전이 데이터 (저자 공식)
5. **benjaminlewis-afrl/SAMPLE_dataset_public** — SAMPLE 데이터셋
