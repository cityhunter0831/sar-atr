# PAPER_SPEC.md — Geng et al. 2023 원문 기반 재현 명세서 (권위 문서)

> **출처**: Geng, Z. et al. "Target Recognition in SAR Images by Deep Learning with Training Data Augmentation." *Sensors* **2023**, 23, 941. (`docs/2b626a7d-sensors2300941.pdf`)
> 이 문서는 **논문 원문의 Table/Figure를 직접 옮긴 것**이며, 코드 구현이 이와 다르면 **논문이 정답**이다.
> 2026-07 논문 정독 후 작성. 이전까지 코드는 추측 기반이라 여러 실험이 논문과 어긋나 있었음 — 아래 "현재 코드와의 차이" 참고.

---

## 논문 전체 구조 (Figure 2)

물리기반 고충실도 SAR 데이터 증강 4종 → open-world SAR-ATR:
1. **Phase-history 보간** (Section 2.1, 4.1) → Exp B
2. **Clutter transfer** (Section 2.2, 4.2) → Exp A
3. **Contrast 기반 증강** (Section 3, 4.3, SAMPLE) → Exp C
4. **OOD 탐지** (Section 4.4, OE=MiniSAR/SAR-ship) → Exp D

핵심 논지: **측정 SAR 데이터가 부족한 상황**에서 물리기반 증강으로 ID 분류 정확도 + OOD 탐지를 모두 개선.

---

## Exp B — Phase Interpolation 증강 (논문 Table 2·3) ⭐ headline

### 실험 설계 (Section 4.1)
- **클래스 5개**: `2S1, BMP2, BTR70, T72, ZSU23` (7개 Mixed Targets 아님!)
- **부각**: **train = El 17°, test = El 15°**
- **핵심 = few-shot**: baseline은 측정 실데이터를 **극소량**만 사용 (아래 Table 2)
- **증강 방식**: PH 도메인 보간으로 **방위각(azimuth) 이웃** 합성. 실이미지 Az=θ+5°에서 Az=θ 합성. 한 샘플당 최대 49×4=196장 생성 가능하나, Aug1은 ±1° 이웃만 사용
- **입력**: **64×64 center-crop** (128 리사이즈 아님!)
- **학습**: 60 epoch, ADAM, lr=1e-3 (epoch 50에 0.0001로 decay), batch 128, 10회 평균
- **손실**: AT(ε=2) 또는 LSM(lblsm=0.1)

### Table 2 — 샘플 수 (물리적으로 이게 핵심)
| Class | MSTAR-R Train | Test | Aug1 Train | Aug2 Train |
|---|---|---|---|---|
| 2S1 | 32 | 274 | 256 | 448 |
| BMP2 | 24 | 195 | 192 | 336 |
| BTR70 | 24 | 196 | 192 | 336 |
| T72 | 24 | 195 | 192 | 336 |
| ZSU23 | 32 | 274 | 256 | 448 |
| **합계** | **136** | **1134** | **1088** | **1904** |

- **MSTAR-R** = 1295장 중 136장만 고른 희소 실데이터 (baseline)
- **MSTAR-Aug1/2** = PH 보간 합성으로 늘린 데이터 (test는 1134로 고정)

### Table 3 — 5-Class 정확도 (%)
| Model | MSTAR-R | Aug1 | Aug2 | Loss |
|---|---|---|---|---|
| Res18 | 83.7±4.35 | 97.4±1.00 | 97.8±0.16 | AT (ε=2) |
| AConv | 68.3±1.22 | 95.6±0.95 | 96.4±0.69 | AT |
| **SMPL** | **56.6±2.33** | **96.4±0.59** | 96.6±0.64 | AT |
| Res18 | 75.1±8.80 | 95.1±3.41 | 95.2±2.61 | LSM |
| AConv | 73.0±5.37 | 97.0±0.74 | 97.2±0.35 | LSM |
| **SMPL** | **61.3±3.37** | **97.6±0.63** | 97.6±0.29 | LSM |

**headline 수치 = SMPL/AT: 56.6% → 96.4% (+39.8%p)**. baseline이 낮은 이유는 **부각 차이가 아니라 학습 샘플이 136장뿐**이기 때문.

### 현재 코드와의 차이 (반드시 수정)
| 항목 | 논문 | 현재 코드 | 수정 필요 |
|---|---|---|---|
| 클래스 | 5개 (2S1,BMP2,BTR70,T72,ZSU23) | 7개 Mixed Targets | ✅ |
| baseline 샘플 | 136장 (희소) | 2049장 전부 | ✅ 결정적 |
| 증강 | azimuth 이웃 PH 보간 | 인접 파일 alpha 보간 | ✅ |
| 입력 | 64×64 crop | 128×128 resize | ✅ |
| 손실 | AT / LSM | CrossEntropy 기본 | ✅ |

---

## Exp A — Clutter Transfer (논문 Table 4)

### 실험 설계 (Section 2.2, 4.2)
- **클래스 5개**: `2S1, BMP2, BTR70, T72, ZSU23`
- **부각**: **train = El 15°, test = El 17°** (Exp B와 반대!)
- **클러터**: 1159개 클러터 칩(128×128, 6종 C1~C6) → 대표 175개 → train pool / test pool 분리. SAR-Bake 픽셀 주석으로 타겟/그림자/클러터 분리 후 클러터만 교체
- **모델**: SMPL7, RN18, AConv, Heiligers

### Table 4 — 5-Class 정확도 (%)
| Model | MSTAR_OR | Train_OR+Test_CT | Train_CT+Test_CT | Train_CTx2+Test_CT |
|---|---|---|---|---|
| SMPL7 | 98.1±0.72 | 38.6±1.17 | 91.5±0.93 | **96.0±1.03** |
| RN18 | 99.8±0.06 | 55.2±1.42 | 97.5±0.65 | **98.4±0.35** |
| AConv | 99.0±0.21 | 54.2±5.87 | 94.1±0.60 | 98.0±0.39 |
| Heiligers | 98.4±0.39 | 54.7±3.39 | 87.1±1.58 | 92.5±1.41 |

핵심: 원본만 학습→CT 테스트 시 급락(38~55%), CT 학습으로 회복(87~97%), **CTx2로 baseline 수준 회복(92~98%)**.

### 현재 코드와의 차이
- ⚠️ **우리 TrainCTx2 결과가 39.4%/41.4%로 붕괴** — 논문은 96.0/98.4로 **회복**. CTx2 증강 로직에 버그 있음 (2배 증강이 이미지를 망치는 중). 수정 필요.
- 부각 train15/test17 확인 필요 (현재 랜덤 80/20일 수 있음).

---

## Exp C — Contrast 기반 증강 (논문 Figure 1, Table 5·6)

### 실험 설계 (Section 3, 4.3)
- **데이터 = SAMPLE** (10 클래스): `2S1, BMP2, BTR70, M1, M2, M35, M548, M60, T72, ZSU23`
- **K 파라미터**: 학습셋 중 **실측(measured) 비율**. K=0 → 100% synthetic 학습 → measured 테스트 (논문 핵심 시나리오)
- **증강**: 복소 데이터에서 **3가지 대비 레벨** 이미지 생성 → 806×3 = 2418 학습 샘플
- **모델**: RN18, SMPL7, AConv, Heiligers

### Table 5 — SAMPLE 샘플 수
합계 **train 806 / test 539**. (2S1 116/58, BMP2 55/52, BTR70 43/49, M1 78/51, M2 75/53, M35 76/53, M548 75/53, M60 116/60, T72 56/52, ZSU23 116/58)

### Table 6 — 대비 증강 정확도 (%) [K=0, 100% synthetic → measured]
| Model | SAMPLE(Ori) | SAMPLE(Aug) |
|---|---|---|
| **RN18** | 91.9±2.17 | **94.5±1.32** |
| SMPL7 | 86.7±3.41 | 91.8±2.00 |
| AConv | 86.5±1.77 | 89.6±1.62 |

(K=0.05: RN18 96.9→98.1 / K=0.1: RN18 97.8→98.9)

### Figure 1 — MSTAR 대비 실험 (보조)
ResNet18, train El17°→test El30°: 원본(밝기 편향) 97.2% / 대비보정 테스트 65.3% / 대비보정 학습 **88.5%**

### 현재 코드와의 차이
- 대체로 방향 맞음 (SAMPLE synth→measured, ResNet18). **목표치 = K=0에서 RN18 94.5%** 로 명시할 것.
- Optuna는 우리 개선(논문엔 없음). 논문은 대비 레벨 3개 고정.

---

## Exp D — OOD 탐지 (논문 Section 4.4, Figure 9)

### 실험 설계
- **ID 데이터 = SAMPLE 10 클래스** (#0~#9 = 2S1,BMP2,BTR70,M1,M2,M35,M548,M60,T72,ZSU23), K=0.1
- **OE 학습셋 #1** = SAR-ship 2048장 + MiniSAR 443장 (총 2491, [17] 대비 4%)
- **OOD 테스트 3종**:
  1. **Holdout**: SAMPLE에서 J개 클래스를 학습에서 제외 (J=1,2,3)
  2. **MSTAR-O**: 공식 raw 기반 5클래스 (BRDM2,BTR60,D7,T62,ZIL131), 1290장
  3. **MSTAR-P**: 공개 웹 이미지 기반 (그림자 흐림, 클러터 부분 제거) 동일 5클래스
- **탐지기**: baseline(softmax threshold), **ODIN**(T=1000), **AdvOE(ε=8) = Mahalanobis 기반**
- **지표**: AUROC, TNR@95TPR

### Figure 9 결과 (J=1, 근사값)
| OOD | baseline AUROC | ODIN AUROC | Maha AUROC |
|---|---|---|---|
| Holdout | ~60 | ~76 | ~93 |
| MSTAR-O | ~94 | ~99 | ~99 |
| MSTAR-P | ~97 | ~100 | ~100 |

핵심: **Mahalanobis(AdvOE)가 대부분 최고**. cross-dataset(MSTAR-O/P)은 쉬움, holdout(같은 SAMPLE 내 미지 클래스)은 어려움. M35(#5)+M548(#6) 동시 holdout 시 TNR 100% (외형이 독특).

### 현재 코드와의 차이 (큼)
| 항목 | 논문 | 현재 코드 |
|---|---|---|
| ID 데이터 | **SAMPLE 10클래스** | MSTAR Mixed 10클래스 |
| OOD 테스트 | Holdout + MSTAR-O + MSTAR-P | Holdout + SAR-ship |
| SAR-ship 역할 | **OE 학습셋** | OOD 테스트셋 |
- 우리는 SAR-ship을 OOD 테스트로 썼지만, 논문은 **OE 학습용**. ID도 SAMPLE이어야 함. 재설계 필요.

---

## 우리 팀 개선 3가지 (논문에 없는 추가 기여)
1. **SSIM 경계 아티팩트 정량화** — clutter transfer 품질 측정 (Exp A). ✅ 구현됨
2. **옵티마이저 비교** — 논문은 ADAM 고정. 우리는 ADAM vs SGD 비교로 확장 (주의: 논문이 이미 ADAM이므로 "ADAM 도입"이 아니라 "대안 비교")
3. **Grad-CAM × 산란점 IoU 검증** — 모델이 물리적 산란점을 보는지 (Exp B). ✅ 구현됨. 단 Exp B 재설계 후 재검증 필요.
4. **Optuna 하이퍼파라미터 탐색** (Exp C) — 논문은 고정값. ✅ 구현됨

---

## 우선순위 (재현 정확도 순)
1. 🔴 **Exp B 재설계**: 5클래스 + 136장 few-shot baseline + azimuth PH 보간 + 64×64 crop + AT/LSM loss
2. 🔴 **Exp A CTx2 버그**: 붕괴(39%) → 논문은 회복(96%)
3. 🟠 **Exp D 재설계**: ID=SAMPLE, SAR-ship=OE, MSTAR-O/P를 OOD로
4. 🟡 **Exp C**: 목표치 K=0 RN18 94.5% 명시, 대체로 유지
