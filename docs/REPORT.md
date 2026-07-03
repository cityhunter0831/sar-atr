# SAR-ATR 재현 및 개선 보고서

> Geng et al. 2023, *"Target Recognition in SAR Images by Deep Learning with Training Data Augmentation"* 재현 및 개선 과제
> 본 문서는 각 실험에 대해 **[논문 실험 설명] → [우리 재현 방법: 데이터셋·툴] → [방법론적 선택 이유] → [결론 및 논문 대비 비교]** 순으로 정리한다.

---

## 요약 표

| 실험 | 논문 지표 | 우리 결과 | 판정 |
|---|---|---|---|
| Exp A Table 4 (클러터 전이) | 도메인 갭 → CT로 회복 | 66.7% → 91.1% (SMPL), 81.6% → 99.5% (ResNet) | ✅ 재현 |
| Exp A SSIM (개선 #1) | — (논문에 없음) | 0.9472 ± 0.0024 | ✅ 신규 |
| Exp B Table 3 (PH 보간) | 56.6% → 96.6% | 86.1% → 98.6% | ✅ 재현 |
| Exp C Figure 1 (대비 보정) | 65.3% → ≥88.5% | 73.6% → 80.9% | 🟡 부분 |
| Exp D OOD (ODIN vs Maha) | OOD 탐지 | far-OOD 완벽, near-OOD 한계 | ✅ 확장 |

---

## Exp A — 클러터 전이 (Clutter Transfer, 논문 Table 4)

### 논문 실험
SAR 타겟 chip을 서로 다른 배경 클러터에 합성(전이)하여 학습 데이터를 증강. 모델이 배경이 아닌 **타겟 자체**를 학습하도록 유도하여, 배경이 바뀌어도(도메인 갭) 견고하게 인식하는지 검증.

### 우리 재현 방법
- **데이터셋**: gengzhe2015 공개 데이터 (Original MSTAR Images + Train_OR_Test_CT / Train_CT_Test_CT / Train_CTx2_Test_CT), 5개 클래스 (2s1, bmp2, btr70, t72, zsu23)
- **모델**: SMPL (논문 경량 CNN) + ResNet18 (일반 심층 모델), 각 3 seed 평균
- **툴**: feather blending 기반 `clutter_transfer()`, 4개 조건 × 2 모델 × 3 seed

### 방법론적 선택 이유
논문의 헤드라인 모델(SMPL)과 일반적 심층 모델(ResNet18)을 **함께 비교**하여, 클러터 전이 효과가 특정 아키텍처에 국한되지 않고 일반적으로 유효함을 입증. 3 seed 평균으로 통계적 신뢰성 확보.

### 결론 및 논문 대비 비교
| 조건 | SMPL | ResNet18 |
|---|---|---|
| MSTAROR (원본→원본) | 93.3±4.2% | 100.0±0.0% |
| TrainOR+TestCT (원본학습→CT테스트) | 66.7±4.0% | 81.6±1.9% |
| TrainCT+TestCT (CT학습→CT테스트) | 91.1±1.8% | 99.5±0.3% |
| TrainCTx2+TestCT | 39.4±1.5% | 41.4±1.2% |

- **핵심 재현 성공**: 원본으로만 학습 시 클러터 전이 테스트에서 성능 하락(도메인 갭, SMPL 93→67%), 클러터 전이로 학습하면 회복(67→91%). 논문의 핵심 주장을 두 모델 모두에서 재현.
- **시사점**: `TrainCTx2`(CT 2배 증강)는 오히려 39%로 붕괴 — 과도한 증강이 이미지 품질을 훼손할 수 있음을 관찰 (추가 분석 필요).

---

## Exp B — 위상이력(PH) 보간 증강 (논문 Table 3)

### 논문 실험
두 SAR 이미지의 **위상이력(Phase History) 도메인**을 선형 보간하여 중간 시점의 합성 이미지를 생성. 학습 데이터를 조밀하게 만들어 부각(depression angle) 차이로 인한 도메인 갭을 극복.

### 우리 재현 방법
- **데이터셋**: MSTAR Mixed Targets **CD1 + CD2 raw binary** (복소 신호 보존), 7개 클래스 (2S1, BRDM_2, BTR_60, D7, T62, ZIL131, ZSU_23_4)
- **툴**: `interpolate_phase_history()` — IFFT → PH 도메인 선형보간 → FFT, Taylor 윈도우(sll=35) 적용. `read_mstar_complex()`로 복소 데이터 파싱
- **분할**: 표준 MSTAR SOC — **train 17° / test 15°** (헤더 `DesiredDepression` 기반 cross-depression split), train 2049장 / test 1838장

### 방법론적 선택 이유
- **왜 raw binary인가**: PH 보간은 복소(real+imag) 신호가 필수. PNG 이미지는 위상 정보가 소실되어 사용 불가 → Mixed Targets raw만이 유효.
- **왜 CD1+CD2 둘 다인가**: 15°는 CD1에만, 17°는 CD2에만 존재. 표준 SOC(17°→15°) 재현을 위해 두 디스크 병합 필수 (진단 교차표로 7개 클래스 모두 양쪽 존재 확인).
- **왜 cross-depression split인가**: 같은 부각 내 랜덤 분할 시 방위각만 다른 유사 이미지로 99%+ trivial 정확도 → 논문 재현 불가. 부각을 분리해야 실제 도메인 갭이 발생.

### 결론 및 논문 대비 비교
| 조건 | 우리 결과 | 논문 |
|---|---|---|
| 원본만 학습 | 86.1% | 56.6% |
| PH 보간 증강 추가 | **98.6%** | **96.6%** |

- **핵심 재현 성공**: 증강으로 86.1% → 98.6% (+12.5%p) 상승. 최종 성능 98.6%가 논문 96.6%와 거의 일치.
- **baseline 차이 해석**: 우리 원본만(86.1%)이 논문(56.6%)보다 높음 — 17°→15°는 부각 차가 작아(2°) 증강 없이도 어느 정도 일반화. 논문 대비 데이터셋 난이도 차이. **핵심은 baseline 절대값이 아니라 증강 효과와 최종 성능이며, 이는 논문을 충실히 재현.**

---

## Exp C — 대비 보정 (Contrast Balance, 논문 Figure 1)

### 논문 실험
합성(synthetic) 데이터로 학습한 모델을 실측(measured) 데이터에서 테스트하면 명암/대비 특성 차이로 성능이 하락. **대비 보정(CLAHE)**으로 이 도메인 갭을 완화.

### 우리 재현 방법
- **데이터셋**: **SAMPLE 데이터셋** (benjaminlewis-afrl/SAMPLE_dataset_public) — synthetic → measured, 10개 클래스
- **툴**: CLAHE 기반 `ContrastBalance` + **Optuna 하이퍼파라미터 자동 탐색** (20 trials), ResNet18
- **평가**: synth로 학습 → real로 테스트 (Figure 1 원본 방법)

### 방법론적 선택 이유 (⭐ 논문과 가장 크게 다른 부분)
- **왜 MSTAR El 교차 대신 SAMPLE인가**: 논문 원본은 MSTAR El=17°→30° 교차 부각을 썼으나, 이는 "부각 차이" 문제라 대비 보정과의 인과가 약함. **SAMPLE은 동일 타겟의 시뮬레이션·실측 쌍**을 제공하여, synth-real 간 **명암/대비 도메인 갭**이 정확히 대비 보정으로 풀어야 할 문제 → 대비 보정 실험의 더 적합한 표준 벤치마크.
- **왜 Optuna인가**: 대비 보정 파라미터(clip_limit, tile_grid_size, global_norm)를 수동 조정 대신 자동 최적화 → 재현성·객관성 강화 (논문에 없는 방법론적 개선).
- MSTAR El=17°→30° 교차 실험은 `run_el_ablation()`으로 **보조 ablation** 유지.

### 결론 및 논문 대비 비교
| 조건 | 우리 결과 | 논문/목표 |
|---|---|---|
| 증강 없음 (synth→measured) | 73.6% | ~65.3% |
| 대비 보정(CLAHE) 적용 | 80.9% | 목표 ≥88.5% |
| Optuna 최적 파라미터 | clip_limit=1.99, tile=4, global_norm=True | — |

- **방향성 재현**: 대비 보정으로 73.6% → 80.9% (+7.3%p) 상승. 대비 보정이 synth-real 갭을 줄인다는 논문 주장 확인.
- **시사점**: 목표 88.5%엔 미달. Optuna trial 수·epoch 증가로 개선 여지. SAMPLE은 MSTAR El 교차보다 도메인 갭이 근본적으로 커서 완전 극복이 더 어려움 — 대비 보정만으로는 한계가 있다는 시사점.

---

## Exp D — OOD 탐지 (ODIN vs Mahalanobis)

### 논문 실험
학습하지 않은(Out-of-Distribution) 입력을 탐지. 알려진 클래스 일부를 숨기고(holdout), 다른 도메인 데이터를 OOD로 사용하여 탐지 성능(AUROC, TNR@95TPR) 측정.

### 우리 재현 방법
- **데이터셋**: MSTAR 10클래스 (ID) + 숨긴 클래스 J=1,2,3 (near-OOD holdout) + **SAR-ship** (far-OOD, cross-domain)
- **툴**: **ODIN** (temperature scaling T=1000 + input perturbation ε=0.0014) vs **Mahalanobis** (클래스별 평균·공분산 거리), `get_features()`로 특징 추출
- **평가**: AUROC, TNR@95TPR

### 방법론적 선택 이유 (논문 확장)
- **왜 두 방법 비교인가**: ODIN(softmax 기반)과 Mahalanobis(특징 거리 기반)는 서로 다른 원리 → 어떤 방법이 어떤 OOD 상황에 강한지 비교 분석 (논문을 넘어선 확장).
- **왜 SAR-ship 추가인가**: holdout(같은 도메인 미지 전차)은 near-OOD, SAR-ship(다른 도메인)은 far-OOD. 두 난이도를 분리하여 OOD의 본질적 난이도 스펙트럼을 드러냄.

### 결론 및 논문 대비 비교
| J | 방법 | OOD | AUROC | TNR@95 |
|---|---|---|---|---|
| 1 | ODIN | holdout | 0.580 | 0.108 |
| 1 | ODIN | sarship | 0.953 | 0.670 |
| 1 | Mahalanobis | holdout | 0.453 | 0.052 |
| 1 | Mahalanobis | sarship | **1.000** | **1.000** |
| 2 | ODIN | holdout | 0.468 | 0.017 |
| 2 | ODIN | sarship | 0.996 | 0.990 |
| 2 | Mahalanobis | holdout | 0.471 | 0.069 |
| 2 | Mahalanobis | sarship | **1.000** | **1.000** |
| 3 | ODIN | holdout | 0.770 | 0.142 |
| 3 | ODIN | sarship | 0.999 | 0.986 |
| 3 | Mahalanobis | holdout | 0.450 | 0.058 |
| 3 | Mahalanobis | sarship | **1.000** | **1.000** |

- **Far-OOD (SAR-ship)**: 두 방법 모두 우수, 특히 **Mahalanobis가 AUROC 1.000 완벽**. 도메인 갭이 클 때 특징 거리 기반이 매우 강력.
- **Near-OOD (holdout 전차)**: 두 방법 모두 랜덤(0.5) 수준으로 실패. 학습한 전차와 유사한 미지 전차는 최신 방법으로도 탐지 어려움.
- **시사점**: OOD 탐지 난이도는 **도메인 거리에 강하게 의존**. far-OOD엔 Mahalanobis 우세, near-OOD는 두 방법 모두 근본적 한계 — near-OOD가 far-OOD보다 훨씬 어렵다는 OOD 문헌의 정설을 실증.

---

## 우리 팀 개선 3가지 (논문에 없는 추가 기여)

| # | 개선 | 위치 | 결과 |
|---|---|---|---|
| 1 | **SSIM 경계 아티팩트 정량화** | `run_boundary_ssim_analysis()` (exp_a) | SSIM 0.9472 — 클러터 전이 경계 왜곡 최소, 타겟 구조 충실 보존 정량 입증 |
| 2 | **Adam 옵티마이저 비교** | `core/train.py` | — |
| 3 | **Grad-CAM × 산란점 IoU 검증** | `run_gradcam_analysis()` (exp_b) | 모델이 실제 물리적 산란점을 보고 분류하는지 검증 (진행 예정) |

---

## 논문과의 설계 차이 총정리

| 실험 | 논문 | 우리 | 차이 성격 |
|---|---|---|---|
| Exp A | 클러터 전이 (Table 4) | 동일 재현 + SMPL/ResNet 비교 | 충실 재현 |
| Exp B | PH 보간 (Table 3) | 동일 재현, CD1+CD2 병합 SOC split | 충실 재현 |
| Exp C | MSTAR El 교차 | **SAMPLE 데이터셋으로 교체 + Optuna** | **데이터셋·방법 교체** |
| Exp D | OOD 탐지 | **ODIN vs Mahalanobis 비교 + SAR-ship** | **방법론 확장** |
