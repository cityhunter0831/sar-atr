# SAR-ATR 재현 및 개선 보고서

> Geng et al. 2023, *"Target Recognition in SAR Images by Deep Learning with Training Data Augmentation"* 재현 및 개선 과제
> 본 문서는 각 실험에 대해 **[논문 실험 설명] → [우리 재현 방법: 데이터셋·툴] → [방법론적 선택 이유] → [결론 및 논문 대비 비교]** 순으로 정리한다.

---

> ⚠️ **2026-07 갱신**: Exp B는 초기에 잘못된 설계(7클래스 전체 데이터, 선형보간)로 86.1%→98.6%가 나왔었으나, 논문 재분석 후 **few-shot(136장) + 5클래스**로 재설계. 위상보간도 산란점 기반 완전판으로 교체 중(진행 중, 아래 참조). Exp A는 CTx2 조건 붕괴 원인 미확정.

## 요약 표

| 실험 | 논문 지표 | 우리 결과 | 판정 |
|---|---|---|---|
| Exp A Table 4 (클러터 전이) | 도메인 갭 → CT로 회복 | 66.7%→91.1%(SMPL) 재현, **CTx2 39.4% 붕괴(논문 96.0%, 원인 미확정)** | 🟠 부분 |
| Exp A SSIM (개선 #1) | — (논문에 없음) | 0.9472 ± 0.0024 | ✅ 신규 |
| Exp B Table 3 (PH 보간, few-shot 136장) | 56.6% → 96.4% (SMPL/AT) | **49.8% → 64.9%** (선형보간, 완전판 구현 중) | 🟠 진행 중 |
| Exp C Figure 1/Table6 (대비 보정) | 65.3%→88.5% / 91.9%→94.5%(K=0) | 73.6% → 80.9% | 🟡 부분 |
| Exp D OOD (ODIN vs Maha, ID=SAMPLE) | Fig9: far-OOD쉬움/near-OOD어려움 | far-OOD 완벽, near-OOD 한계 (재설계 후 재검증 필요) | 🟠 재검증 필요 |

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
| 조건 | 우리 SMPL | 논문 SMPL7 | 우리 ResNet18 | 논문 RN18 |
|---|---|---|---|---|
| MSTAROR (원본→원본) | 93.3±4.2% | 98.1±0.72% | 100.0±0.0% | 99.8±0.06% |
| TrainOR+TestCT (원본학습→CT테스트) | 66.7±4.0% | 38.6±1.17% | 81.6±1.9% | 55.2±1.42% |
| TrainCT+TestCT (CT학습→CT테스트) | 91.1±1.8% | 91.5±0.93% | 99.5±0.3% | 97.5±0.65% |
| TrainCTx2+TestCT | **39.4±1.5%** ⚠️ | **96.0±1.03%** | **41.4±1.2%** ⚠️ | **98.4±0.35%** |

- **부분 재현**: 원본으로만 학습 시 클러터 전이 테스트에서 하락(도메인 갭) → 클러터 전이로 학습하면 회복(TrainCT+TestCT는 논문과 근접: 91.1% vs 91.5%). **핵심 갭 발생·일부 회복 패턴은 재현됨.**
- **🔴 미해결**: `TrainCTx2+TestCT`(CT 2배 증강)가 논문은 **96.0%로 완전 회복**하는데 우리는 **39.4%로 오히려 최악**. 정반대 결과라 원인 규명 필요 (T5). 진단 로깅(`[T5 진단]`, 폴더별 로드 수) 코드에 추가됨 — Colab 재실행해 폴더 로드 수 확인 필요.
- **가설**: (1) `Train_CTx2_Test_CT` 폴더 로딩 실패로 빈 데이터 학습, (2) 클러터 전이 2배 적용 시 이미지 손상. 진단 로그로 확정 예정.

---

## Exp B — 위상이력(PH) 보간 증강 (논문 Table 2·3) ⭐ 진행 중 (재설계됨)

### 논문 실험
**핵심은 few-shot**: 실측 SAR 데이터가 클래스당 24~32장(총 136장)뿐인 극히 적은 상황에서, 산란점 희소성(Agarwal et al. 방법)을 이용해 단일 실이미지로부터 인접 방위각(±6°) 이미지를 물리적으로 재합성 → 학습 데이터를 1088장으로 증강. 부각 차이(17°→15°) 자체가 목적이 아니라, **표본 부족을 물리 기반 증강으로 극복**하는 것이 논문의 주제("Training Data Augmentation").

### 우리 재현 방법 (2차 개정)
- **데이터셋**: MSTAR raw — Targets 패키지(BMP2·BTR70·T72) + Mixed Targets CD1/CD2(2S1·ZSU23), **5클래스**(2S1,BMP2,BTR70,T72,ZSU23), train El17°/test El15°
- **few-shot 제한**: 클래스별 24~32장(`FEWSHOT_COUNTS`)만 무작위 추출해 baseline 총 136장 구성 (논문 Table 2 그대로)
- **입력**: 64×64 center-crop (128 아님)
- **손실**: AT(ε=2, FGSM 적대적 학습)
- **증강(1차, 임시)**: `interpolate_phase_history()` — 두 실이미지의 위상이력을 azimuth 이웃 기준 **선형 평균**. Taylor 윈도우(sll=35) 적용
- **증강(2차, 완전판 구현 중)**: Agarwal 방법 포팅 — ①이미지→PH 극좌표 변환 ②그룹 희소 복원(FISTA, Eq.7)으로 산란점 계수 추정 ③임의 방위각 재합성(Eq.8). `augmentation/ph_sparse.py`

### 방법론적 선택 이유
- **왜 5클래스로 축소·재구성**: 초기 구현이 Mixed Targets 7클래스 전체(2049장)로 학습해 few-shot 취지를 놓침(98% trivial). 논문 Table 2의 정확한 5클래스·136장으로 재설계.
- **왜 선형보간이 틀렸는가**: Agarwal 논문 Fig.4b가 선형보간 방식[37]이 산란점을 왜곡시켜 실제와 다른 이미지를 만듦을 직접 실증. 우리 1차 구현이 이 오류를 그대로 재현해 성능이 낮았음(65% 정체).
- **왜 산란점 기반 완전판으로 교체하는가**: 논문(Geng)이 "we adopt the method proposed in Agarwal et al."로 명시 — 정답 구현체는 선형보간이 아니라 산란점 희소모델. `docs/THEORY_REFERENCES.md` 참조.

### 결론 및 논문 대비 비교 (현재까지, 진행 중)
| 조건 | 우리 결과 (few-shot, 선형보간) | 논문 SMPL/AT |
|---|---|---|
| MSTAR-R (few-shot 136장, 증강 없음) | **49.8%** | 56.6% |
| MSTAR-Aug (PH 증강 추가) | **64.9%** | **96.4%** |

- **Baseline(few-shot) 재현 근접**: 49.8% vs 논문 56.6% — few-shot 설계(136장, 5클래스, 17°/15°)가 논문과 유사한 난이도를 재현함을 확인. **이전 오류(98%)가 few-shot 미적용 때문이었음을 검증.**
- **증강 효과 방향은 맞으나 크기 부족**: 49.8%→64.9%(+15.1%p)로 개선은 확인되나 논문의 96.4%엔 크게 못 미침. 원인: 선형보간이 산란점 모델을 쓰지 않아 물리적으로 부정확한 합성 이미지를 생성.
- **진행 상황**: 산란점 기반 완전판 1·2단계(PH 변환, 희소 복원) Python 포팅·검증 진행 중.
  - 1단계(이미지→PH 변환) round-trip 검증: 실데이터 center-crop 64 기준 |corr|=0.99 통과.
  - 2단계(그룹 희소 복원, FISTA): 연산자 adjoint 정합 확인(오차 4e-8), λ 스케일링(λ_max 상대값)으로 산란점 희소화 확인 중.
  - 3단계(방위각 외삽 재합성) 및 최종 few-shot 재학습은 미완료.

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

## Exp D — OOD 탐지 (ODIN vs Mahalanobis, 논문 Figure 9)

> ⚠️ **재설계됨(T6)**: 아래 수치는 **ID=MSTAR 10클래스**로 실행한 구버전 결과. 논문은 **ID=SAMPLE 10클래스**, OE(SAR-ship)는 OOD 테스트가 아니라 outlier-exposure 학습 재료로 사용. 코드는 `SampleDataset` 기반으로 재설계 완료(T6) — **ID=SAMPLE 기준 재실행 및 수치 갱신 필요.**

### 논문 실험
학습하지 않은(Out-of-Distribution) 입력을 탐지. ID=SAMPLE 10클래스로 학습, 일부 클래스를 숨기고(holdout, near-OOD), MSTAR-O/P(cross-dataset, far-OOD)로 탐지 성능(AUROC, TNR@95TPR) 측정. OE(outlier exposure) 학습에는 SAR-ship+MiniSAR 사용.

### 우리 재현 방법 (재설계 후)
- **데이터셋**: **ID = SAMPLE 10클래스** (synth 학습/real 테스트, K=0), Holdout = SAMPLE 일부 클래스(J=1,2,3, near-OOD), **SAR-ship** = far-OOD 겸 OE 대체재(MiniSAR 비공개 대체)
- **툴**: **ODIN** (T=1000, ε=0.0014) vs **Mahalanobis** (클래스별 평균·공분산 거리, `_SubsetWithNames`로 `class_names` 보존), `get_features()`
- **평가**: AUROC, TNR@95TPR

### 방법론적 선택 이유 (논문 확장 + 필수 정정)
- **ID를 SAMPLE로 정정한 이유**: 논문 Figure 9의 실험 설계가 ID=SAMPLE이므로, MSTAR를 ID로 쓰면 다른 실험이 됨. 재현 정확성을 위해 정정.
- **왜 두 방법 비교인가**: ODIN(softmax 기반)과 Mahalanobis(특징 거리 기반)는 서로 다른 원리 → 어떤 방법이 어떤 OOD 상황에 강한지 비교 분석 (논문을 넘어선 확장).
- **왜 SAR-ship인가**: MiniSAR가 논문 저자 자체 개발 비공개 데이터라 공개된 SAR-ship으로 대체 (far-OOD/OE 역할, 문헌 표준).

### 결론 및 논문 대비 비교 (구버전 결과 — ID=MSTAR, 재검증 대상)
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
- **시사점(패턴은 유효할 것으로 예상)**: OOD 탐지 난이도는 **도메인 거리에 강하게 의존**. far-OOD엔 Mahalanobis 우세, near-OOD는 두 방법 모두 근본적 한계 — 이 정성적 패턴은 ID=SAMPLE로 재실행해도 유지될 가능성이 높으나, **정확한 수치는 재실행 후 교체 필요**.

---

## 우리 팀 개선 3가지 (논문에 없는 추가 기여)

| # | 개선 | 위치 | 결과 |
|---|---|---|---|
| 1 | **SSIM 경계 아티팩트 정량화** | `run_boundary_ssim_analysis()` (exp_a) | SSIM 0.9472 — 클러터 전이 경계 왜곡 최소, 타겟 구조 충실 보존 정량 입증 |
| 2 | **옵티마이저 비교 (ADAM vs SGD)** | `core/train.py` | 논문은 ADAM 고정 — 증강 효과의 옵티마이저 의존성 분석 (결과 수집 예정) |
| 3 | **Grad-CAM × 산란점 IoU 검증** | `run_gradcam_analysis()` (exp_b) | 초기(선형보간, 10epoch) 모델 기준 coverage/IoU 측정 완료. **완전판(산란점 기반) 모델로 재실행 필요** — 지금 모델은 few-shot 재설계 전 버전 |

---

## 논문과의 설계 차이 총정리

| 실험 | 논문 | 우리 | 차이 성격 | 상태 |
|---|---|---|---|---|
| Exp A | 클러터 전이 (Table 4) | 동일 재현 + SMPL/ResNet 비교 | 충실 재현 | 🟠 CTx2 원인 미확정 |
| Exp B | PH 보간 few-shot (Table 2·3) | 5클래스·136장 few-shot으로 재설계, 산란점 기반 완전판 구현 중 | 충실 재현 지향 (완전판 진행 중) | 🟠 진행 중 |
| Exp C | MSTAR El 교차(도입부) / SAMPLE(주실험) | **SAMPLE 데이터셋(논문 주실험과 동일) + Optuna 추가** | 주실험 그대로 + 방법 확장 | ✅ |
| Exp D | OOD 탐지 (ID=SAMPLE) | ID=SAMPLE로 재설계 완료, **ODIN vs Mahalanobis 비교 + SAR-ship(MiniSAR 대체)** | 정합 재현 + 방법론 확장 | 🟠 재실행 필요 |

> 자세한 근거는 `docs/DATASET_METHOD.md`(데이터셋 정당성), `docs/PAPER_SPEC.md`(논문 원문 수치), `docs/THEORY_REFERENCES.md`(이론·구현난점) 참조.
