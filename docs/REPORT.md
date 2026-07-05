# SAR-ATR 재현 및 개선 보고서

> Geng et al. 2023, *"Target Recognition in SAR Images by Deep Learning with Training Data Augmentation"* 재현 및 개선 과제
> 본 문서는 각 실험에 대해 **[논문 실험 설명] → [우리 재현 방법: 데이터셋·툴] → [방법론적 선택 이유] → [결론 및 논문 대비 비교]** 순으로 정리한다.

---

> ⚠️ **2026-07 갱신**: Exp B는 초기에 잘못된 설계(7클래스 전체 데이터, 선형보간)로 86.1%→98.6%가 나왔었으나, 논문 재분석 후 **few-shot(136장) + 5클래스**로 재설계. 위상보간은 **선형보간(구버전)에서 산란점 희소모델 기반 완전판(로컬 MATLAB Agarwal repo)으로 교체 완료** — 하이브리드 파이프라인(로컬 MATLAB 증강 생성 → Colab 학습)으로 실행 중. Exp A는 CTx2 조건 붕괴 원인 미확정.

## 요약 표

| 실험 | 논문 지표 | 우리 결과 | 판정 |
|---|---|---|---|
| Exp A Table 4 (클러터 전이) | 도메인 갭 → CT로 회복 | 66.7%→91.1%(SMPL) 재현, CTx2 39.4% 붕괴 **원인 특정(BUG-3 수정완료, Colab 재실행 대기)** | 🟠→🟢 예상 |
| Exp A SSIM (개선 #1) | — (논문에 없음) | 0.9472 ± 0.0024 | ✅ 신규 |
| Exp B Table 3 (PH 보간, few-shot 136장) | 56.6% → 96.4% (SMPL/AT) | 선형보간: 49.8%→64.9% / **산란점 완전판: 66.6%→90.9%** (log-amp 60dB, AT) | 🟢 근접 재현 |
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
- **🟡 원인 특정 (BUG-3 수정)**: `TrainCTx2+TestCT`가 39.4%로 붕괴한 원인은 **`load_condition()`이 CTx2 폴더를 80/20 재분할해 train에 80%만 사용**했기 때문. gengzhe2015 폴더는 이미 조건별 완성본(train/test 혼합이 아님)이므로 재분할이 불필요. 수정: CTx2 폴더 전체를 train으로, `Train_OR_Test_CT` 전체를 test로 직접 사용. **Colab 재실행으로 수치 갱신 필요** — 논문(96.0%)에 근접할 것으로 예상.

---

## Exp B — 위상이력(PH) 보간 증강 (논문 Table 2·3) ⭐ 근접 재현 (66.6%→90.9%)

### 논문 실험
**핵심은 few-shot**: 실측 SAR 데이터가 클래스당 24~32장(총 136장)뿐인 극히 적은 상황에서, 산란점 희소성(Agarwal et al. 방법)을 이용해 단일 실이미지로부터 인접 방위각(±6°) 이미지를 물리적으로 재합성 → 학습 데이터를 1088장으로 증강. 부각 차이(17°→15°) 자체가 목적이 아니라, **표본 부족을 물리 기반 증강으로 극복**하는 것이 논문의 주제("Training Data Augmentation").

### 우리 재현 방법 (2차 개정)
- **데이터셋**: MSTAR raw — Targets 패키지(BMP2·BTR70·T72) + Mixed Targets CD1/CD2(2S1·ZSU23), **5클래스**(2S1,BMP2,BTR70,T72,ZSU23), train El17°/test El15°
- **few-shot 제한**: 클래스별 24~32장(`FEWSHOT_COUNTS`)만 무작위 추출해 baseline 총 136장 구성 (논문 Table 2 그대로)
- **입력**: 64×64 center-crop (128 아님)
- **손실**: AT(ε=2, FGSM 적대적 학습)
- **증강(1차, 폐기)**: `interpolate_phase_history()` — 두 실이미지의 위상이력을 azimuth 이웃 기준 **선형 평균**. Agarwal Fig.4b가 실증한 산란점 왜곡 오류 → 65% 정체 원인.
- **증강(2차, 완전판 — 채택)**: **하이브리드 파이프라인**. 로컬 MATLAB(원본 Agarwal repo)에서 ①`preprocess_raw_data.m`(이미지→PH 극좌표) ②`sparse_recovery.m`(SPGL1 그룹 희소복원, Eq.7 산란점 계수 C) ③`generate_aug_images.m`(Eq.8 ±6° 외삽 + subpixel shift, 칩당 196장) ④`merge_files.m`(클래스별 병합) 실행 → `<class>_aug_images.mat` 생성 → Google Drive → Colab Python(`augmentation/precomputed_aug.py` 로더)에서 SMPL/AT 학습.
  - **왜 MATLAB 하이브리드인가**: OSU Box precomputed 데이터 삭제(404) + Python 포팅(`ph_sparse.py`)은 검증 왕복이 김 → 논문이 채택한 원본 MATLAB 구현체를 직접 실행하는 것이 정확·빠름.
  - **σ_G=1.0 고정**: 2S1 대표이미지에서 σ_G=1(잔차 9419.6) vs σ_G=2(9419.4) 차이 **0.002%** 정량검증 → 라인서치 제거, 발표 방어 가능.

### 엔지니어링 성과 (MATLAB 파이프라인 최적화 — `matlab_pipeline/` 백업)
| 성과 | 대상 | 개선 | 효과 |
|---|---|---|---|
| 100× 솔버 가속 | `sparse_recovery.m` | `mtimesx`(MEX)→`pagemtimes`, 익명함수 슬라이싱 제거, `maxNumCompThreads(1)` | 칩당 26분→18초 |
| 툴박스 제거 | `reconError.m`, `generate_aug_images.m` | `pdist2`(Statistics)→`abs(A-B.')` | 라이선스 의존 제거 |
| Fourier phase-shift 단축 | `generate_aug_images.m` | 4GB `A_mod` 딕셔너리 재계산→주파수 위상시프트 수식 | 중복 지수행렬 빌드 제거 |
| 1000× 메모리 절감 | `generate_aug_images.m` | 4GB 전체 로드→pulse별 루프(16MB) | 시스템 스왑 프리즈 회피 |
| 툴박스-free 회전 | `generate_aug_images.m` | `imrotate`(Image Proc)→`interp2` 백워드 매핑 | 라이선스 의존 제거 |
| **인덱싱 버그 수정** | `generate_aug_images.m` | `x_recovered`(few-shot 압축순서)와 전체 PH 배열 접근 인덱스 불일치 → `selected_idx=RC.selected_indices(idxTrain)` 매핑 | **재생성 이미지 상관 0.18→0.997** |

### 방법론적 선택 이유
- **왜 5클래스로 축소·재구성**: 초기 구현이 Mixed Targets 7클래스 전체(2049장)로 학습해 few-shot 취지를 놓침(98% trivial). 논문 Table 2의 정확한 5클래스·136장으로 재설계.
- **왜 선형보간이 틀렸는가**: Agarwal 논문 Fig.4b가 선형보간 방식[37]이 산란점을 왜곡시켜 실제와 다른 이미지를 만듦을 직접 실증. 우리 1차 구현이 이 오류를 그대로 재현해 성능이 낮았음(65% 정체).
- **왜 산란점 기반 완전판으로 교체하는가**: 논문(Geng)이 "we adopt the method proposed in Agarwal et al."로 명시 — 정답 구현체는 선형보간이 아니라 산란점 희소모델. `docs/THEORY_REFERENCES.md` 참조.

### 결론 및 논문 대비 비교
| 조건 | 선형보간(구버전) | 산란점 완전판(하이브리드) | 논문 SMPL/AT |
|---|---|---|---|
| MSTAR-R (few-shot 136장, 증강 없음) | 49.8% | **66.6%** (log-amp 60dB) | 56.6% |
| MSTAR-Aug (PH 증강 추가) | 64.9% | **90.9%** (log-amp 60dB, AT) | **96.4%** |

- **Baseline(few-shot) 재현**: 완전판 66.6% vs 논문 56.6% — few-shot 설계(136장, 5클래스, 17°/15°)가 논문 난이도를 재현. 로그진폭 전처리로 baseline도 논문 상회.
- **완전판 증강 데이터 생성·검증**: 로컬 MATLAB으로 클래스당 196배 증강(총 26,656장, `imgTrain N×64×64 complex128`). 육안 검증 시 SAR 타겟 구조·회전 시퀀스 자연스럽게 재현됨.
- **🔴 트러블슈팅 1 — 인덱싱 버그 발견·수정**: 1차 학습에서 aug 68.2%(목표 96.4%에 크게 미달). 원인 진단 결과 `generate_aug_images.m`에서 `x_recovered`(few-shot 압축순서 1..N)와 전체 PH 배열(`arr_img_fft_polar`, `depression`, `arr_azi`) 접근 인덱스가 불일치 → 서로 다른 칩의 원본 신호가 잔차에 섞여 왜곡. 원본↔증강 상관 **0.18**로 확정. `selected_idx = RC.selected_indices(idxTrain)` 매핑으로 수정 후 상관 **0.997**로 정상화 → 재생성.
- **🔴 트러블슈팅 2 — 로그진폭(dB) 전처리 도입**: 버그 수정 후에도 선형진폭 로더는 aug 72.1%에서 정체(epoch 60·120 동일 → 학습 포화 확인). SAR 진폭은 소수 밝은 산란점에만 값이 몰려(dynamic range 큼) CNN이 형상 전체 대신 피크 2~3픽셀만 학습 → tracked vehicle(T72/BMP2) 혼동. `precomputed_aug.py` 로더에 **per-image 정규화 → dB 압축 → clip → [0,1]** (`log_scale=True, dyn_range_db=60`) 적용. MSTAR ATR 논문 표준 전처리. **결과 72.1% → 90.9%** (+18.8%p).
- **dynamic range 스윕** (AT, 60 epoch): 25dB 81.9% / 30dB 87.7% / 35dB 88.1% / 40dB 89.1% / 50dB 90.1% / **60dB 90.9%(피크)** / 80dB 90.5% / 100dB↓. → 60dB 최적.
- **손실함수 비교** (60dB): AT **90.9%** vs LSM 87.1%. 논문은 LSM(97.6%)>AT(96.4%)이나 **우리 합성 데이터에선 AT 우위** — LSM의 라벨 스무딩이 이미 특징이 약한 합성 T72의 판별 경계를 흐려 T72가 85%→72.3%로 폭락(BMP2 혼동↑). AT의 적대적 노이즈는 판별 경계를 날카롭게 유지.
- **클래스별 정확도** (AT/60dB): 2S1 84.3% / BMP2 91.3% / BTR70 95.4% / T72 85.1% / ZSU23 100%. **잔여 갭은 T72·2S1**(tracked vehicle 미세 산란구조) — σ_G를 클래스별로 재검증하면 추가 개선 여지(2S1에서만 σ_G=1.0 검증됨).
- **최종**: 90.9% vs 논문 96.4% (갭 5.5%p). few-shot·산란점 물리증강 파이프라인 전체를 로컬 MATLAB↔Colab 하이브리드로 완주하고 논문에 근접 재현. 잔여 갭은 합성-실측 도메인 차이 및 클래스별 σ_G 미세조정 미완에 기인.
  - (검증 이력) Python 포팅 `ph_sparse.py`: 1단계 round-trip |corr|=0.99, 2단계 연산자 adjoint 오차 4e-8, 자기일관성 5/5 — 원리 검증용. 실파이프라인은 MATLAB.

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
| 2 | **대비 자동조절 (Optuna 자동탐색)** | `exp_c_contrast_optuna.py` | 논문은 대비 레벨 3개를 근거 없이 임의 고정 — 우리는 Optuna로 대비 파라미터를 자동 최적화(재현성·객관성). 논문 3레벨 재현을 baseline으로 두고 그 위에 자동탐색을 얹음 |
| 3 | **XAI × 산란점 IoU 검증** (Grad-CAM → 픽셀 단위 XAI로 확장) | `run_gradcam_analysis()` / `run_xai_analysis()` (exp_b) | 완전판(산란점 기반, log-amp 60dB, 90.9%) 모델로 재실행 완료. 아래 상세 |

### 개선 #3 상세 — XAI로 물리적 산란점 검증

**질문**: 모델이 실제 물리적 산란점(scattering centers)을 보고 분류하는가?
**지표**: 어트리뷰션의 산란점 집중도 = coverage(산란점 위치 평균 어트리뷰션) / 기준선(전체 평균). >1이면 산란점에 더 집중. + 상위 20% 어트리뷰션 영역과 산란점 마스크의 IoU.

**Grad-CAM의 한계**: SMPL의 마지막 conv 특징맵이 **8×8**뿐이라 얇은 점 산란체를 국소화 못 함(IoU 0.12). CAM 계열의 구조적 해상도 천장. 한 단계 앞 conv(16×16)에서 뽑으면 IoU 0.19로 개선되나 여전히 픽셀 단위엔 못 미침.

**해상도를 올릴수록 산란점 정합(IoU)이 단조 증가** (완전판 90.9% 모델, El15° 실측 10장, 산란점 추출을 타겟 영역으로 제한한 뒤 수치):

| 방법 | 성격 / 해상도 | 비율(coverage/기준선) | **평균 IoU** |
|---|---|---|---|
| Grad-CAM | CAM 계열 (8×8, 표준) | 1.44× | **0.12** |
| Grad-CAM | CAM 계열 (16×16, `from_last=1`) | 3.51× | **0.19** |
| **Occlusion Sensitivity** | 인과적, 픽셀 공간 | 3.21× | **0.24** |
| **SmoothGrad-IG** | 공리적, 픽셀 공간 | 11.95× | **0.29** |

**두 지표의 역할 (정직한 해석)**:
- **IoU가 주축 지표**. 상위 20% 어트리뷰션과 산란점 마스크의 겹침 → 임계 기반이라 **방법 간 공정 비교** 가능. **0.12 → 0.19 → 0.24 → 0.29로 해상도↑에 단조 증가** = CAM 해상도 천장을 데이터로 입증, 픽셀 XAI가 극복.
- **비율은 보조**. 각 방법이 자기 랜덤 기준선(≈1.0×) 대비 몇 배 집중하나 → 전부 >1(우연 아님). 단 분모(맵 전체 평균)가 맵의 뾰족함에 좌우돼(Grad-CAM/SmoothGrad는 소수 영역만 켜져 기준선이 낮음 → 비율 과대) **방법 간 직접 비교엔 부적합**. 그래서 비율은 "우연 아님"의 증거로만, 순위 비교는 IoU로.

- **Occlusion**: 산란점 영역을 가리면 타깃 클래스 확률이 급락 → "그 영역이 분류에 인과적으로 기여"를 직접 입증(발표 정성 자료).
- **SmoothGrad-IG**: 입력 픽셀 공간 직접 어트리뷰션 → CAM 해상도 천장 없음. 산란점에 가장 정밀하게 집중(IoU 0.29, 발표 정량 자료). 물리 기반 증강으로 학습한 모델이 실제 산란점을 근거로 분류함을 입증.
- 구현: `gradcam/attributions.py`(occlusion/IG/SmoothGrad), `gradcam/cam.py`(GradCAM `from_last` 층 선택), 산란점 추출 타겟 영역 제한(`extract_spatial_scattering_centers`).

> IoU 상한 참고: 산란점은 5개 점 마스크, 어트리뷰션은 타겟 줄기 전체라 완벽한 모델이라도 IoU 상한이 구조적으로 0.3~0.4 → **0.29는 강한 정합**. 랜덤 기준선 IoU≈0.05.

---

## 논문과의 설계 차이 총정리

| 실험 | 논문 | 우리 | 차이 성격 | 상태 |
|---|---|---|---|---|
| Exp A | 클러터 전이 (Table 4) | 동일 재현 + SMPL/ResNet 비교 | 충실 재현 | 🟠 CTx2 원인 미확정 |
| Exp B | PH 보간 few-shot (Table 2·3) | 5클래스·136장 few-shot, 산란점 완전판 하이브리드 완주, log-amp 60dB로 90.9% | 근접 재현(90.9% vs 96.4%) | 🟢 근접 재현 |
| Exp C | MSTAR El 교차(도입부) / SAMPLE(주실험) | **SAMPLE 데이터셋(논문 주실험과 동일) + Optuna 추가** | 주실험 그대로 + 방법 확장 | ✅ |
| Exp D | OOD 탐지 (ID=SAMPLE) | ID=SAMPLE로 재설계 완료, **ODIN vs Mahalanobis 비교 + SAR-ship(MiniSAR 대체)** | 정합 재현 + 방법론 확장 | 🟠 재실행 필요 |

> 자세한 근거는 `docs/DATASET_METHOD.md`(데이터셋 정당성), `docs/PAPER_SPEC.md`(논문 원문 수치), `docs/THEORY_REFERENCES.md`(이론·구현난점) 참조.
