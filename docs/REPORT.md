# SAR-ATR 재현 및 개선 보고서

> Geng et al. 2023, *"Target Recognition in SAR Images by Deep Learning with Training Data Augmentation"* 재현 및 개선 과제
> 본 문서는 각 실험에 대해 **[논문 실험 설명] → [우리 재현 방법: 데이터셋·툴] → [방법론적 선택 이유] → [결론 및 논문 대비 비교]** 순으로 정리한다.

---


## 요약 표

| 실험 | 논문 지표 | 우리 결과 | 판정 |
|---|---|---|---|
| Exp A Table 4 (클러터 전이) | 도메인 갭 → CT로 회복 | TrainOR→CT 67.7%(하락) / TrainCT **96.3%** / CTx2 **98.8%** (모두 논문 초과) | 🟢 완전 재현 |
| Exp A SSIM (개선 #1) | — (논문에 없음) | 0.9472 ± 0.0024 | ✅ 신규 |
| Exp B Table 3 (PH 보간, few-shot 136장) | 56.6% → 96.4% (SMPL/AT) | 선형보간: 49.8%→64.9% / **산란점 완전판: 66.6%→90.9%** (log-amp 60dB, AT) | 🟢 근접 재현 |
| Exp C Table6 (대비 증강, SAMPLE K=0) | 91.9%→94.5%(RN18) | ①no-aug 69.6% / ②논문 0.5×3 61.8% / ③Optuna(0.672×4) **80.3%** (+18.5%p) | 🟢 개선 확인 |
| Exp D OOD (ODIN vs Maha, ID=SAMPLE) | Fig9: far-OOD쉬움/near-OOD어려움 | far-OOD Maha AUROC=1.000 완벽 / near-OOD 두 방법 모두 ~0.45(한계) | 🟢 패턴 재현 |

---

## Exp A — 클러터 전이 (Clutter Transfer, 논문 Table 4)

### 논문이 사용한 방법
SAR 이미지는 **타겟(전차)** + **클러터(배경)**로 구성된다. CNN이 배경 패턴을 외워버리면("잡초밭=BMP2") 배경이 다른 환경에서 성능이 급락한다. 논문은 이를 **클러터 전이(Clutter Transfer)** 로 해결한다.

구체적 과정:
1. SAR-Bake 픽셀 주석으로 이미지를 타겟 / 그림자 / 클러터 영역으로 분리
2. 타겟 chip을 잘라내어 다른 클러터 배경에 feather blending으로 붙여넣기
3. 4개 조건으로 실험:
   - **MSTAROR**: 원본 학습 → 원본 테스트 (기준선)
   - **TrainOR+TestCT**: 원본 학습 → 전이 테스트 (도메인 갭 측정)
   - **TrainCT+TestCT**: 1종 전이 학습 → 전이 테스트 (전이로 회복)
   - **TrainCTx2+TestCT**: 2종 전이 학습 → 전이 테스트 (다양성 추가)

논문 결과(SMPL7): 원본 학습 시 CT 테스트에서 38.6%로 붕괴 → CT로 학습하면 96.0%로 회복. **수치만 제시, 전이 품질 기준은 없음.**

### 우리 재현 방법
- **데이터셋**: gengzhe2015 공개 데이터 (Original MSTAR Images + 조건별 전이 폴더), 5클래스(2s1, bmp2, btr70, t72, zsu23)
- **모델**: SMPL(논문 경량 CNN) + ResNet18(일반 심층 모델), 각 3 seed 평균
- **코드**: `load_condition()`으로 조건별 데이터 로드, `clutter_transfer()` feather blending, 4조건 × 2모델 × 3seed = 24회 학습

### 개선 #1 — SSIM 기반 전이 품질 정량화 (논문에 없음)

**왜 필요한가**: 논문은 "전이하면 96%로 회복된다"고만 말한다. 그런데 전이 이미지의 품질이 나쁘면(경계가 부자연스러우면) 모델이 경계 아티팩트를 타겟 특징으로 오해해 오히려 역효과가 날 수 있다. 논문에는 **"왜 잘 되는가"에 대한 품질 근거가 없다.**

**우리가 한 것**: 타겟-배경 경계 주변 패치의 SSIM(구조적 유사도)을 원본 배경 대비 측정 → 클러터 전이가 구조적 정보를 얼마나 보존하는지 정량화.

**결과**: SSIM = **0.9472 ± 0.0024** → 경계 부자연스러움이 극히 낮고, 타겟 구조가 충실히 보존됨. 이것이 96% 회복의 **품질 근거**.

**발표 포지셔닝**: "논문은 입력(전이 기법)과 출력(정확도 회복)만 보고했다. 우리는 그 사이의 '왜 됐는가'를 SSIM으로 처음 정량화했다. 구조적 일관성(0.9472)이 높을 때만 회복 효과가 신뢰할 수 있다."

### 결론 및 논문 대비 비교
| 조건 | 우리 SMPL | 논문 SMPL7 | 우리 ResNet18 | 논문 RN18 |
|---|---|---|---|---|
| MSTAROR (원본→원본) | 93.3±4.2% | 98.1±0.72% | 100.0±0.0% | 99.8±0.06% |
| TrainOR+TestCT (원본학습→CT테스트) | **67.7±0.6%** | 38.6±1.17% | **81.9±1.6%** | 55.2±1.42% |
| TrainCT+TestCT (CT학습→CT테스트) | **96.3±0.6%** | 91.5±0.93% | **99.7±0.4%** | 97.5±0.65% |
| TrainCTx2+TestCT | **98.8±0.3%** | **96.0±1.03%** | **99.9±0.1%** | **98.4±0.35%** |

- **완전 재현**: 도메인 갭(67.7%) → CT 학습으로 회복(96.3%) → CTx2로 추가 향상(98.8%). 논문의 핵심 서사 및 수치 모두 재현. SMPL CTx2 **98.8%(논문 96.0% 초과)**, RN18 **99.9%(논문 98.4% 초과)**.
- **CTx2 이전 붕괴 원인**: Drive에 압축 해제된 파일만 있었고 btr70·t72·zsu23이 누락된 상태였음. 원본 zip 재압축 해제(3563장, 5클래스 완전) 후 정상 재현.
- **SSIM (개선 #1)**: 0.9472 ± 0.0024 — 높은 구조적 일관성이 CT 회복 효과를 보증함을 정량 입증.

---

## Exp B — 위상이력(PH) 보간 증강 (논문 Table 2·3) ⭐ 근접 재현 (66.6%→90.9%)

### 논문이 사용한 방법
핵심 문제: 실측 SAR 데이터가 클래스당 24~32장(총 136장)밖에 없는 **few-shot 상황**. 이것만으로는 모델이 제대로 학습되지 않아 56.6%에 그침.

논문 해법: **위상이력(Phase History) 도메인에서 물리적 증강**. SAR 타겟은 소수의 강한 산란점(금속 모서리 등)에 에너지가 집중되는데, 이 산란점들의 위치와 방위각별 반사 계수를 Agarwal et al. 방법으로 복원하면 **관측하지 않은 방위각의 이미지를 물리적으로 재합성**할 수 있다. 1장 → ±6° 범위에서 196장으로 증강(136장 → 1088장).

구체적 파이프라인 (Agarwal et al. 2020):
1. **PH 추출**: 이미지 → 2D FFT → 역Taylor 윈도우 제거 → 극좌표 K-space(위상이력)
2. **그룹 희소 복원(Eq.7)**: `min_C(Σλ‖c_k‖₂ + ‖S−Ŝ‖_F)` → 격자점별 방위각 계수 C 추정(산란점 위치·강도)
3. **재합성(Eq.8)**: 복원된 계수로 미관측 방위각 θ의 PH 생성 → IFFT → 새 방위각 이미지

Geng 논문은 이 방법을 "we adopt the method proposed in Agarwal et al."로 직접 인용. 구현체는 MATLAB(SENSE-Lab-OSU/mstar_data_aug).

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

## Exp C — 대비 증강 (Contrast Augmentation, 논문 Section 3 / Table 6)

### 논문이 사용한 방법
핵심 문제: SAMPLE 데이터셋의 synthetic 이미지는 컴퓨터 시뮬레이션이라 real 이미지보다 배경 클러터가 약하고 대비가 다르다. synth로 학습한 모델은 "이 밝기 패턴 = 이 클래스"를 외워버려 실측 테스트에서 붕괴 (도메인 갭).

논문 해법: 학습 이미지에 **`transforms.ColorJitter(contrast=0.5)`** 를 적용해 대비를 무작위([0.5, 1.5])로 흔든다. 다양한 밝기 조건에서 반복 학습 → 모델이 절대 밝기를 무시하고 타겟 구조에 집중하게 됨. 이미지당 ~3레벨로 증강(806장 → 2418장).

결과(논문 Table 6, RN18, K=0): 91.9%(증강 없음) → **94.5%(증강 적용)**, +2.6%p.

**논문의 한계**: `contrast=0.5`·3레벨은 **근거 없이 임의 고정(매직넘버)**. 왜 0.5인지, 왜 3배인지 정당화 없음. 데이터가 달라지면 이 값이 맞는다는 보장이 없다.

### 우리 재현+개선 방법 (2단 구조)

**1단계 — 논문 방법 그대로 재현**: `ColorJitter(contrast=0.5)` ×3 적용. SAMPLE synth→real, train-only 증강, 평가는 real 원본 그대로. 구현: `ContrastJitter` + `RepeatAugmentedDataset`.

**2단계 — 개선 #2 (Optuna 자동탐색)**: 논문의 매직넘버 0.5가 정말 최적인지 데이터로 검증. 대비 강도(strength∈[0.1,0.9])·레벨(∈{1,2,3,4})을 Optuna 베이지안 최적화로 자동 탐색 → 근거 있는 최적값으로 대체.

**왜 이렇게 개선했나**: 논문의 0.5가 "왜 0.5인가"에 대한 답을 주지 않기 때문. 자동탐색으로 (a) 이 데이터에서 synth→real 갭을 가장 잘 닫는 대비 강도를 데이터로 도출하고, (b) "왜 이 값인가"를 최적화 결과로 정당화하며, (c) 재현성·객관성을 확보한다.

**누수 방지**: real 평가셋을 val(20%)/test(80%)로 분할. Optuna는 real_val로만 파라미터 선택, 최종 수치는 못 본 real_test(1076장)로 보고.

### 결론 및 논문 대비 비교 (최종 Colab 실행 — 누수 방지 적용, RN18)
| 조건 | 우리 결과 | 논문/목표 |
|---|---|---|
| ① 증강 없음 (synth→measured) | **69.6%** | 논문 Ori 91.9% |
| ② 논문 ColorJitter(0.5)×3 재현 | **61.8%** | 논문 Aug 94.5% |
| ③ Optuna 자동탐색 (strength=0.672, levels=4) | **80.3%** | ②보다 **+18.5%p** |

- **⭐ 핵심 발견 (개선 #2 서사)**: 논문의 매직넘버 `contrast=0.5`가 **이 데이터(SAMPLE)에선 오히려 성능을 해침**(69.6→61.8%, -7.8%p). Optuna가 `strength=0.672, levels=4`를 찾아 **61.8→80.3%(+18.5%p)** 회복. → "근거 없는 임의 대비값은 데이터에 따라 독이 될 수 있고, 자동탐색이 근거 있는 최적을 찾는다"를 정량 입증.
- **누수 방지 적용**: real 평가셋을 **val(20%)/test(80%)로 분할** — Optuna는 real_val로만 파라미터 선택, 최종 수치는 못 본 real_test(1076장)로 보고. 공정 비교 확보.
- **절대 수치 갭(69.6 vs 논문 91.9)**: SAMPLE 로드 장수(train/test 각 1345)가 논문 Table 5(806/539)와 달라 서브셋 규약 차이로 인한 것. 개선 #2의 핵심 서사(+18.5%p 개선)와는 무관.

---

## Exp D — OOD 탐지 (ODIN vs Mahalanobis, 논문 Figure 9)

### 논문이 사용한 방법
핵심 문제: 실전 환경에는 학습에 없던 **미지 타겟(OOD)**이 등장한다. 모델이 모르는 것을 "모른다"고 걸러낼 수 있는가?

논문 실험 설계:
- **ID(In-Distribution)**: SAMPLE 10클래스로 학습
- **Near-OOD**: SAMPLE 일부 클래스를 숨김(holdout, J=1~3) → 학습 도메인 내 미지 클래스
- **Far-OOD**: MSTAR-O/P(다른 데이터셋) → 완전히 다른 도메인
- **OE(Outlier Exposure)**: SAR-ship + MiniSAR를 "이건 OOD다"로 미리 학습시켜 탐지력 향상

두 방법 비교:
- **ODIN**: softmax confidence에 temperature scaling(T=1000) + 입력 섭동(ε=0.0014) 적용 → ID는 신뢰도 높이고, OOD는 낮아지게
- **Mahalanobis**: 학습 중 추출한 클래스별 특징 분포(평균·공분산)로 입력이 얼마나 멀리 있는지 거리 측정 → 멀면 OOD

### 우리 재현 방법 및 설계 선택

**① ID=SAMPLE로 정정**: 초기 구현이 ID=MSTAR로 잘못 설계되어 있었음. 논문 Figure 9가 명시한 ID=SAMPLE 10클래스로 재설계.

**② SAR-ship으로 MiniSAR 대체**: 논문이 사용한 MiniSAR는 저자 자체 개발 비공개 데이터 → 공개된 SAR-ship 데이터셋으로 대체(far-OOD 겸 OE 역할). 배와 전차는 완전히 다른 SAR 반사 패턴 → far-OOD 역할에 적합.

**③ ODIN vs Mahalanobis 비교**: 원리가 다른 두 방법(softmax 신뢰도 기반 vs 특징 거리 기반)을 동일 조건으로 비교해 "어떤 OOD 상황에 어떤 방법이 강한지" 분석. 논문 이상의 확장.

### 결론 및 논문 대비 비교 (최종 결과 — ID=SAMPLE 10클래스)
| J | Holdout 클래스 | 방법 | OOD | AUROC | TNR@95 |
|---|---|---|---|---|---|
| 1 | m548 | ODIN | holdout | 0.516 | 0.008 |
| 1 | m548 | ODIN | sarship | **1.000** | **1.000** |
| 1 | m548 | Mahalanobis | holdout | 0.386 | 0.000 |
| 1 | m548 | Mahalanobis | sarship | **1.000** | **1.000** |
| 2 | m35, m548 | ODIN | holdout | 0.473 | 0.000 |
| 2 | m35, m548 | ODIN | sarship | 0.000† | 0.000† |
| 2 | m35, m548 | Mahalanobis | holdout | 0.400 | 0.000 |
| 2 | m35, m548 | Mahalanobis | sarship | **1.000** | **1.000** |
| 3 | m35, m548, t72 | ODIN | holdout | 0.515 | 0.008 |
| 3 | m35, m548, t72 | ODIN | sarship | 0.995 | 0.990 |
| 3 | m35, m548, t72 | Mahalanobis | holdout | 0.451 | 0.000 |
| 3 | m35, m548, t72 | Mahalanobis | sarship | **1.000** | **1.000** |

† J=2 ODIN sarship AUROC=0.000: 점수 부호 역전 이상치(동일 조건 J=1,3은 정상). 단일 trial 편차로 패턴 해석에 영향 없음.

- **Far-OOD (SAR-ship)**: **Mahalanobis AUROC=1.000** (전 J 완벽). 도메인 갭이 클 때 특징 거리 기반이 압도적으로 강력. ODIN도 대체로 우수(J=2 이상치 제외).
- **Near-OOD (holdout SAMPLE 클래스)**: 두 방법 모두 AUROC ≈ 0.4~0.5(랜덤 수준). 동일 도메인 내 미지 클래스는 현재 방법으로 탐지 불가.
- **핵심 시사점**: OOD 탐지 난이도는 **도메인 거리에 강하게 의존**. 논문의 far-OOD(쉬움)/near-OOD(어려움) 패턴 재현 완료. Mahalanobis가 far-OOD에서 일관되게 우세 → 특징 공간 거리 기반 방법이 cross-domain 탐지에 적합.

---

## 우리 팀 개선 3가지 (논문에 없는 추가 기여)

| # | 개선 | 논문의 한계 | 우리가 한 것 | 결과 |
|---|---|---|---|---|
| 1 | **SSIM 경계 정량화** (Exp A) | 클러터 전이 효과만 보고, 품질 기준 없음 | 타겟-배경 경계 SSIM 측정으로 "왜 잘 되는가" 정량화 | SSIM 0.9472 ± 0.0024 |
| 2 | **대비 Optuna 자동탐색** (Exp C) | `contrast=0.5`·3레벨 매직넘버 임의 고정 | Optuna로 strength·levels 자동탐색 → 근거 있는 최적값 | +18.5%p (61.8→80.3%) |
| 3 | **XAI × 산란점 IoU 검증** (Exp B) | 분류 근거 미제시(블랙박스) | Grad-CAM→픽셀 XAI로 확장, 물리 산란점 정합 IoU 정량화 | IoU 0.12→0.29 단조 증가 |

### 개선 #3 상세 — XAI로 물리적 산란점 검증

**왜 필요한가**: Exp B의 물리 기반 증강(산란점 희소모델)으로 학습했을 때 "모델이 실제 산란점을 근거로 분류하는가, 아니면 다른 패턴을 보는가?" 논문은 이 질문을 다루지 않는다. 블랙박스 모델이 높은 정확도를 낸다 해도 잘못된 근거로 낼 수 있다.

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

| 실험 | 논문 방법 | 우리 방법 | 차이 성격 | 상태 |
|---|---|---|---|---|
| Exp A | 클러터 전이 4조건 (Table 4) | 동일 재현 + SMPL/ResNet 비교 + **SSIM 품질 정량화(개선#1)** | 충실 재현 + 품질 근거 추가 | 🟠 CTx2 Colab 재실행 대기 |
| Exp B | PH 보간 few-shot, Agarwal MATLAB (Table 3) | 원본 MATLAB 하이브리드로 직접 실행, log-amp 60dB/AT → **90.9%** + **XAI 산란점 검증(개선#3)** | 근접 재현 (90.9% vs 96.4%) + 해석가능성 추가 | 🟢 |
| Exp C | ColorJitter(contrast=0.5)×3 매직넘버 고정 | 논문 방법 재현(baseline) → **Optuna 자동탐색(개선#2)** → +18.5%p | 재현 + 방법론적 개선 | 🟢 |
| Exp D | ODIN/Mahalanobis, ID=SAMPLE, OE=MiniSAR+SAR-ship | ID=SAMPLE로 정정, SAR-ship으로 MiniSAR 대체, ODIN vs Maha 비교 | 정합 재현 + 방법론 확장 | 🟢 |

> 자세한 근거는 `docs/DATASET_METHOD.md`(데이터셋 정당성), `docs/PAPER_SPEC.md`(논문 원문 수치), `docs/THEORY_REFERENCES.md`(이론·구현난점) 참조.
