# THEORY_REFERENCES.md — 이론적 배경 · 참조 논문/레포 · 구현 난점 (보고서용)

> 실험별 이론적 배경, 참조한 논문·레포와 "무엇을 참조하는지", 구현 시 어려웠던 점, 검증 기준을 정리.
> 보고서 작성 + precomputed 검증의 기준 문서. (수치·표는 `docs/PAPER_SPEC.md`, 데이터·정당성은 `docs/DATASET_METHOD.md`)
> 핵심 방향/이해가 바뀌면 갱신.

---

## 0. 참조 논문·레포 목록 (무엇을 참조하는가)

| 자료 | 정체 | 우리가 참조하는 것 | 위치 |
|---|---|---|---|
| **Geng et al. 2023** (Sensors 23,941) | **우리 재현 대상 논문** | 4개 실험 설계·수치(Table 2-6, Fig 1,2,9). 위상보간은 "Agarwal 채택"이라 명시 | `docs/2b626a7d-sensors2300941.pdf` |
| **Agarwal et al. 2020** (arxiv 2012.09284) | 위상보간 **방법 원전** | 산란점 희소모델 수식(Eq1-8), 파이프라인(Fig3), 합성품질 검증(Fig4), 정확도(Table I-V) | `docs/9eac9493-2012.09284v2.pdf` |
| **SENSE-Lab-OSU/mstar_data_aug** | Agarwal 방법 **MATLAB 구현** | 정확한 연산 순서·상수·좌표규약 (preprocess/sparse_recovery/generate) | github (참고용, precomputed 데이터는 404 삭제) |
| **SPGL1** (van den Berg & Friedlander) | 그룹 희소 복원 **솔버** | 우리는 이걸 **FISTA+그룹prox로 대체** (동등한 group-lasso) | 원 MATLAB에 포함 |
| **mtimesx** (J. Tursa) | MATLAB 고속 배치 행렬곱 MEX | numpy matmul로 대체 (언어 차이일 뿐) | 원 MATLAB에 포함 |
| **gengzhe2015/SAR-target-recognition** | Geng의 **클러터 전이 이미지** (코드 아님) | Exp A 데이터 (논문 저자 공식 업로드본) | github |
| **benjaminlewis-afrl/SAMPLE_dataset_public** | 합성·실측 쌍 SAR | Exp C/D 데이터 | github |

---

## Exp A — Clutter Transfer

### 이론적 배경
SAR 이미지 = 타겟 + 클러터(배경). CNN이 배경 밝기/질감에 과적합하면(예: "밝으면 BRDM2") 배경 바뀔 때 붕괴. 타겟은 그대로 두고 **배경만 다른 클러터로 교체(SAR-Bake 픽셀 주석으로 타겟/그림자/클러터 분리)** → 모델이 타겟에 집중하도록 강제.

### 참조 & 무엇을 참조
- Geng 논문 Section 2.2, 4.2, Table 4 — 4조건 정확도, 클러터 6종(C1~C6) 분류
- gengzhe2015 레포 — 이미 가공된 전이 이미지 (직접 생성 불필요)

### 구현 난점
- 실제 클러터 배경 데이터(clutter_bg) 부재 → SSIM 분석은 합성 SAR 스페클로 폴백
- **CTx2 조건이 붕괴(39%)** — 논문은 회복(96%). 원인 미확정(T5, Colab 진단 로깅 대기)

---

## Exp B — Phase-History Interpolation ⭐ (가장 어려움)

### 이론적 배경 (Agarwal Eq1-8)
1. **산란점 스파시티**: SAR 타겟 = 소수의 지배적 산란점(금속 모서리 등)에 에너지 집중. 광학과 달리 마이크로파는 정반사(specular) → 각도 의존성 큼.
2. **PH(위상이력) 모델**: 이미지를 K-space(주파수)로 변환하면, 산란점들의 위치(x_k,y_k)와 각도의존 계수 h_k(θ,φ)로 신호를 표현 (Eq2,3).
3. **한계 지속성(limited persistence)**: 계수 h_k는 방위각에 대해 **매끄럽게 변함** → 3° 서브개구 내에서 가우시안 기저 D=12개로 근사 (Eq4,5).
4. **그룹 희소 복원(Eq7)**: `min_C(Σλ‖c_k‖₂ + ‖S−Ŝ‖_F)` — 공간 격자점당 계수를 한 그룹으로 묶어 L2,1 희소화 → "소수 격자점만 활성"(산란점 위치 추정).
5. **재합성(Eq8)**: 복원된 모델로 **미관측 방위각 θ의 PH를 외삽** → IFFT → 새 각도 이미지. subpixel shift로 scintillation 효과까지.

### 참조 & 무엇을 참조
- **Agarwal Eq6-8** — forward 모델·희소복원·재합성 수식
- **Agarwal Fig3** — 파이프라인 흐름도 (박스/화살표. PH 이미지의 "모양"은 검증기준 아님)
- **Agarwal Fig4** ⭐ — **합성 품질 검증 기준**. T62를 θc=57°로 합성:
  - (a) 순진한 극좌표 회전 → 나쁨
  - (b) **선형보간 [37]** → 나쁨 (= **우리 옛 코드가 했던 것**, 그래서 65% 정체)
  - (c) **제안 비선형 외삽** → 지배적 산란점 보존 (우리 목표)
- **Agarwal Table IV/V** — R=2⁻⁴ confusion matrix: 증강 전 18.6% → 증강 후 7.7% 에러
- **MATLAB 코드** — 정확한 상수(L=30m, 격자100, f_c=9.6GHz, BW=521MHz, Taylor(100,4,−35), D=12, 외삽±6° η=3)

### 구현 난점 (실제 겪은 것)
1. **선형보간 오류**: 초기 코드가 "두 이미지 평균"으로 구현 → 산란점 안 씀 → few-shot 65% 정체. Agarwal Fig4b가 이게 틀렸음을 직접 증명.
2. **복소 좌표·FFT 규약**: fftshift/ifftshift 순서, Cartesian↔Polar 보간, de-Taylor 순서 하나만 틀려도 노이즈. round-trip corr로 검증(현재 합성 0.966).
3. **그룹 희소 복원**: SPGL1(MATLAB MEX) 대신 **FISTA+그룹 soft-threshold**로 대체. group-lasso 표준 근접경사법 → 동등, 순정 Python.
4. **precomputed 소실**: 저자 OSU Box 404 → 우리가 직접 생성해야 함(수 시간→few-shot 136장이라 ~1시간).
5. **파일 포맷**: MSTAR = [진폭][위상] 블록(교차 아님, BUG-X4). 5클래스가 Targets+Mixed 두 패키지에 분산, 시리얼 서브폴더.

### 검증 기준 (precomputed가 적절한지)
- **1차**: round-trip |corr| ≥ 0.9 (좌표/FFT 규약 정확)
- **2차**: 합성 이미지가 Agarwal Fig4c처럼 원본의 지배적 산란점(밝은 점) 위치를 보존 (실vs합성 시각 비교)
- **최종**: few-shot 136 + 증강 → SMPL/AT **56.6%→96.4%** (Geng Table 3)

---

## Exp C — Contrast-Based Augmentation

### 이론적 배경
합성(synthetic) SAR은 실측(measured)보다 배경 클러터가 약함(대비 다름). synth로 학습→real 테스트 시 "밝기 편향" 학습으로 붕괴. **여러 대비 레벨(CLAHE류)로 증강** → 모델이 절대 밝기 무시하고 타겟 구조에 집중.

### 참조 & 무엇을 참조
- Geng Section 3, 4.3, Table 6 — SAMPLE 10클래스, K=0(100% synth) → RN18 91.9%→94.5%
- Geng Fig1 — MSTAR El17→30 대비 붕괴 데모 (97.2/65.3/88.5)

### 구현 난점
- SAMPLE 경로/클래스 규약(real/synth, 소문자 10클래스)
- Optuna는 우리 추가(논문은 대비 3레벨 고정)

---

## Exp D — OOD Detection

### 이론적 배경
open-world엔 미지 타겟 존재 → "모르는 것"으로 걸러야. **ODIN**(softmax + temperature + 입력섭동)은 신뢰도 기반, **Mahalanobis**(특징공간 클래스별 거리)는 거리 기반. **OE(Outlier Exposure)**: 다른 도메인 SAR(SAR-ship)을 "이건 OOD다"로 미리 학습시켜 탐지력↑.

### 참조 & 무엇을 참조
- Geng Section 4.4, Fig9-11 — ID=SAMPLE, OOD=Holdout+MSTAR-O/P, Maha 대체로 최고
- inkawhich/ood-sar-atr [17] — Geng이 네트워크 구현 참조로 지목

### 구현 난점
- MiniSAR 비공개 → SAR-ship 대체
- ID를 SAMPLE로 (MSTAR 아님), SAR-ship은 OE 역할

---

## 용어 정리 (보고서용)

- **위상이력(Phase History, PH)**: SAR 원시 주파수(K-space) 도메인. 이미지의 2D FFT ≈ PH. 산란점 모델링은 여기서 함.
- **산란점(Scattering Center)**: 타겟에서 강하게 반사하는 소수 지점(모서리·코너). 에너지가 여기 집중(스파시티).
- **그룹 희소/Group Lasso (L2,1)**: 변수를 그룹으로 묶어, 그룹 단위로 0이 되게 하는 정규화. 여기선 "공간 격자점당 계수 그룹" → 소수 격자점만 활성.
- **SPGL1**: basis pursuit denoising(희소 복원) 솔버. 우리는 FISTA로 대체.
- **FISTA**: 가속 근접경사법. group-lasso를 그룹 soft-threshold prox로 풂. SPGL1 대체.
- **Taylor 윈도우**: 사이드로브 억제용 창함수. SAR 디스프레딩에 적용됨 → PH 복원 시 역적용(de-window) 필요. scipy `taylor(sll=+35)` = MATLAB `taylorwin(-35)`.
- **mtimesx**: MATLAB 배치 행렬곱 MEX 라이브러리. numpy matmul로 대체.
- **subpixel shift**: 반픽셀 이동으로 scintillation(반짝임) 효과 재현. 일반 보간으론 불가(복소 커널 필요).
