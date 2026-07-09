# DATASET_METHOD.md — 실험별 데이터셋 · 구현방법 · 정당성

> 각 실험이 **어떤 데이터로, 어떻게 구현하고, 논문과 다른 데이터를 쓰면 왜 정당한지**를 정리.
> 정확한 표/수치는 `docs/PAPER_SPEC.md`, 지향점·트러블슈팅은 `CLAUDE.md` 참조.

범례: 🟢 논문과 동일 · 🟡 의도적 대체(정당) · 🔴 어긋남(수정 대상)

---

## Exp A — Clutter Transfer

### 데이터셋
- **논문**: MSTAR 5클래스(2S1,BMP2,BTR70,T72,ZSU23) El15°/17° + MSTAR clutter chips 1159개
- **우리**: 🟢 **gengzhe2015/SAR-target-recognition** repo (Original MSTAR Images + Train_CT/CTx2 PNG)
- **정당성**: gengzhe2015는 **논문 저자(Geng)가 직접 올린 공식 데이터**다. 논문 본문: *"the image samples ... used to generate the simulation results in Section 4.2 have been uploaded to Github (gengzhe2015/SAR-target-recognition)."* → 대체가 아니라 **논문의 권위 소스 그대로**. SDMS raw 없이도 정확 재현 가능.

### 구현 방법
1. `MSTARImageFolder`로 클래스별 PNG 로드 (5클래스)
2. `clutter_transfer()` — 타겟 칩을 다른 배경 클러터에 feather/Poisson 블렌딩
3. 4조건 데이터셋 구성 후 `train_model()` (SMPL, ResNet18)

### 실험 설계 (4조건 × 2모델 × 3seed)
| 조건 | train | test |
|---|---|---|
| MSTAR_OR | 원본 | 원본 |
| Train_OR+Test_CT | 원본 | 클러터 전이 |
| Train_CT+Test_CT | 클러터 전이 | 클러터 전이 |
| Train_CTx2+Test_CT | 2배 전이 | 클러터 전이 |

### 우리 개선
- **#1 SSIM**: 원본 chip vs feather-blend 결과의 SSIM 측정 → 경계 아티팩트 정량화. clutter_bg 없으면 합성 SAR 스페클 배경 폴백.

---

## Exp B — Phase-History Interpolation (few-shot)

### 데이터셋
- **논문**: MSTAR raw 5클래스, El17°(train)/El15°(test). **BMP2,BTR70,T72 = Targets 패키지 / 2S1,ZSU23 = Mixed Targets** → 두 패키지 병합 필요
- **우리(현재)**: 🔴 Mixed Targets 7클래스(2S1,BRDM_2,BTR_60,D7,T62,ZIL131,ZSU_23_4) 전체 → **클래스도 다르고 few-shot도 아님**
- **수정 방향**: 5클래스로 교체(Targets 패키지 3 + Mixed 2), baseline은 클래스당 24~32장만

### 구현 방법 (논문 재현 목표)
1. `read_mstar_complex()` — [진폭][위상] 블록 → complex 이미지 (BUG-X4 반영)
2. **azimuth 이웃 PH 보간**: 실이미지 Az=θ+5°에서 Az=θ 합성 (±1° 이웃)
   - `interpolate_phase_history()`: IFFT→PH 도메인 선형보간→FFT, Taylor(sll=35) 윈도우
3. **입력 64×64 center-crop** (128 resize 아님)
4. `train_model()` with **AT(ε=2) 또는 LSM(lblsm=0.1)** 손실, 60 epoch ADAM

### 실험 설계 (few-shot 핵심)
| 데이터셋 | 구성 | 총 학습샘플 | SMPL/AT 정확도 |
|---|---|---|---|
| MSTAR-R | 실측 희소 | 136 | 56.6% |
| MSTAR-Aug1 | +PH 보간 | 1088 | 96.4% |
| MSTAR-Aug2 | ++PH 보간 | 1904 | 96.6% |
(test 고정 1134장, El15°)

### 우리 개선
- **#3 Grad-CAM × 산란점 IoU**: 학습 모델의 CAM이 물리적 산란점에 얹히는지 coverage/IoU로 검증.

---

## Exp C — Contrast-Based Augmentation

### 데이터셋
- **논문**: 주 실험 = **SAMPLE**(10클래스), 보조 동기부여 = MSTAR El17°/30°(Figure 1)
- **우리**: 🟢 **SAMPLE**(benjaminlewis-afrl/SAMPLE_dataset_public) — 논문의 **주 실험(Table 6)과 동일**
- **정당성**: 논문의 대비 증강 핵심 결과(Table 6, 94.5%)가 SAMPLE 기반. MSTAR El17/30(Figure 1)은 "밝기 편향" 문제를 보여주는 도입부일 뿐. 우리는 SAMPLE 주 실험을 재현하고 MSTAR El은 `run_el_ablation()` 보조로 유지 → 오히려 논문 핵심에 충실.
- **SAMPLE 10클래스**: 2s1,bmp2,btr70,m1,m2,m35,m548,m60,t72,zsu23 (806 train / 539 test)

### 구현 방법
1. `SampleDataset`: `png_images/decibel/synth/<cls>` (학습), `.../real/<cls>` (테스트)
2. `ContrastBalance` (CLAHE) 증강 — 여러 대비 레벨
3. K=0(100% synthetic)로 학습 → measured로 테스트

### 실험 설계
| 단계 | 내용 | 목표 |
|---|---|---|
| baseline | synth→measured, 증강 없음 | RN18 91.9% |
| +대비증강 | 3레벨 대비 (또는 Optuna 최적) | RN18 **94.5%** (K=0) |

### 우리 개선(부가)
- **Optuna**: clip_limit/tile_grid/global_norm 자동 탐색 (논문은 3레벨 고정) → 재현성·객관성.

---

## Exp D — OOD Detection

### 데이터셋
- **논문**:
  - ID = **SAMPLE 10클래스**(#0~9), K=0.1
  - OE 학습 = **SAR-ship 2048 + MiniSAR 443**
  - OOD 테스트 = Holdout(J개 클래스 제외) + MSTAR-O + MSTAR-P (5클래스: BRDM2,BTR60,D7,T62,ZIL131)
- **우리(현재, 2026-07 재검토 반영)**: 🟢 ID=SAMPLE(수정 완료) / 🟢 Holdout 클래스(HLD1/2/3) 정정 완료 / 🔴 K=0.1 미적용(K=0으로 학습 중) / 🔴 MSTAR-O/P 미구현, 대신 SAR-ship을 보조 far-OOD로 사용 중
- **의도적 대체(정당)**: **MiniSAR는 논문 저자 자체 개발 비공개 데이터** → 공개된 SAR-ship로 OE 대체. OOD/OE에 cross-domain SAR을 쓰는 것은 문헌 표준(원 논문도 SAR-ship 병용)이라 정당.
- **수정 필요(어긋남)**: SAR-ship은 **OE 학습용**이지 논문 Figure 9의 OOD 테스트셋이 아님(Figure 9는 Holdout/MSTAR-O/MSTAR-P 3종만 비교) — 지금은 SAR-ship을 "우리가 추가한 보조 far-OOD"로 명확히 구분해 서술. ID 학습에 K=0.1(실측 10% 혼합)과 대비 증강(Section 3)도 아직 미적용.

### 구현 방법 (논문 재현 목표)
1. ID = SAMPLE 10클래스로 분류기 학습, **K=0.1**(synth 90%+real 10% 혼합) + 대비 증강 적용 (`get_features()` 특징 추출) — K=0.1/대비 증강은 아직 미구현
2. OE = SAR-ship을 adversarial outlier exposure 학습 재료로
3. OOD 테스트: Holdout(SAMPLE 일부 클래스 제외, HLD1/2/3 정정 완료) + MSTAR-O(구현 필요, 데이터는 이미 보유) + MSTAR-P(우선순위 낮음, 정확한 소스 특정 불가)
4. 탐지기: **ODIN**(T=1000) vs **Mahalanobis**(AdvOE), AUROC·TNR@95TPR — AdvOE의 adversarial 학습 목적함수(Eq.4) 자체는 미구현, 현재는 일반 학습 후 사후 스코어링만

### 실험 설계
| OOD 종류 | 성격 | 기대 |
|---|---|---|
| Holdout (J=1,2,3 = {M1} / {M35,M548} / {M1,M35,M548}) | 같은 SAMPLE 내 미지 클래스 (near-OOD) | 어려움 |
| MSTAR-O/P | cross-dataset (far-OOD), 논문 Figure 9의 실제 비교 대상 | 쉬움, Maha 우세 |
| SAR-ship (우리 추가, 논문엔 없음) | cross-domain 보조 검증 | 참고용 |

### 우리 방법론 확장 (Exp D)
- **ODIN vs Mahalanobis 비교** — 논문 정성 결론(Maha 우세)을 정량 재현. (팀 개선 3종은 A=SSIM / C=대비 자동조절 / B=XAI에 배치, Exp D 자체엔 별도 개선 없음)

---

## 데이터셋 총괄표

| Exp | 논문 데이터 | 우리 데이터 | 상태 | 대체 근거 |
|---|---|---|---|---|
| A | MSTAR 5cls + clutter | gengzhe2015 PNG | 🟢 | 논문 저자 공식 업로드본 |
| B | MSTAR raw 5cls (Targets+Mixed) | Mixed 7cls (현재) | 🔴 | 수정: 5cls few-shot |
| C | SAMPLE 10cls | SAMPLE 10cls | 🟢 | 논문 주 실험과 동일 |
| D | SAMPLE(ID, K=0.1)+SAR-ship/MiniSAR(OE)+Holdout/MSTAR-O/P(OOD) | SAMPLE(ID, K=0)+SAR-ship(보조 far-OOD) | 🟡🔴 | MiniSAR 비공개→SAR-ship (정당); ID/Holdout은 수정 완료, K=0.1·MSTAR-O·대비증강 미구현 |
