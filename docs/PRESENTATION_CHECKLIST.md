# PRESENTATION_CHECKLIST.md — 발표 슬라이드별 필요 항목

> 발표 논리: **"① 논문처럼 돌려보니 논문과 같은 결과가 나왔다(재현 확인) → ② 우리 개선을 얹으니 여기서 더 좋아졌다(기여)"**
> 각 실험마다 이 표의 항목이 다 채워져야 슬라이드 하나가 완성된다. 실측 수치는 채워지는 대로 `docs/REPORT.md`에 반영.

---

## 공통 슬라이드 구조 (실험당 1~2장)

1. **논문 실험 설명** (한 줄 요약 + 원문 표/그림 번호)
2. **우리 셋업** (데이터셋, 모델, 조건)
3. **재현 결과** — 우리 수치 vs 논문 수치 (막대그래프 또는 표, 나란히)
4. **개선 결과** — 우리 개선 적용 전/후 (있는 경우)
5. **한 줄 결론** (재현 성공/부분/시사점)

---

## Exp A — 클러터 전이 (Table 4)

| 필요 항목 | 값 |
|---|---|
| 모델 | **SMPL**, **ResNet18** (둘 다 — 논문처럼 headline+비교군) |
| 데이터셋 | gengzhe2015 (Original MSTAR Images / Train_OR_Test_CT / Train_CT_Test_CT / Train_CTx2_Test_CT), 5클래스 |
| 조건 (4개) | MSTAR_OR / Train_OR+Test_CT / Train_CT+Test_CT / Train_CTx2+Test_CT |
| 재현 비교 수치 | 우리 4조건×2모델 정확도(±std, 3seed) **vs** 논문 Table 4 (SMPL7: 98.1/38.6/91.5/96.0, RN18: 99.8/55.2/97.5/98.4) |
| 개선 결과 | **SSIM(개선#1)**: clutter transfer 전후 경계 SSIM 값 — "경계 아티팩트가 작다"는 정량적 근거 |
| ⚠️ 상태 | **BUG-3 수정 완료**: `load_condition()`이 CT 폴더를 불필요하게 80/20 재분할 → 제거. Colab 재실행으로 CTx2 수치 갱신 필요 (96% 근접 예상) |
| 발표 문장 예시 | "논문처럼 클러터 전이 시 도메인 갭(하락)과 CTx2를 통한 회복을 확인했다 / 확인 중이다" |

---

## Exp B — 위상이력(PH) 보간 (Table 3) ⭐ 핵심

| 필요 항목 | 값 |
|---|---|
| 모델 | **SMPL** (논문 headline). 참고: 논문은 ResNet18/AConvNet도 비교하나 우리는 SMPL 우선 |
| 데이터셋 | MSTAR raw (Targets 패키지: BMP2/BTR70/T72 + Mixed: 2S1/ZSU23), 5클래스, **train 17°/test 15°** |
| 조건 | **few-shot baseline(136장, MSTAR-R)** vs **PH 증강(1088장, MSTAR-Aug1)** |
| 재현 비교 수치 | 우리 SMPL/AT 정확도 **vs** 논문 56.6%→96.4% |
| 개선 결과 | **XAI(개선#3)**: 어트리뷰션이 산란점 위치에 집중하는가(coverage/IoU) — "물리적으로 타당한 특징을 학습했다"는 근거. **IoU 단조증가 서사** (주축=IoU, 방법 간 공정비교) — Grad-CAM 8×8(0.12) → 16×16(0.19) → Occlusion(0.24) → **SmoothGrad-IG(0.29)**. 비율은 보조("우연 아님" 증거, 전부 >1이나 맵 뾰족함에 좌우돼 순위비교엔 부적합). 정성=Occlusion(가리면 붕괴), 정량=SmoothGrad-IG |
| ✅ 현재 결과 | **하이브리드 파이프라인(MATLAB Agarwal + Python) 완성, AT loss + log-amp 60dB → 90.9%** (논문 96.4%). 선형보간 66.6% → 산란점 기반 90.9%로 대폭 개선. MATLAB 재인증 없이 현 수치로 발표 가능 — "물리기반 증강의 효과를 확인, 완전한 재현에는 Agarwal 파라미터 미세조정 필요"로 정직 보고 |
| 발표 문장 예시 | "few-shot(136장)에서 baseline이 논문과 유사하게 낮게 나옴을 확인 → PH 물리기반 증강으로 OO%까지 개선(논문 96.4% 대비)" |

---

## Exp C — 대비 보정 (Figure 1 / Table 6)

| 필요 항목 | 값 |
|---|---|
| 모델 | **ResNet18** (논문 Table 6 최고 성능 모델) |
| 데이터셋 | SAMPLE (synthetic→measured), 10클래스, K=0 (100% synthetic 학습) |
| 조건 | 증강 없음 vs 대비 보정(CLAHE) 적용 |
| 재현 비교 수치 | 우리 RN18 정확도 **vs** 논문 91.9%→94.5% (K=0) |
| 개선 결과 | **Optuna(부가 개선)**: 자동 탐색된 최적 파라미터(clip_limit, tile_grid_size) + trial별 수렴 그래프 — "논문의 수동 3레벨보다 체계적으로 최적화" |
| 발표 문장 예시 | "synth→measured 도메인 갭을 확인했고, 대비 보정으로 완화됨을 재현. Optuna로 논문보다 원칙적인 파라미터 탐색 수행" |

---

## Exp D — OOD 탐지 (Figure 9)

| 필요 항목 | 값 |
|---|---|
| 모델 | **SMPL** (또는 ID 분류에 쓴 모델) |
| 데이터셋 | ID=SAMPLE 10클래스, OOD=Holdout(J=1,2,3) + SAR-ship(far-OOD 대체, MiniSAR 비공개 대체) |
| 조건 | **ODIN vs Mahalanobis** 비교 |
| 재현 비교 수치 | 우리 AUROC/TNR@95TPR **vs** 논문 Figure 9 (holdout/MSTAR-O/MSTAR-P 각각) |
| 방법론 확장 | **ODIN vs Mahalanobis 비교** — 논문 정성 결론(Maha가 대체로 우세) 재현·정량화 (Exp D 자체엔 팀 개선 #없음; 개선 3종은 A/C/B에 배치) |
| 발표 문장 예시 | "Mahalanobis가 far-OOD(SAR-ship)에서 거의 완벽, near-OOD(holdout)에서는 두 방법 다 어려움 — 논문의 정성적 결론과 일치" |

---

## 종합 슬라이드 (마무리)

| 항목 | 내용 |
|---|---|
| 재현 성공 요약표 | 4실험 × (논문 수치 / 우리 수치 / 판정) 한 표로 |
| 우리 개선 3가지 요약 | SSIM 클러터경계(#1, Exp A) / 대비 자동조절 Optuna(#2, Exp C) / XAI 산란점검증(#3, Exp B) — 각각 무엇을 보여줬는지 한 줄씩 |
| 논문과 의도적으로 다른 점 | Exp C(SAMPLE 채택 이유), Exp D(SAR-ship 대체 이유) — `docs/DATASET_METHOD.md` 표 인용 |
| 한계·향후 과제 | Exp B 완전판 미완성 시 명시, Exp A CTx2 원인 미확정 시 명시 — **정직하게 보고하는 것 자체가 방법론적 엄밀성으로 어필 가능** |

---

## 지금 시점 우선순위 (발표 전 반드시 채워야 할 것)

1. ✅ **Exp B**: 하이브리드 파이프라인 완성, 90.9% 확정 (AT + log-amp 60dB)
2. ✅ **Exp A BUG-3 수정**: `load_condition()` 분할 로직 제거 완료 → **Colab 재실행만 하면 CTx2 수치 갱신**
3. 🟡 **Exp A Colab 재실행**: BUG-3 수정 코드로 CTx2 최종 수치 확보
4. 🟡 **Exp C**: SAMPLE dataset으로 Optuna + CLAHE 실행 → 수치 확보
5. 🟡 **Exp D**: ID=SAMPLE, OOD=holdout+SAR-ship → AUROC/TNR 확보
6. ✅ **Exp B XAI(개선#3)**: 90.9% 모델로 재실행 완료 — IoU 기준 Grad-CAM 8×8 0.12 / Grad-CAM 16×16 0.19 / Occlusion 0.24 / SmoothGrad-IG **0.29** (`run_gradcam_analysis` + `run_xai_analysis`, Cell 7b·7c)

### 시각화/문서 상태
- ✅ `core/evaluate.py`: `plot_confusion_matrix()`, `plot_accuracy_bar()` 구현 완료
- ✅ `notebooks/visualize.ipynb`: 전 실험 시각화 노트북 작성 완료
- ✅ `docs/REPORT.md`: 최신 결과 반영 완료
- ✅ `gradcam/`: GradCAM + scatter_overlap IoU 모듈 구현 완료
