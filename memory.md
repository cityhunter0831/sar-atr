# memory.md — 세션 연속성 핸드오프 (다른 채팅에서 이어가기)

> 이 문서는 "지금 어디까지 왔고, 다음에 뭘 해야 하는지"의 스냅샷.
> 상세 배경은 `CLAUDE.md`(지향점·트러블슈팅·진척도) + `docs/`(PAPER_SPEC/DATASET_METHOD/THEORY_REFERENCES/PRESENTATION_CHECKLIST).
> 브랜치: `claude/kind-gauss-84kr3m`. 매 작업 후 커밋·푸시.

## 프로젝트 한 줄
Geng et al. 2023 SAR-ATR 논문 4개 실험 재현 + 우리 팀 개선 3개. 코드는 GitHub(sar-atr), 데이터·결과는 Google Drive, 학습은 Colab(GPU).

**우리 팀 개선 3종 (혼동 금지)**: #1 SSIM 클러터경계(Exp A) / #2 대비 자동조절 Optuna(Exp C) / #3 XAI 산란점검증(Exp B). ※옵티마이저(ADAM vs SGD) 비교는 개선사항 아님 — 과거 문서 드리프트로 잘못 들어갔던 항목.

## 실험 현황 요약
| Exp | 목표 | 현재 상태 |
|---|---|---|
| A 클러터전이 | Table4 재현 | 🟢 **최종 확정**: TrainOR→CT 67.7%(하락) / TrainCT 96.3%(회복, 논문 91.5% 초과). CTx2 40.1% 붕괴 원인 **공개 데이터 불완전**(btr70/t72/zsu23 0장) — 재현 불가, 정직 보고. |
| B PH보간 few-shot ⭐ | 56.6%→96.4% | 🟢 **66.6%→90.9%** (log-amp 60dB/AT). 하이브리드 완주, 논문 근접 (아래 상세) |
| C 대비증강 | SAMPLE 91.9%→94.5% | 🟢 **최종 수치 확정** (누수 방지 적용): ①no-aug 69.6% / ②논문 0.5×3 61.8% / ③Optuna(strength=0.672,levels=4) **80.3%** → **+18.5%p 개선**. 논문 0.5가 오히려 해가 됨을 정량 입증. |
| D OOD | ID=SAMPLE, ODIN vs Maha | 🟢 **최종 수치 확정**: far-OOD(sarship) Maha AUROC=1.000 완벽 / near-OOD(holdout) 두 방법 모두 ~0.45(한계). 논문 패턴 재현. |

## ⭐ Exp B 하이브리드 파이프라인 (지금의 핵심 작업)
**노선**: 로컬 MATLAB(원본 Agarwal repo, 다른 채팅에서 실행)로 증강 데이터 생성 → Drive → Colab Python(이 repo)에서 학습.
- 왜: OSU Box precomputed 삭제(404) + Python 포팅은 검증 왕복이 김 → 원본 MATLAB 직접 실행이 정확·빠름(가속 완료 18초/장).
- ✅ `gaussWidth=1.0` 편차 검증 완료: 2S1에서 σ_G=1(9419.6) vs σ_G=2(9419.4) 잔차차이 **0.002%** → σ_G 민감도 낮음 정량 확인. **1.0 고정 확정, 재복원 불필요.** (blind 아니라 검증 후 고정 → 발표 방어 가능)

### 데이터 3종 (`.mat`, Drive 통해 전달)
- `<class>_aug_images.mat`: El17 증강 학습(imgTrain N×64×64 복소, 샘플당196장, v7.3/HDF5) — **🔄 버그 수정 후 재생성 중**
- `<class>_baseline.mat`: El17 원본 136장 — ✅ 완료
- `<class>_test.mat`: El15 실측 1913장(imgTest) — ✅ 완료
- 5클래스=2S1,BMP2,BTR70,T72,ZSU23. few-shot 분포 24/32/24/24/32=136. aug 총 26,656장(136×196), test 총 1,913장(274/587/196/582/274).

### 🔴 발견·수정한 버그 (이번 세션 핵심)
1. **v7.3 로더**: `.mat`가 MATLAB v7.3(HDF5)로 저장돼 `scipy.io.loadmat`이 `NotImplementedError`. → `precomputed_aug.py` `_load_mat`에 h5py 폴백 추가(F-order 전치, complex compound dtype real+imag 처리). 커밋 `0416cf2`.
2. **인덱싱 버그**(`generate_aug_images.m`): `x_recovered`는 few-shot 압축순서(1..N)인데 전체 PH 배열(`arr_img_fft_polar`/`depression`/`arr_azi`)을 같은 idxTrain으로 접근 → 다른 칩 신호가 잔차에 섞임(상관 0.18). `selected_idx=RC.selected_indices(idxTrain)` 매핑으로 수정 → 상관 0.997. MATLAB 파일은 `matlab_pipeline/`에 백업.
3. **로그진폭(dB) 전처리**(`precomputed_aug.py` `_normalize_amplitude`): 버그 수정 후에도 선형진폭은 aug 72.1% 정체(epoch 60·120 동일 → 학습 포화). SAR 진폭 dynamic range가 커 CNN이 피크 산란점만 학습→T72/BMP2 혼동. per-image 정규화→dB압축→clip→[0,1] (`log_scale=True, dyn_range_db=60`) 도입 → **72.1%→90.9%**. dB 스윕 결과 60dB 피크(50=90.1/60=90.9/80=90.5). **AT>LSM** (LSM은 합성 T72 경계 흐려 72%로 폭락).

### Python 쪽 준비 (완료)
`augmentation/precomputed_aug.py`: `AugImagesDataset/BaselineDataset/TestImagesDataset`(SARDataset 호환), `inspect_mat`, `_resolve_class`(파일명 매핑 검증됨).

### 다음에 할 일 (순서)
1. ✅ stage2 계수 136장 완공, σ_G=1.0 확정 / ✅ v7.3 로더·인덱싱 버그 수정 / ✅ 재생성·재학습 완료
2. ✅ Colab 60 epoch: baseline 66.6% / **aug 90.9%** (log-amp 60dB/AT) — REPORT.md 확정 완료
3. 잔여 갭(90.9 vs 96.4, T72 85%·2S1 84% 병목): σ_G를 T72/2S1에도 클래스별 재검증하면 개선 여지(현재 2S1만 검증). 로컬 MATLAB 시간 있을 때.
4. Grad-CAM(개선#3)을 완전판 aug 모델로 재실행 가능 (완전판 모델 확보됨)

### 최종 학습 셀 (데이터 준비되면)
```python
%cd /content/repo
!git pull
import sys, importlib
for m in list(sys.modules):
    if m.split('.')[0] in ('augmentation','core','experiments'): del sys.modules[m]
importlib.invalidate_caches()
from augmentation.precomputed_aug import AugImagesDataset, BaselineDataset, TestImagesDataset
from core.models import get_model
from core.train import train_model
from core.interfaces import TrainConfig
DATA_DIR='경로/gen_aug_data'
test_ds=TestImagesDataset(DATA_DIR); baseline_ds=BaselineDataset(DATA_DIR); aug_ds=AugImagesDataset(DATA_DIR)
cfg=TrainConfig(model_name='smpl',num_classes=5,epochs=60,loss_type='at')
_,r1=train_model(get_model('smpl',5),baseline_ds,test_ds,cfg)
_,r2=train_model(get_model('smpl',5),aug_ds,test_ds,cfg)
print(f'MSTAR-R {r1.accuracy*100:.1f}% (논문56.6) / MSTAR-Aug {r2.accuracy*100:.1f}% (논문96.4)')
```

## 두 채팅 병행 규칙
- 이 채팅(Claude/sar-atr repo): Python 로더·학습·문서. 다른 채팅(Antigravity/Agarwal repo): MATLAB 증강 생성.
- 인터페이스=`.mat` 파일(Drive). 사용자가 다리 역할로 규격·결과 전달.
- MATLAB 에러/수식 문제는 이 채팅에 붙여주면 논문·코드 대조로 해결(내가 강한 영역).

## 미해결·주의 (새 세션이 알아야 할 것)
- Exp A T5(CTx2 붕괴): 진단 로깅만 넣음, Colab 재실행 필요.
- Exp D: ID=SAMPLE로 재설계됨, 구버전 수치(REPORT.md)는 ID=MSTAR라 재실행 후 교체.
- gaussWidth=1.0 편차: 정확도 미달 시 1순위 의심.
- Colab 세션 끊기면 데이터(/content)는 날아감 → data는 Drive에, results도 Drive 심볼릭. 노트북 Cell1,3 재실행.
- 커밋 규칙: feat/fix/exp/docs. 푸시는 `git push -u origin claude/kind-gauss-84kr3m`.
