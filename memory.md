# memory.md — 세션 연속성 핸드오프 (다른 채팅에서 이어가기)

> 이 문서는 "지금 어디까지 왔고, 다음에 뭘 해야 하는지"의 스냅샷.
> 상세 배경은 `CLAUDE.md`(지향점·트러블슈팅·진척도) + `docs/`(PAPER_SPEC/DATASET_METHOD/THEORY_REFERENCES/PRESENTATION_CHECKLIST).
> 브랜치: `claude/kind-gauss-84kr3m`. 매 작업 후 커밋·푸시.

## 프로젝트 한 줄
Geng et al. 2023 SAR-ATR 논문 4개 실험 재현 + 우리 팀 개선 3개. 코드는 GitHub(sar-atr), 데이터·결과는 Google Drive, 학습은 Colab(GPU).

## 실험 현황 요약
| Exp | 목표 | 현재 상태 |
|---|---|---|
| A 클러터전이 | Table4 재현 | 🟠 TrainOR→CT 하락·CT회복은 재현. **CTx2 붕괴(39% vs 논문96%) 원인 미확정(T5)** — Colab 재실행해 `[T5 진단]` 폴더로드수 확인 필요 |
| B PH보간 few-shot ⭐ | 56.6%→96.4% | 🔄 **하이브리드 파이프라인 진행 중** (아래 상세) |
| C 대비보정 | SAMPLE 91.9%→94.5% | 🟡 73.6%→80.9% 나옴. Optuna 결과 정리 필요 |
| D OOD | ID=SAMPLE, ODIN vs Maha | 🟠 코드 재설계 완료(ID=SAMPLE), **재실행해 수치 갱신 필요** |

## ⭐ Exp B 하이브리드 파이프라인 (지금의 핵심 작업)
**노선**: 로컬 MATLAB(원본 Agarwal repo, 다른 채팅에서 실행)로 증강 데이터 생성 → Drive → Colab Python(이 repo)에서 학습.
- 왜: OSU Box precomputed 삭제(404) + Python 포팅은 검증 왕복이 김 → 원본 MATLAB 직접 실행이 정확·빠름(가속 완료 18초/장).
- ⚠️ 유일한 편차: `gaussWidth=1.0` 고정. **합성품질=정확도에 영향**하므로, 96% 미달 시 σ_G 재검토. 대표이미지로 σ_G∈{1,2,3} 잔차비교해 최적 고정 권장.

### 데이터 3종 (`.mat`, Drive 통해 전달)
- `<class>_aug_images.mat`: El17 증강 학습(imgTrain N×64×64 복소, 샘플당196장) — **🔴 아직 생성 중(stage3 대기)**
- `<class>_baseline.mat`: El17 원본 136장 — ✅ 완료
- `<class>_test.mat`: El15 실측 1913장(imgTest) — ✅ 완료
- 5클래스=2S1,BMP2,BTR70,T72,ZSU23. few-shot 분포 24/32/24/24/32=136.

### Python 쪽 준비 (완료)
`augmentation/precomputed_aug.py`: `AugImagesDataset/BaselineDataset/TestImagesDataset`(SARDataset 호환), `inspect_mat`, `_resolve_class`(파일명 매핑 검증됨).

### 다음에 할 일 (순서)
1. 로컬 MATLAB: stage3(`generate_aug_images.m`)+merge → `<class>_aug_images.mat` 9개 생성 → Drive 업로드
2. Colab에서 `inspect_mat`로 변수명·shape 검증 (특히 test의 imgTest 변수명 확인)
3. 최종 학습 셀(아래) 실행 → baseline 56.6% vs aug 96.4% 재현
4. 되면 REPORT.md 수치 갱신, Grad-CAM(개선#3)을 이 완전판 모델로 재실행

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
