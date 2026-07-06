# MATLAB Pipeline Backup (Agarwal PH sparse-recovery augmentation)

원본 repo: `SENSE-Lab-OSU/mstar_data_aug` (fork 아님, push 불가) → 로컬 수정본을 이곳에 백업.
실제 실행은 로컬 MATLAB에서, 산출물(`.mat`)은 Google Drive를 통해 Colab(`sar-atr`)으로 전달.

## 파이프라인 단계
| 단계 | 파일 | 하는 일 |
|---|---|---|
| 1 | `preprocess_raw_data.m` | 복소이미지 → K-space → Polar PH |
| 2 | `sparse_recovery.m` | Eq.7 그룹 희소복원 (SPGL1) → 산란점 계수 C |
| 3 | `generate_aug_images.m` | Eq.8 ±6° 외삽 + subpixel shift → 증강 이미지 |
| 4 | `merge_files.m` | 클래스별 샘플 병합 → `<class>_aug_images.mat` |
| export | `export_baseline_data.m`, `export_test_data.m` | El17 baseline(136) / El15 test(1913) |
| util | `SAR_operator_gen.m`, `reconError.m`, `taylorwin.m`, `profile_run.m`, `test_gauss_width.m` | 연산자·유틸 |

## 주요 수정 내역 (원본 대비)
- **인덱싱 버그 수정** (`generate_aug_images.m`): `x_recovered`는 few-shot 압축순서(1..N)로 저장되므로
  전체 PH 배열(`PH.arr_img_fft_polar`, `PH.depression`, `PH.arr_azi`) 접근 시
  `selected_idx = RC.selected_indices(idxTrain)`로 매핑. 이전엔 `idxTrain` 직접 사용 →
  서로 다른 칩이 섞여 상관 0.18. 수정 후 **0.997**.
- **가속**: `mtimesx`(MEX) → `pagemtimes`, 익명함수 슬라이싱 제거, `maxNumCompThreads(1)` (26분→18초/칩)
- **툴박스 제거**: `pdist2` → `abs(A - B.')`, `imrotate` → `interp2` 백워드 매핑
- **메모리**: 4GB `A_mod` 딕셔너리 재계산 → pulse별 루프 + phase-shift 수식 (16MB)
- **σ_G=1.0 고정**: 2S1에서 σ_G=1 vs 2 잔차차이 0.002% 정량검증 → 라인서치 제거

## few-shot 분포 (총 136장)
2S1:24, BMP2:32(11/11/10), BTR70:24, T72:24(8/8/8), ZSU23:32
