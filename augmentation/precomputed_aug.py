"""
로컬 MATLAB 하이브리드 파이프라인 연동 — precomputed .mat 로더.

로컬에서 MATLAB(원본 mstar_data_aug)으로 1·2단계를 돌려 생성한:
  - phase_histories/<folder>_PH.mat   : arr_azi, arr_img_fft_polar, arr_img_comp, depression
  - recovered_coefficients/<folder>.mat: x_recovered(80000×nChips), gaussWidthStore, selected_indices

를 Python으로 로드해 3단계(방위각 외삽) 또는 학습에 연결.

주의: MATLAB 인덱스는 1-based → Python 0-based 변환 필요.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _load_mat(path):
    from scipy.io import loadmat
    return loadmat(str(path), squeeze_me=False, struct_as_record=False)


def inspect_mat(path: str | Path) -> dict:
    """.mat 파일의 변수명·shape·dtype을 출력용 dict로 반환 (내용 검증용)."""
    m = _load_mat(path)
    out = {}
    for k, v in m.items():
        if k.startswith("__"):
            continue
        arr = np.asarray(v)
        out[k] = {"shape": tuple(arr.shape), "dtype": str(arr.dtype)}
    return out


def load_coefficients(coeff_path: str | Path) -> dict:
    """recovered_coefficients/<folder>.mat 로드.
    반환: x_recovered(80000×nChips 복소), gauss_width, selected_indices(0-based).
    """
    m = _load_mat(coeff_path)
    x = np.asarray(m["x_recovered"])                       # (80000, nChips) complex
    gw = np.asarray(m.get("gaussWidthStore", np.ones((x.shape[1], 1)))).ravel()
    if "selected_indices" in m:
        sel = np.asarray(m["selected_indices"]).ravel().astype(int) - 1   # 1→0 based
    else:
        sel = np.arange(x.shape[1])
    return {"x_recovered": x, "gauss_width": gw, "selected_indices": sel}


def load_phase_history(ph_path: str | Path) -> dict:
    """phase_histories/<folder>_PH.mat 로드.
    반환: azi(방위각), depression(부각), ph_polar(100×100×nAll), img_comp(nAll×128×128).
    """
    m = _load_mat(ph_path)
    return {
        "azi": np.asarray(m["arr_azi"]).ravel().astype(float),
        "depression": np.asarray(m["depression"]).ravel().astype(float),
        "ph_polar": np.asarray(m["arr_img_fft_polar"]),    # (100,100,nAll) complex
        "img_comp": np.asarray(m.get("arr_img_comp", np.zeros((0, 128, 128)))),
    }


def load_folder(coeff_path: str | Path, ph_path: str | Path) -> dict:
    """한 폴더(클래스-시리얼)의 계수 + PH를 selected_indices로 매칭해 반환.

    반환 dict:
      x_recovered  : (80000, nSel) — 선택된 few-shot 샘플들의 산란점 계수
      azi          : (nSel,) 각 샘플 방위각
      depression   : (nSel,) 각 샘플 부각
      ph_polar     : (100,100,nSel) 각 샘플 위상이력 (외삽 검증용)
      gauss_width  : (nSel,)
    """
    rc = load_coefficients(coeff_path)
    ph = load_phase_history(ph_path)
    sel = rc["selected_indices"]

    # PH 파일은 전체 이미지 기준 → selected_indices로 few-shot 샘플만 추림
    nAll = ph["ph_polar"].shape[2]
    if sel.max(initial=-1) >= nAll:
        raise ValueError(f"selected_indices 최대값 {sel.max()} ≥ PH 샘플수 {nAll} — "
                         "인덱스 규약(1 vs 0-based) 또는 파일 짝 확인 필요.")
    return {
        "x_recovered": rc["x_recovered"],                  # (80000, nSel)
        "azi": ph["azi"][sel],
        "depression": ph["depression"][sel],
        "ph_polar": ph["ph_polar"][:, :, sel],
        "gauss_width": rc["gauss_width"],
    }


# ─── 학습용 Dataset: merge_files 출력(<class>_aug_images.mat) 로드 ────────────
# 계약(Antigravity가 생성): imgTrain(N×64×64 복소), aziTrain(N×1), elev(N×1).
# 5클래스 ↔ 시리얼 폴더 매핑 (여러 .mat이 한 클래스로 묶임).
AUG_FOLDER_TO_CLASS = {
    "2S1": "2S1",
    "BMP2_SN_9563": "BMP2", "BMP2_SN_9566": "BMP2", "BMP2_SN_C21": "BMP2",
    "BTR70_SN_C71": "BTR70",
    "T72_SN_132": "T72", "T72_SN_812": "T72", "T72_SN_S7": "T72",
    "ZSU_23_4": "ZSU23",
}
AUG_CLASSES = ["2S1", "BMP2", "BTR70", "T72", "ZSU23"]


def _mat_stem_to_class(stem: str) -> str | None:
    """파일명(예: BMP2_SN_9563_aug_images) → 클래스(BMP2)."""
    folder = stem.replace("_aug_images", "")
    return AUG_FOLDER_TO_CLASS.get(folder)


def load_aug_images(mat_dir: str | Path, class_names=AUG_CLASSES):
    """<class>_aug_images.mat 들을 읽어 (images[N,64,64] float32 진폭, labels[N]) 반환.
    imgTrain은 복소 → |·| 진폭. 여러 시리얼 .mat을 클래스로 합침."""
    mat_dir = Path(mat_dir)
    cls_idx = {c: i for i, c in enumerate(class_names)}
    imgs, labels = [], []
    for p in sorted(mat_dir.glob("*_aug_images.mat")):
        cls = _mat_stem_to_class(p.stem)
        if cls is None or cls not in cls_idx:
            continue
        m = _load_mat(p)
        arr = np.asarray(m["imgTrain"])                    # (N,64,64) complex
        amp = np.abs(arr).astype(np.float32)
        imgs.append(amp)
        labels.append(np.full(amp.shape[0], cls_idx[cls], dtype=np.int64))
    if not imgs:
        raise FileNotFoundError(f"{mat_dir}에 *_aug_images.mat 없음")
    return np.concatenate(imgs, 0), np.concatenate(labels, 0)


try:
    from core.interfaces import SARDataset as _SARDataset, SARSample as _SARSample
except Exception:  # core 미로딩 환경(단독 검사)에서도 import 되게
    _SARDataset = object
    _SARSample = None


class AugImagesDataset(_SARDataset):
    """SARDataset — 로컬 MATLAB이 생성한 증강 이미지(.mat)로 few-shot 학습.
    `train_model(model, AugImagesDataset(...), test_ds, cfg)`로 바로 투입 가능."""

    def __init__(self, mat_dir: str | Path, class_names=AUG_CLASSES,
                 per_image_norm: bool = True):
        self._class_names = list(class_names)
        imgs, labels = load_aug_images(mat_dir, class_names)
        if per_image_norm:                                     # 이미지별 [0,1] 정규화
            mx = imgs.reshape(imgs.shape[0], -1).max(1)[:, None, None] + 1e-8
            self._imgs = (imgs / mx).astype(np.float32)
        else:
            self._imgs = (imgs / (imgs.max() + 1e-8)).astype(np.float32)
        self._labels = labels

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        import torch
        from core.interfaces import SARSample
        img = torch.from_numpy(self._imgs[idx]).unsqueeze(0)   # [1,64,64]
        return SARSample(image=img, label=int(self._labels[idx]),
                         meta={"class_name": self._class_names[self._labels[idx]]})

    @property
    def class_names(self):
        return self._class_names


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(f"=== {sys.argv[1]} 변수 목록 ===")
        for k, v in inspect_mat(sys.argv[1]).items():
            print(f"  {k}: shape={v['shape']} dtype={v['dtype']}")
