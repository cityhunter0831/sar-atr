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
    """scipy v7.3 이전 포맷 먼저 시도, 실패 시 h5py(v7.3/HDF5)로 재시도."""
    from scipy.io import loadmat
    try:
        return loadmat(str(path), squeeze_me=False, struct_as_record=False)
    except NotImplementedError:
        return _load_mat_hdf5(path)


def _load_mat_hdf5(path):
    """MATLAB v7.3 (.mat = HDF5) 파일을 h5py로 읽어 {key: ndarray} dict 반환."""
    import h5py
    out = {}
    with h5py.File(str(path), "r") as f:
        for k in f.keys():
            v = f[k]
            if isinstance(v, h5py.Dataset):
                arr = v[()]
                # MATLAB은 F-order(column-major)로 저장 → C-order로 전치
                if arr.ndim >= 2:
                    arr = arr.T
                # MATLAB complex: 'r'+'i' compound dtype → complex128
                if arr.dtype.names and set(arr.dtype.names) >= {"real", "imag"}:
                    arr = arr["real"] + 1j * arr["imag"]
                out[k] = arr
    return out


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
# 5클래스. 파일명이 시리얼(BMP2_SN_9563)일 수도, 병합명(BMP2)일 수도, ZSU_23_4/ZSU23 혼용.
AUG_CLASSES = ["2S1", "BMP2", "BTR70", "T72", "ZSU23"]
_SUFFIXES = ("_aug_images", "_baseline", "_test")


def _resolve_class(stem: str) -> str | None:
    """파일명 stem → 5클래스 중 하나. 시리얼명·병합명·ZSU 표기 혼용 모두 대응."""
    s = stem
    for suf in _SUFFIXES:
        if s.endswith(suf):
            s = s[: -len(suf)]
            break
    sl = s.lower()
    # ZSU 표기 통일 (ZSU_23_4 / ZSU23 / zsu23 …)
    if sl.startswith("zsu"):
        return "ZSU23"
    for c in ("2S1", "BMP2", "BTR70", "T72"):
        if sl.startswith(c.lower()):
            return c
    return None


def _find_3d(m: dict) -> np.ndarray:
    """.mat에서 이미지 3D 배열 찾기 (imgTrain/imgTest/... 변수명 무관)."""
    for k in ("imgTrain", "imgTest", "img_test", "images", "img"):
        if k in m:
            return np.asarray(m[k])
    for k, v in m.items():
        if not k.startswith("__") and np.asarray(v).ndim == 3:
            return np.asarray(v)
    raise KeyError("3D 이미지 배열을 못 찾음")


def load_mat_images(mat_dir: str | Path, suffix: str, class_names=AUG_CLASSES):
    """<*>{suffix}.mat 들을 읽어 (images[N,64,64] float32 진폭, labels[N]) 반환.
    suffix: '_aug_images' | '_baseline' | '_test'. 복소면 |·| 진폭. 시리얼→클래스 자동 병합."""
    mat_dir = Path(mat_dir)
    cls_idx = {c: i for i, c in enumerate(class_names)}
    imgs, labels = [], []
    for p in sorted(mat_dir.glob(f"*{suffix}.mat")):
        cls = _resolve_class(p.stem)
        if cls is None or cls not in cls_idx:
            continue
        amp = np.abs(_find_3d(_load_mat(p))).astype(np.float32)
        imgs.append(amp)
        labels.append(np.full(amp.shape[0], cls_idx[cls], dtype=np.int64))
    if not imgs:
        raise FileNotFoundError(f"{mat_dir}에 *{suffix}.mat 없음")
    return np.concatenate(imgs, 0), np.concatenate(labels, 0)


def load_aug_images(mat_dir, class_names=AUG_CLASSES):
    return load_mat_images(mat_dir, "_aug_images", class_names)


try:
    from core.interfaces import SARDataset as _SARDataset, SARSample as _SARSample
except Exception:  # core 미로딩 환경(단독 검사)에서도 import 되게
    _SARDataset = object
    _SARSample = None


class MatImagesDataset(_SARDataset):
    """SARDataset — MATLAB 생성 .mat 이미지 세트. suffix로 용도 구분:
      '_aug_images' (El17 증강 학습), '_baseline' (El17 원본 136), '_test' (El15 실측 평가).
    이미지별 [0,1] 정규화. `train_model(model, ds, test_ds, cfg)`에 바로 투입."""

    def __init__(self, mat_dir: str | Path, suffix: str, class_names=AUG_CLASSES):
        self._class_names = list(class_names)
        imgs, labels = load_mat_images(mat_dir, suffix, class_names)
        mx = imgs.reshape(imgs.shape[0], -1).max(1)[:, None, None] + 1e-8
        self._imgs = (imgs / mx).astype(np.float32)
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


# 편의 래퍼 (용도별)
def AugImagesDataset(mat_dir, class_names=AUG_CLASSES):
    """El17° PH 증강 학습셋 (<class>_aug_images.mat)."""
    return MatImagesDataset(mat_dir, "_aug_images", class_names)


def BaselineDataset(mat_dir, class_names=AUG_CLASSES):
    """El17° few-shot 원본 baseline (136장, <class>_baseline.mat)."""
    return MatImagesDataset(mat_dir, "_baseline", class_names)


def TestImagesDataset(mat_dir, class_names=AUG_CLASSES):
    """El15° 실측 테스트셋 (1913장, <class>_test.mat)."""
    return MatImagesDataset(mat_dir, "_test", class_names)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(f"=== {sys.argv[1]} 변수 목록 ===")
        for k, v in inspect_mat(sys.argv[1]).items():
            print(f"  {k}: shape={v['shape']} dtype={v['dtype']}")
