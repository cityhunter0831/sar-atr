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


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(f"=== {sys.argv[1]} 변수 목록 ===")
        for k, v in inspect_mat(sys.argv[1]).items():
            print(f"  {k}: shape={v['shape']} dtype={v['dtype']}")
