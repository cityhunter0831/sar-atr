"""
SAMPLE 데이터셋 복소 원본(.mat) 로더 — Exp C 논문 충실 재현용.

benjaminlewis-afrl/SAMPLE_dataset_public 저장소의 `mat_files/{real,synth}/<class>/*.mat`
(우리가 Exp B에 쓰는 MATLAB 희소복원 산출물 .mat과는 완전히 별개 — 이건 SAMPLE
데이터셋 자체가 원래 배포하는 복소 원본이다).

논문 Section 4.3: "grayscale images with multiple different contrast levels are
generated with the complex image data in '.mat' format" — 우리가 지금까지 쓰던
ColorJitter(이미 8bit로 렌더링된 PNG에 사후 대비 조작)와 달리, 이 모듈은 복소
원본에서 직접 대비 레벨을 만든다.

⚠️ 실제 .mat 내부 변수명은 원저장소에 문서화돼 있지 않아 이 환경에서 확인 불가했다.
`_find_complex_2d()`가 여러 후보 이름을 시도하고, 실패하면 파일 안의 아무 2D
(복소 또는 float) 배열이나 채택한다 — 노트북 Cell 5의 `inspect_mat()` 진단 출력으로
실제 변수명을 확인해 필요하면 후보 목록을 좁혀도 된다.

⚠️ Contrast 1/Contrast 2의 정확한 dB 파라미터는 논문에 없음(Figure 8 그림으로만
제시). 아래 DYN_RANGE_PRESETS_DB는 "Original보다 점점 더 어두운 배경"이라는
Figure 8의 육안 경향을 근사한 값이며, Colab에서 실제 렌더링 결과를 Figure 8과
대조해 조정이 필요할 수 있다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from augmentation.precomputed_aug import _load_mat, _normalize_amplitude
from core.interfaces import SARDataset, SARSample

# 논문 Figure 8: Original → Contrast 1 → Contrast 2로 갈수록 배경이 어두워짐
# (다이나믹레인지가 좁아지는 방향) — 근사치, Colab에서 육안 대조 후 조정 필요.
CONTRAST_LEVEL_NAMES = ["original", "contrast1", "contrast2"]
DYN_RANGE_PRESETS_DB = {"original": 60.0, "contrast1": 35.0, "contrast2": 20.0}

# 후보 변수명 — SAMPLE mat_files는 이미지 1장당 파일 1개(png와 1:1 대응)로 추정.
_CANDIDATE_KEYS = ["complex_img", "img", "data", "x", "image", "iq", "cimg"]


def _find_complex_2d(m: dict) -> np.ndarray:
    """.mat dict에서 이미지 배열(2D, complex 또는 float)을 찾는다.
    후보 이름 우선 탐색 → 실패 시 __로 시작하지 않는 첫 2D 배열을 채택."""
    for k in _CANDIDATE_KEYS:
        if k in m:
            arr = np.asarray(m[k])
            if arr.ndim == 2:
                return arr
    for k, v in m.items():
        if k.startswith("__"):
            continue
        arr = np.asarray(v)
        if arr.ndim == 2 and arr.size > 1:
            return arr
    raise KeyError(
        f".mat에서 2D 이미지 배열을 못 찾음. 발견된 키: {[k for k in m if not k.startswith('__')]} "
        "— 노트북 Cell 5의 inspect_mat() 출력을 확인해 _CANDIDATE_KEYS를 조정할 것."
    )


def load_complex_amplitude(mat_path: str | Path) -> np.ndarray:
    """.mat 1개 → 진폭(amplitude) 2D float32 배열. 복소면 |·|, 이미 실수면 그대로."""
    m = _load_mat(mat_path)
    arr = _find_complex_2d(m)
    if np.iscomplexobj(arr):
        amp = np.abs(arr)
    else:
        amp = np.asarray(arr, dtype=np.float64)
    return amp.astype(np.float32)


def _class_dir(root: Path, split: str, cls: str) -> Path | None:
    """대소문자/언더스코어 차이 허용하며 <root>/<split>/<cls> 탐색."""
    split_dir = root / split
    if not split_dir.exists():
        return None
    direct = split_dir / cls
    if direct.exists():
        return direct
    for d in sorted(split_dir.iterdir()):
        if d.name.lower() == cls.lower():
            return d
    return None


def collect_mat_files(root: Path, split: str, class_names: list[str]) -> dict[str, list[Path]]:
    """mat_files/<split>/<class>/*.mat 수집. png_images와 동일 클래스 폴더 구조 가정."""
    result: dict[str, list[Path]] = {c: [] for c in class_names}
    for cls in class_names:
        cls_dir = _class_dir(root, split, cls)
        if cls_dir is None:
            continue
        result[cls] = sorted(cls_dir.glob("*.mat"))
    return result


def render_contrast_levels(
    amp_images: np.ndarray, levels: list[str] = CONTRAST_LEVEL_NAMES,
) -> dict[str, np.ndarray]:
    """진폭 이미지 배치[N,H,W] → {레벨명: [0,1] 정규화된 [N,H,W]} 3종.
    각 레벨은 서로 다른 dB dynamic range로 압축한 고정 프리셋(논문 Figure 8과 동일 발상:
    무작위 지터가 아니라 결정론적 3가지 버전)."""
    out = {}
    for level in levels:
        db = DYN_RANGE_PRESETS_DB[level]
        out[level] = _normalize_amplitude(amp_images, log_scale=True, dyn_range_db=db)
    return out


class SampleContrastDataset(SARDataset):
    """SAMPLE mat_files 기반, 논문 Figure 8 방식 3단계 고정 대비 증강 데이터셋.

    소스 이미지마다 Original/Contrast1/Contrast2 3개 버전을 만들어 806장→2418장
    패턴을 재현한다(논문 그대로 결정론적 고정 버전 — 우리 기존 ColorJitter처럼
    매번 랜덤으로 다시 뽑는 방식이 아님).

    lazy loading: __init__은 (경로, 라벨, 레벨) 튜플만 저장, 실제 .mat 읽기·렌더링은
    __getitem__에서 수행.
    """

    def __init__(
        self, root: Path, split: str, class_names: list[str],
        levels: list[str] = CONTRAST_LEVEL_NAMES, target_size: int = 128,
    ):
        self._class_names = class_names
        self._levels = levels
        self._target_size = target_size
        files_by_class = collect_mat_files(root, split, class_names)
        self._samples: list[tuple[Path, int, str]] = []
        for idx, cls in enumerate(class_names):
            for p in files_by_class.get(cls, []):
                for level in levels:
                    self._samples.append((p, idx, level))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> SARSample:
        path, label, level = self._samples[idx]
        try:
            amp = load_complex_amplitude(path)
        except Exception:
            amp = np.zeros((self._target_size, self._target_size), dtype=np.float32)

        db = DYN_RANGE_PRESETS_DB[level]
        norm = _normalize_amplitude(amp[None, ...], log_scale=True, dyn_range_db=db)[0]
        t = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        if norm.shape[0] != self._target_size or norm.shape[1] != self._target_size:
            t = F.interpolate(t, size=(self._target_size, self._target_size),
                              mode="bilinear", align_corners=False)
        t = t.squeeze(0)  # [1,H,W]

        return SARSample(
            image=t, label=label,
            meta={"class_name": self._class_names[label], "level": level, "source": str(path)},
        )

    @property
    def class_names(self) -> list[str]:
        return self._class_names


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        amp = load_complex_amplitude(p)
        print(f"{p}: amplitude shape={amp.shape} dtype={amp.dtype} "
              f"min={amp.min():.4g} max={amp.max():.4g}")
