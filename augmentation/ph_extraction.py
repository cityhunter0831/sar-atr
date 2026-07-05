"""
Phase History (PH) extraction from MSTAR raw SAR data (박승준 담당 — Exp B).

MSTAR binary format:
    - Phoenix ASCII header (key=value pairs) terminated by "EndofPhoenixHeader\n"
    - Binary payload: complex float32, row-major, real+imag interleaved

Pipeline:
    read_mstar_raw → complex HxW array
    → 2D FFT (azimuth × range) → magnitude spectrum
    → extract_scattering_centers → dominant azimuth peaks

Reference: SENSE-Lab-OSU/mstar_data_aug (MATLAB counterpart)
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


# ─── Raw file I/O ─────────────────────────────────────────────────────────────

def read_mstar_header(path: Path) -> dict[str, str]:
    """Parse the Phoenix ASCII header from a raw MSTAR file."""
    import re
    with open(path, "rb") as f:
        peek = f.read(256)
    hlen_m = re.search(rb"PhoenixHeaderLength=\s*(\d+)", peek)
    ascii_len = int(hlen_m.group(1)) if hlen_m else 2048

    header: dict[str, str] = {}
    with open(path, "rb") as f:
        text = f.read(ascii_len).decode("ascii", errors="replace")
    for line in text.splitlines():
        line = line.strip().strip("[]")
        if line.lower().startswith("endofphoenixheader"):
            break
        if "=" in line:
            k, _, v = line.partition("=")
            header[k.strip()] = v.strip()
    return header


def _header_byte_length(path: Path) -> int:
    """Return byte offset where SAR data starts.

    MSTAR Mixed Targets 파일 구조:
        [ASCII header  — PhoenixHeaderLength bytes]
        [Signature data — PhoenixSigSize bytes    ]
        [SAR data (complex float32)               ]

    올바른 오프셋 = PhoenixHeaderLength + PhoenixSigSize.
    PhoenixHeaderLength만 쓰면 Signature 구간을 SAR 데이터로 잘못 읽음.
    """
    import re
    with open(path, "rb") as f:
        peek = f.read(256)

    hlen_m = re.search(rb"PhoenixHeaderLength=\s*(\d+)", peek)
    sig_m  = re.search(rb"PhoenixSigSize=\s*(\d+)", peek)

    if hlen_m:
        # PhoenixSigSize는 이 포맷(ver01.05)에서 파일 전체 크기를 의미하므로
        # 오프셋은 ASCII 헤더 길이(PhoenixHeaderLength)만 사용
        return int(hlen_m.group(1))

    # 폴백: 파일 전체에서 종결자 탐색
    with open(path, "rb") as f:
        data = f.read()
    for marker in (b"[EndofPhoenixHeader]", b"EndofPhoenixHeader\n", b"EndofPhoenixHeader\r\n"):
        idx = data.find(marker)
        if idx != -1:
            return idx + len(marker)
    raise ValueError(f"MSTAR header terminator not found in {path}")


def read_mstar_raw(path: str | Path) -> np.ndarray:
    """
    Read a raw MSTAR file and return complex magnitude image.

    Returns:
        np.ndarray of shape [H, W], float32, linear amplitude (not dB).
    """
    path = Path(path)
    header = read_mstar_header(path)

    n_rows = int(header.get("NumberOfRows", header.get("numrows", 128)))
    n_cols = int(header.get("NumberOfColumns", header.get("numcols", 128)))
    offset = _header_byte_length(path)

    n = n_rows * n_cols
    with open(path, "rb") as f:
        f.seek(offset)
        raw = f.read(n * 8)  # 진폭 블록 + 위상 블록, 각 n개 float32

    # MSTAR 표준 포맷: [진폭 n개][위상 n개] (블록 저장, big-endian).
    # real+imag 교차가 아님 — 교차로 읽으면 이미지가 상하로 갈림(위=진폭 오독, 아래=위상 노이즈).
    vals = np.frombuffer(raw, dtype=">f4").astype(np.float32)
    amplitude = vals[:n].reshape(n_rows, n_cols)   # 첫 블록이 곧 진폭
    return amplitude


def read_mstar_complex(path: str | Path) -> np.ndarray:
    """
    Read a raw MSTAR file and return raw complex array (no abs).

    Returns:
        np.ndarray of shape [H, W], complex64.
    """
    path = Path(path)
    header = read_mstar_header(path)
    n_rows = int(header.get("NumberOfRows", header.get("numrows", 128)))
    n_cols = int(header.get("NumberOfColumns", header.get("numcols", 128)))
    offset = _header_byte_length(path)

    n = n_rows * n_cols
    with open(path, "rb") as f:
        f.seek(offset)
        raw = f.read(n * 8)

    # MSTAR 표준: [진폭 블록][위상 블록] → complex = 진폭 · exp(i·위상)
    vals = np.frombuffer(raw, dtype=">f4").astype(np.float32)
    amplitude = vals[:n]
    phase = vals[n:2 * n]
    return (amplitude * np.exp(1j * phase)).reshape(n_rows, n_cols)


def interpolate_phase_history(
    img_a: np.ndarray,
    img_b: np.ndarray,
    alpha: float = 0.5,
    apply_window: bool = True,
) -> np.ndarray:
    """
    논문 Section 2.1: 두 SAR 이미지의 Phase History 도메인 사이를 보간.

    SENSE-Lab-OSU/mstar_data_aug MATLAB 구현 참조:
    - fftshift/ifftshift로 DC 중앙 정렬
    - Taylor 윈도우로 스펙트럼 leakage 억제

    Args:
        img_a: complex HxW array (elevation angle α)
        img_b: complex HxW array (elevation angle β), same shape as img_a
        alpha: interpolation weight — 0.0 → pure A, 1.0 → pure B
        apply_window: Taylor 윈도우 적용 여부 (MATLAB 참조 구현과 일치)

    Returns:
        Amplitude image float32 [H, W] of the interpolated SAR image.
    """
    if apply_window:
        from scipy.signal.windows import taylor
        h, w = img_a.shape
        win_h = taylor(h, nbar=4, sll=35, norm=False).astype(np.float32)
        win_w = taylor(w, nbar=4, sll=35, norm=False).astype(np.float32)
        window_2d = np.outer(win_h, win_w)
        img_a = img_a * window_2d
        img_b = img_b * window_2d

    # MATLAB: fftshift(fft2(ifftshift(img))) → phase history domain
    ph_a = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(img_a)))
    ph_b = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(img_b)))

    # Linear interpolation in PH domain
    ph_interp = (1.0 - alpha) * ph_a + alpha * ph_b

    # MATLAB: ifftshift(ifft2(fftshift(ph))) → back to image domain
    img_interp = np.fft.fft2(np.fft.ifftshift(ph_interp))
    return np.abs(img_interp).astype(np.float32)


def amplitude_to_tensor(
    amp: np.ndarray, target_size: int = 128, center_crop: int | None = None
) -> Tensor:
    """Normalize amplitude image to [0,1] float32 Tensor [1, H, W].
    이미지 크기 다양(158×158 등) → target_size로 리사이즈.
    center_crop 지정 시(예: 64) 중앙을 잘라냄 — 논문 Exp B는 64×64 center-crop 사용.
    """
    import torch.nn.functional as F
    a = amp.astype(np.float32)
    a = a / (a.max() + 1e-8)
    t = torch.from_numpy(a).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    if a.shape[0] != target_size or a.shape[1] != target_size:
        t = F.interpolate(t, size=(target_size, target_size), mode="bilinear", align_corners=False)
    if center_crop is not None and center_crop < target_size:
        off = (target_size - center_crop) // 2
        t = t[:, :, off:off + center_crop, off:off + center_crop]
    return t.squeeze(0)  # [1, H, W]


def read_azimuth(path) -> float:
    """Phoenix 헤더에서 방위각(TargetAz)을 읽음. PH 보간의 azimuth 이웃 pairing에 사용."""
    try:
        hdr = read_mstar_header(path)
        return float(hdr.get("TargetAz", "nan"))
    except Exception:
        return float("nan")


# ─── Phase history analysis ───────────────────────────────────────────────────

@dataclass
class PhaseHistoryMap:
    """FFT-domain magnitude spectrum and derived scattering information."""
    spectrum: np.ndarray          # [H, W] float32, log-magnitude of 2D FFT
    scattering_centers: list[tuple[float, float]] = field(default_factory=list)
    # [(azimuth_bin, range_bin), ...]  — in FFT coordinates


def compute_phase_history(amplitude: np.ndarray) -> np.ndarray:
    """
    2D FFT of amplitude image → log-magnitude phase history spectrum.

    Centred via fftshift so DC is at image centre.
    """
    spec = np.fft.fftshift(np.fft.fft2(amplitude))
    log_spec = np.log1p(np.abs(spec)).astype(np.float32)
    return log_spec


def extract_scattering_centers(
    amplitude: np.ndarray,
    k: int = 5,
    min_distance: int = 5,
) -> PhaseHistoryMap:
    """
    Identify top-K scattering centres in the phase history domain.

    Args:
        amplitude:    [H, W] float32 linear amplitude image.
        k:            Number of dominant scattering centres to return.
        min_distance: Minimum pixel separation between centres.

    Returns:
        PhaseHistoryMap with spectrum and detected scattering_centers.
    """
    spectrum = compute_phase_history(amplitude)

    # Non-maximum suppression: find local maxima
    from scipy.ndimage import maximum_filter
    local_max = maximum_filter(spectrum, size=min_distance * 2 + 1)
    peaks_mask = (spectrum == local_max)

    # Exclude DC (centre region)
    h, w = spectrum.shape
    cy, cx = h // 2, w // 2
    r_dc = max(h, w) // 8
    ys, xs = np.where(peaks_mask)
    valid = ((ys - cy) ** 2 + (xs - cx) ** 2) > r_dc ** 2

    peak_ys = ys[valid]
    peak_xs = xs[valid]
    peak_vals = spectrum[peak_ys, peak_xs]

    # Top-K
    order = np.argsort(peak_vals)[::-1][:k]
    centers = [(float(peak_ys[i]), float(peak_xs[i])) for i in order]

    return PhaseHistoryMap(spectrum=spectrum, scattering_centers=centers)


def extract_spatial_scattering_centers(
    amplitude: np.ndarray,
    k: int = 5,
    min_distance: int = 5,
) -> list[tuple[float, float]]:
    """
    공간 도메인 amplitude 이미지에서 직접 산란점(밝은 점) 좌표 추출.

    Grad-CAM과 IoU 비교를 위해 공간 도메인 좌표를 반환.
    extract_scattering_centers()의 FFT 도메인과 달리 (row, col) 픽셀 좌표.

    타겟 영역 제한: 밝은 픽셀을 형태학적으로 연결(dilation)해 연결성분을 만들고,
    총 강도가 지배적인 성분(=타겟 덩어리)만 남긴다. 모서리에 고립된 배경 스페클
    봉우리를 산란점으로 오검출하던 문제를 제거 → XAI IoU 지표 정확도 향상.
    """
    from scipy.ndimage import (maximum_filter, gaussian_filter,
                               label, binary_dilation)
    # 스페클 노이즈(단일 픽셀 스파이크) 억제 → 공간적으로 일관된 강한 산란체(타겟)만 남김
    smoothed = gaussian_filter(amplitude, sigma=1.0)
    thr = float(np.percentile(smoothed, 92))
    fg = smoothed > thr
    if not fg.any():
        return []
    # 인접 밝은 픽셀을 연결해 타겟 덩어리 형성 (고립된 배경 스페클은 작은 성분으로 분리)
    lab, n = label(binary_dilation(fg, iterations=2))
    if n == 0:
        return []
    # 성분별 총 강도 → 지배 성분(타겟) 및 그에 준하는 성분만 유지, 약한 배경 성분 배제
    sums = np.array([smoothed[lab == i].sum() for i in range(1, n + 1)])
    areas = np.array([(lab == i).sum() for i in range(1, n + 1)])
    keep = [i + 1 for i in range(n)
            if sums[i] >= 0.2 * sums.max() and areas[i] >= 2]
    target_mask = np.isin(lab, keep) & fg
    # 타겟 영역 내부의 국소 최대만 산란점 후보로
    local_max = maximum_filter(smoothed, size=min_distance * 2 + 1)
    peaks_mask = target_mask & (smoothed == local_max)
    ys, xs = np.where(peaks_mask)
    if len(ys) == 0:                                   # 국소최대 없으면 타겟 영역 최상위 픽셀
        ys, xs = np.where(target_mask)
    vals = smoothed[ys, xs]
    order = np.argsort(vals)[::-1][:k]
    return [(float(ys[i]), float(xs[i])) for i in order]


def visualize_scattering(ph_map: PhaseHistoryMap, save_path: str | None = None):
    """Plot phase history spectrum with marked scattering centres."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(ph_map.spectrum, cmap="hot", origin="upper")
    for y, x in ph_map.scattering_centers:
        ax.plot(x, y, "c+", markersize=10, markeredgewidth=2)
    ax.set_title("Phase History Spectrum + Scattering Centers")
    ax.axis("off")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.tight_layout()
    return fig
