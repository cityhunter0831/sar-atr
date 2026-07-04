"""
Exp B 위상보간 완전판 — Agarwal et al. 2020 (arxiv 2012.09284) 방법의 Python 포팅.

참조: SENSE-Lab-OSU/mstar_data_aug (MATLAB) — preprocess_raw_data.m / sparse_recovery.m /
      generate_aug_images.m. 논문 Fig.3 파이프라인, Eq.6-8.

3단계:
  1. image_to_ph()  : 복소이미지 → K-space → de-Taylor → Cartesian→Polar → 위상이력(PH) [stage 1]
  2. sparse_recover(): PH → 그룹 희소 복원(FISTA)으로 산란점 계수 C [stage 2, 추후]
  3. synthesize()   : 계수 C → ±dθ 외삽 → Polar→Cartesian → Taylor → IFFT [stage 3, 추후]

이 파일은 stage 1과 round-trip 검증부터. MATLAB 상수/연산을 그대로 옮김(주석에 대응 표기).
"""
from __future__ import annotations

import numpy as np

# ─── 상수 (preprocess_raw_data.m / generate_aug_images.m 확인값) ──────────────
C_LIGHT = 299792458.0
F_CENTER = 9.6e9
BANDWIDTH = 521e6
F_LOWER = F_CENTER - BANDWIDTH / 2
N_CROP = 128          # numPixelsCrop
N_TARGET = 100        # numPixelsTarget (freq/angle bins, 격자)
THETA_STEP = 0.03
# thetas = (-1.5:0.03:1.5-0.03) → 100개
THETAS = np.arange(-1.5, 1.5 - 1e-9, THETA_STEP)[:N_TARGET]
F_VEC = np.linspace(F_LOWER, F_LOWER + BANDWIDTH, N_TARGET)  # 100


def _taylor_window_2d(n: int = N_TARGET) -> np.ndarray:
    """taylorwin(n,4,-35) ⊗ taylorwin(n,4,-35)  (MATLAB SLL=-35 == scipy sll=35)."""
    from scipy.signal.windows import taylor
    w = taylor(n, nbar=4, sll=35, norm=False).astype(np.float64)
    return np.outer(w, w)


TAYLOR_2D = _taylor_window_2d()


def _polar_kgrid(depression_deg: float):
    """preprocess_raw_data.m: 극좌표 k_1,k_2 (freq×angle 격자에서의 파수 좌표)."""
    dep = np.deg2rad(depression_deg)
    coef = 4 * np.pi / C_LIGHT * np.cos(dep)
    theta = np.deg2rad(THETAS)
    # fRep[i,j]=f[i], thetaRep[i,j]=thetas[j]  (MATLAB repmat 규약)
    fRep = np.tile(F_VEC[:, None], (1, N_TARGET))         # (100,100)
    thetaRep = np.tile(theta[None, :], (N_TARGET, 1))     # (100,100)
    k_1 = coef * fRep * np.sin(thetaRep)                  # cross-range 파수
    k_2 = coef * fRep * np.cos(thetaRep)                  # range 파수
    return k_1, k_2


def _center_crop(img: np.ndarray, size: int) -> np.ndarray:
    """중앙 size×size crop (MATLAB centerIm 기준: floor(N/2))."""
    h, w = img.shape
    cy, cx = h // 2, w // 2
    return img[cy - size // 2:cy - size // 2 + size,
               cx - size // 2:cx - size // 2 + size]


def image_to_ph(complex_img: np.ndarray, depression_deg: float) -> np.ndarray:
    """[STAGE 1] 복소 SAR 이미지 → 위상이력(PH) 극좌표 표현 (100×100 복소).

    preprocess_raw_data.m 포팅:
      flipud → square/crop 128 → mask 64 → fftshift(fft2(ifftshift)) →
      de-Taylor + crop 100 → Cartesian→Polar 보간(interp2 spline).
    """
    from scipy.interpolate import RectBivariateSpline

    img = np.flipud(np.asarray(complex_img))              # MATLAB: flipud(ImageData)
    n = min(img.shape)                                    # "square" the image
    img = img[:n, :n]
    img = _center_crop(img, N_CROP)                       # 128×128 중앙

    # 이미지 도메인 64×64 중앙 마스크 후 K-space 변환
    mask = np.zeros((N_CROP, N_CROP), dtype=np.float64)
    c = N_CROP // 2
    mask[c - 32:c + 32, c - 32:c + 32] = 1.0
    fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(mask * img)))

    # 중앙 100×100 crop + Taylor 윈도우 제거(de-window)
    off = c - N_TARGET // 2                                # 64-50 = 14
    fft_crop = fft[off:off + N_TARGET, off:off + N_TARGET] / TAYLOR_2D

    # Cartesian(정규격자) → Polar(k_1,k_2) 보간. XX,YY = 정규격자.
    k_1, k_2 = _polar_kgrid(depression_deg)
    xx = np.linspace(k_1.min(), k_1.max(), N_TARGET)
    yy = np.linspace(k_2.min(), k_2.max(), N_TARGET)
    # RectBivariateSpline: 격자(yy=행, xx=열) 위 값 → (k_2,k_1) 질의. 복소는 real/imag 분리.
    def _interp(vals):
        sp = RectBivariateSpline(yy, xx, vals, kx=3, ky=3)
        return sp.ev(k_2.ravel(), k_1.ravel()).reshape(N_TARGET, N_TARGET)
    ph = _interp(fft_crop.real) + 1j * _interp(fft_crop.imag)
    return ph.astype(np.complex128)


def ph_to_image(ph_polar: np.ndarray, depression_deg: float) -> np.ndarray:
    """[검증용 역변환] PH 극좌표 → 이미지. Polar→Cartesian → Taylor → IFFT.

    generate_aug_images.m의 재구성부와 동일 구조. stage 1 round-trip 검증에 사용.
    (모델 없이 그대로 되돌리므로 근사 복원 — 산란점 모델 미적용)
    """
    from scipy.interpolate import griddata

    k_1, k_2 = _polar_kgrid(depression_deg)
    xx = np.linspace(k_1.min(), k_1.max(), N_TARGET)
    yy = np.linspace(k_2.min(), k_2.max(), N_TARGET)
    XX, YY = np.meshgrid(xx, yy)
    pts = np.column_stack([k_1.ravel(), k_2.ravel()])
    cart = griddata(pts, ph_polar.ravel(), (XX, YY), method="nearest", fill_value=0)

    fft_cart = np.zeros((N_CROP, N_CROP), dtype=np.complex128)
    off = N_CROP // 2 - N_TARGET // 2
    fft_cart[off:off + N_TARGET, off:off + N_TARGET] = TAYLOR_2D * cart
    img = np.fft.ifftshift(np.fft.ifft2(fft_cart))
    return img


# ─── round-trip 자체검증 (합성 데이터로 파이프라인 sanity check) ──────────────
if __name__ == "__main__":
    # 실데이터 없이 파이프라인이 도는지 + round-trip 상관도 확인
    rng = np.random.default_rng(0)
    img = np.zeros((128, 128), dtype=np.complex128)
    img[58:70, 55:75] = (rng.random((12, 20)) + 1j * rng.random((12, 20)))  # 가짜 타겟
    ph = image_to_ph(img, depression_deg=17.0)
    rec = ph_to_image(ph, depression_deg=17.0)
    a = np.abs(_center_crop(np.flipud(img), 128))
    b = np.abs(rec)
    corr = np.corrcoef(a.ravel(), b.ravel())[0, 1]
    print(f"PH shape={ph.shape}, round-trip |corr|={corr:.3f} "
          f"(1에 가까울수록 좌표/FFT 규약 정확)")
