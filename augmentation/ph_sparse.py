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
    # 'linear'(부드러움, MATLAB scatteredInterpolant 'natural'에 근접) + convex hull 밖은 nearest
    cart = griddata(pts, ph_polar.ravel(), (XX, YY), method="linear")
    nan = np.isnan(cart)
    if nan.any():
        cart_n = griddata(pts, ph_polar.ravel(), (XX, YY), method="nearest")
        cart[nan] = cart_n[nan]

    fft_cart = np.zeros((N_CROP, N_CROP), dtype=np.complex128)
    off = N_CROP // 2 - N_TARGET // 2
    fft_cart[off:off + N_TARGET, off:off + N_TARGET] = TAYLOR_2D * cart
    img = np.fft.ifftshift(np.fft.ifft2(fft_cart))
    return img


# ─── [STAGE 2] SAR forward 연산자 + FISTA 그룹 희소 복원 ──────────────────────
# sparse_recovery.m + SAR_operator_gen.m 포팅. Eq.6-7.
# 산란점 모델: s(f,θ) = Σ_k Σ_v c[k,v]·ψ_v(θ)·exp(j·4π f cos(φ)/c·(x_k cosθ + y_k sinθ))
#   k=공간격자(100×100=10000), v=가우시안 방위각 기저(D=8), (f,θ)=주파수·방위각(100×100)

_L_PATCH = 30.0          # 패치 30m
_RES = 0.3               # 해상도 0.3m → 100 격자
_N_BASIS = 8             # round(3/0.4) — 방위각 가우시안 기저 개수 (MATLAB)


def _spatial_grid(side: int = N_TARGET):
    """공간 격자 (side×side). 기본=100(논문 L/res). 테스트용 축소 가능."""
    g = np.linspace(-_L_PATCH / 2, _L_PATCH / 2 - _RES, side)
    X, Y = np.meshgrid(g, g)
    return X.ravel(), Y.ravel()


def _azimuth_basis(sigma_g: float):
    """가우시안 방위각 기저 ψ_v(θ). (100 azimuth × D basis), 열 정규화."""
    bisector = 90.0 + THETAS
    centers = 90.0 + np.linspace(-1.5, 1.5, _N_BASIS)
    dist = bisector[:, None] - centers[None, :]
    B = np.exp(-0.5 * dist ** 2 / sigma_g ** 2)
    B = B / np.sqrt((B ** 2).sum(axis=0, keepdims=True))
    return B


class SAROperator:
    """Eq.6 forward/adjoint 연산자. exp 딕셔너리를 __init__에서 1회 precompute(complex64)
    → FISTA 반복마다 matmul만 (mtimesx 대체). grid_side로 격자 축소(테스트).
    메모리: (n_az×n_freq×n_spatial) complex64. side=100 → 0.8GB(Colab), side≤40 → 로컬 가능.
    """

    def __init__(self, depression_deg: float, sigma_g: float = 1.0,
                 grid_side: int = N_TARGET):
        self.X, self.Y = _spatial_grid(grid_side)                     # (side²,)
        self.n_spatial = self.X.size
        self.B = _azimuth_basis(sigma_g).astype(np.complex64)         # (100, D)
        dep = np.deg2rad(depression_deg)
        az = np.deg2rad(90.0 + THETAS)                                # (100,)
        proj = np.cos(az)[:, None] * self.X[None, :] + np.sin(az)[:, None] * self.Y[None, :]  # (100az, S)
        fcoef = 4 * np.pi * F_VEC * np.cos(dep) / C_LIGHT             # (100 freq,)
        # phase[i] = 1/√F · exp(j·fcoef ⊗ proj_i)  →  (100az, 100freq, S) complex64. 1회만.
        self.phase = (1.0 / np.sqrt(N_TARGET) *
                      np.exp(1j * fcoef[None, :, None] * proj[:, None, :])).astype(np.complex64)

    def forward(self, C: np.ndarray) -> np.ndarray:
        """C (S×D) → S (100 freq × 100 azimuth)."""
        ang = self.B @ C.astype(np.complex64).T                       # (100az, S) = B(100,D)@C.T(D,S)
        S = np.einsum("afs,as->fa", self.phase, ang, optimize=True)   # (freq, az)
        return S.astype(np.complex128)

    def adjoint(self, S: np.ndarray) -> np.ndarray:
        """S (100×100) → C (S×D)."""
        Sc = S.astype(np.complex64)
        ang = np.einsum("afs,fa->as", self.phase.conj(), Sc, optimize=True)  # (100az, S)
        return (self.B.conj().T @ ang).T.astype(np.complex128)        # (S, D)


def _group_soft_threshold(C: np.ndarray, tau: float) -> np.ndarray:
    """그룹 prox: 각 행(공간격자점의 D계수) L2 노름 기준 soft-threshold."""
    norms = np.sqrt((np.abs(C) ** 2).sum(axis=1, keepdims=True))      # (10000,1)
    scale = np.maximum(0.0, 1.0 - tau / (norms + 1e-12))
    return C * scale


def sparse_recover(ph_polar: np.ndarray, depression_deg: float,
                   sigma_g: float = 1.0, lam: float = 0.05,
                   n_iter: int = 200, grid_side: int = N_TARGET
                   ) -> tuple[np.ndarray, "SAROperator"]:
    """[STAGE 2] Eq.7 그룹 희소 복원 (FISTA). SPGL1 대체.
       min_C 0.5‖A(C)−S‖² + λ Σ_k‖c_k‖₂
    반환: (복원 계수 C [S×D], 연산자 A)
    """
    A = SAROperator(depression_deg, sigma_g, grid_side=grid_side)
    D = A.B.shape[1]
    n = A.n_spatial
    # 립시츠 상수 근사(파워법 대신 보수적 스텝). ‖A‖² 추정.
    C = np.zeros((n, D), dtype=np.complex128)
    # 스텝: power iteration으로 ‖AᴴA‖(최대고유값) 추정 → step=1/L
    v = np.random.default_rng(0).standard_normal((n, D)) + 0j
    v /= np.linalg.norm(v)
    L = 1.0
    for _ in range(8):
        w = A.adjoint(A.forward(v))
        L = np.linalg.norm(w)
        v = w / (L + 1e-30)
    step = 1.0 / (L * 1.01)                                            # 약간 보수적

    Z = C.copy(); t = 1.0
    for _ in range(n_iter):
        grad = A.adjoint(A.forward(Z) - ph_polar)
        C_new = _group_soft_threshold(Z - step * grad, lam * step)
        t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        Z = C_new + ((t - 1) / t_new) * (C_new - C)
        C, t = C_new, t_new
    return C, A


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
    # 공정 비교 = 중앙 64×64 (마스킹된 유효 영역, 논문도 64 crop)
    a64, b64 = _center_crop(a, 64), _center_crop(b, 64)
    corr_full = np.corrcoef(a.ravel(), b.ravel())[0, 1]
    corr_64 = np.corrcoef(a64.ravel(), b64.ravel())[0, 1]
    print(f"PH shape={ph.shape}  round-trip |corr| full128={corr_full:.3f}  "
          f"center64={corr_64:.3f} (center64가 유효 기준)")
