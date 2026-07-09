"""
Exp B — Phase History Interpolation Augmentation (논문 Section 2.1, Table 3)

⭐ 실제 학습 파이프라인은 로컬 MATLAB(Agarwal 희소복원)이 생성한 .mat 파일을
augmentation/precomputed_aug.py(AugImagesDataset/BaselineDataset/TestImagesDataset)로
로드해 notebooks/colab_template.ipynb Cell 7a에서 직접 학습한다.
정확한 설계는 docs/PAPER_SPEC.md 참조.

이 파일은 그 파이프라인이 만든 체크포인트를 대상으로 하는 해석가능성 분석만 담당한다:
  - run_gradcam_analysis(): 개선 #3 — Grad-CAM 히트맵 vs 산란점 IoU
  - run_xai_analysis(): 개선 #3 확장 — Occlusion/SmoothGrad-IG 픽셀 XAI vs 산란점 IoU
  - _collect_raw_files()/_depression_angle(): scripts/diag_mstar_format.py에서도 재사용

과거에는 이 파일 안에 자체 학습 루프(run(), PHAugmentedDataset)가 있었으나,
azimuth 이웃 두 이미지를 PH 도메인에서 단순 선형평균하는 방식(산란점 미사용,
논문 방법 아님, few-shot에서 65% 정체의 원인)이라 삭제했다. 실제 증강은
MATLAB 희소복원(Eq.7 그룹 Lasso)이 담당한다.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from augmentation.ph_extraction import (
    amplitude_to_tensor,
    extract_scattering_centers,
    extract_spatial_scattering_centers,
    read_mstar_header,
    read_mstar_raw,
    visualize_scattering,
)
from core.models import get_model
from gradcam.cam import GradCAM
from gradcam.scatter_overlap import centers_to_mask, iou as compute_iou
from gradcam.attributions import occlusion_sensitivity, smoothgrad_ig

# 논문 Table 2/3 재현: 5클래스 few-shot. train El17° / test El15°.
# BMP2·BTR70·T72 = Targets 패키지, 2S1·ZSU23 = Mixed Targets → 세 디렉토리 모두 로드.
MSTAR_RAW_DIRS = [
    Path("data/mstar/MSTAR_PUBLIC_TARGETS_CHIPS_T72_BMP2_BTR70_SLICY"),
    Path("data/mstar/MSTAR_PUBLIC_MIXED_TARGETS_CD1"),
    Path("data/mstar/MSTAR_PUBLIC_MIXED_TARGETS_CD2"),
]
MSTAR_RAW_DIR = MSTAR_RAW_DIRS[-1]  # 하위호환
RESULTS_DIR = Path("results/exp_b")

# 논문 5클래스 (canonical) + 파일 경로 매칭 별칭 (패키지마다 폴더명이 다름)
CLASSES = ["2S1", "BMP2", "BTR70", "T72", "ZSU23"]
CLASS_ALIASES = {
    "2S1":   ["2S1", "2s1"],
    "BMP2":  ["BMP2", "bmp2"],
    "BTR70": ["BTR70", "btr70"],
    "T72":   ["T72", "t72"],
    "ZSU23": ["ZSU23", "ZSU_23_4", "zsu23", "ZSU_23"],
}
EXP_B_INPUT = 64  # 논문: 64×64 center-crop (T2)


# ─── MSTAR raw file dataset ───────────────────────────────────────────────────

def _has_phoenix_header(path: Path) -> bool:
    """파일 앞 4KB만 읽어 Phoenix 헤더 존재 여부를 빠르게 확인."""
    try:
        with open(path, "rb") as f:
            chunk = f.read(4096)
        return b"PhoenixHeaderVer" in chunk
    except Exception:
        return False


def _collect_raw_files(
    roots: "Path | list[Path]", classes: list[str]
) -> dict[str, list[Path]]:
    """Mixed Targets 디렉토리(들)에서 클래스별 raw 파일 목록 수집.
    여러 디스크(CD1+CD2)를 합산 지원. Phoenix 헤더 없는 파일은 제외."""
    if isinstance(roots, Path):
        roots = [roots]

    result: dict[str, list[Path]] = {c: [] for c in classes}

    def _match_class(path: Path) -> str | None:
        """경로의 '폴더명(part)' 단위로 클래스 판별.
        패키지 루트명(..._T72_BMP2_BTR70_SLICY)에 클래스명이 섞여 있으므로
        substring이 아니라 part가 alias로 시작하는지로 판별해야 오분류 방지."""
        parts = path.parts
        for cls in classes:
            for alias in CLASS_ALIASES.get(cls, [cls]):
                al = alias.lower()
                for part in parts:
                    pl = part.lower()
                    # 폴더명이 alias와 정확히 같거나 alias_로 시작 (시리얼 서브폴더 대응)
                    if pl == al or pl.startswith(al + "_") or pl.startswith(al + "-"):
                        return cls
        return None

    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not (p.is_file() and p.suffix.lstrip(".").isdigit() and len(p.suffix) >= 3):
                continue
            cls = _match_class(p)
            if cls is None:
                continue
            if _has_phoenix_header(p):
                result[cls].append(p)
    return result


# ─── train/test 분리 ──────────────────────────────────────────────────────────

_DEPRESSION_KEYS = [
    "DesiredDepression", "MeasuredDepression", "Depression",
    "DepressionAngle", "TargetElevation", "Elevation",
]


def _depression_angle(path: Path) -> str:
    """Phoenix 헤더에서 부각(depression angle)을 정수 도(°) 문자열로 추출.
    확장자(.000/.001 등 일련번호)는 앙각이 아니므로 헤더를 사용.
    실패 시 'unknown'."""
    try:
        hdr = read_mstar_header(path)
    except Exception:
        return "unknown"
    for key in _DEPRESSION_KEYS:
        if key in hdr:
            try:
                return str(int(round(float(hdr[key]))))
            except ValueError:
                continue
    return "unknown"


def _data_available() -> bool:
    return any(d.exists() and any(d.rglob("*")) for d in MSTAR_RAW_DIRS)


def run_gradcam_analysis(
    model_name: str = "smpl",
    checkpoint: Path | None = None,
    k: int = 5,
    n_samples: int = 10,
    save_dir: Path = RESULTS_DIR / "gradcam",
    log_scale: bool = False,
    dyn_range_db: float = 60.0,
    cam_from_last: int = 1,
) -> list[dict]:
    """
    우리 팀 개선 #3: Grad-CAM 히트맵과 공간 산란점 좌표 IoU 비교.
    모델이 실제 산란점 위치를 보고 분류하는지 물리적 신뢰도 검증.

    ⚠️ log_scale/dyn_range_db는 반드시 모델 학습 시 전처리와 일치시켜야 함.
    precomputed(MATLAB) aug 모델은 log_scale=True, dyn_range_db=60으로 학습됨.
    불일치 시 모델이 OOD 입력을 받아 Grad-CAM이 배경으로 흩어짐(IoU 급락).

    cam_from_last: CAM 추출 층 선택. SMPL은 마지막 conv가 8×8로 거칠어 얇은 타겟을
    못 짚음 → 기본 1(뒤에서 두 번째 conv, 16×16)로 해상도 2배 국소화 개선.
    """
    num_classes = len(CLASSES)
    model = get_model(model_name, num_classes)

    # checkpoint 미지정 시 노트북 Cell 7a가 저장한 PH 보간 모델을 기본 사용
    if checkpoint is None:
        default_ckpt = RESULTS_DIR / f"{model_name}_ph_aug.pth"
        if default_ckpt.exists():
            checkpoint = default_ckpt

    if checkpoint is not None and checkpoint.exists():
        try:
            model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
            print(f"Loaded checkpoint: {checkpoint}")
        except RuntimeError as e:
            print(f"⚠️  체크포인트 구조 불일치({checkpoint}): {e}\n"
                  "    랜덤 가중치 모델 사용 (결과 무의미). Cell 7a를 다시 실행해 재학습하세요.")
            model = get_model(model_name, num_classes)  # 부분 로드된 가중치 폐기
    else:
        print("⚠️  학습된 체크포인트 없음 — 랜덤 가중치 모델 사용 (결과 무의미). "
              "먼저 notebooks/colab_template.ipynb Cell 7a를 실행해 모델을 저장하세요.")
    model.eval()

    raw_files: list[Path] = []
    if _data_available():
        files_by_class = _collect_raw_files(MSTAR_RAW_DIRS, CLASSES)
        for files in files_by_class.values():
            raw_files.extend(files)
    if not raw_files:
        print("[Grad-CAM] No raw files found — skipping.")
        return []

    save_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for p in raw_files[:n_samples]:
        try:
            amp = read_mstar_raw(p)
        except Exception as e:
            print(f"  Skip {p.name}: {e}")
            continue

        # 모델이 64×64로 학습됐으므로 Grad-CAM도 동일 입력 크기 사용
        image_lin = amplitude_to_tensor(amp, center_crop=EXP_B_INPUT)  # [1,64,64] 선형진폭
        # 산란점은 선형진폭(실제 물리 밝은점)에서 추출 — dB 압축 전이 물리적으로 정확
        amp128 = image_lin.squeeze(0).numpy()       # CAM과 동일 좌표계 (64×64)
        spatial_centers = extract_spatial_scattering_centers(amp128, k=k)
        ph_map = extract_scattering_centers(amp128, k=k)

        # CAM 입력은 모델 학습 전처리와 일치시킴 (precomputed aug 모델은 log-amp 60dB로 학습)
        # 좌표계는 동일하고 픽셀값 스케일만 바뀌므로 산란점 좌표와 IoU 비교 유효
        if log_scale:
            from augmentation.precomputed_aug import _normalize_amplitude
            arr = _normalize_amplitude(amp128[None, ...], log_scale=True, dyn_range_db=dyn_range_db)
            image_t = torch.from_numpy(arr[0]).unsqueeze(0)     # [1,64,64]
        else:
            image_t = image_lin

        gcam = GradCAM(model, from_last=cam_from_last)
        cam = gcam(image_t.unsqueeze(0))            # [64,64]
        gcam.remove()

        h, w = amp128.shape                          # cam과 일치
        scatter_mask = centers_to_mask(spatial_centers, h, w, radius=6)
        # CAM을 상위 분위수로 이진화 (고정 0.5는 peaked CAM에서 거의 비어 IoU=0 유발)
        cam_thr = float(np.percentile(cam, 80))      # 상위 20% attention 영역
        cam_bin = (cam > cam_thr).astype(np.float32)
        iou = compute_iou(scatter_mask, cam_bin, threshold=0.5)
        # 보조(주) 지표: 산란점 위치에서의 평균 CAM 활성도 (0~1, 강건함)
        center_vals = [
            float(cam[int(np.clip(cy, 0, h - 1)), int(np.clip(cx, 0, w - 1))])
            for cy, cx in spatial_centers
        ]
        cam_coverage = float(np.mean(center_vals)) if center_vals else 0.0
        cam_baseline = float(cam.mean())             # 전체 평균 CAM (기준선)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(amp, cmap="gray"); axes[0].set_title("Amplitude"); axes[0].axis("off")
        axes[1].imshow(ph_map.spectrum, cmap="hot")
        for cy, cx in ph_map.scattering_centers:
            axes[1].plot(cx, cy, "c+", markersize=8, markeredgewidth=2)
        axes[1].set_title(f"PH Spectrum (top-{k})"); axes[1].axis("off")
        axes[2].imshow(amp128, cmap="gray")          # cam과 동일 128×128
        axes[2].imshow(cam, cmap="jet", alpha=0.5)
        # 산란점 위치 표시 (CAM이 여기 몰리는지 눈으로 확인)
        for cy, cx in spatial_centers:
            axes[2].plot(cx, cy, "w+", markersize=10, markeredgewidth=2)
        axes[2].set_title(f"Grad-CAM (coverage={cam_coverage:.2f}, IoU={iou:.2f})")
        axes[2].axis("off")
        plt.tight_layout()
        fig.savefig(save_dir / f"{p.stem}_analysis.png", dpi=150)
        plt.close(fig)

        rec = {"file": p.name, "iou": iou, "cam_coverage": cam_coverage,
               "cam_baseline": cam_baseline, "n_centers": len(spatial_centers)}
        print(f"  {p.name:40s}  coverage={cam_coverage:.3f} (기준선 {cam_baseline:.3f})  IoU={iou:.3f}")
        records.append(rec)

    if records:
        mean_cov = np.mean([r["cam_coverage"] for r in records])
        mean_base = np.mean([r["cam_baseline"] for r in records])
        mean_iou = np.mean([r["iou"] for r in records])
        ratio = mean_cov / mean_base if mean_base > 0 else 0.0
        print(f"\n── Grad-CAM 산란점 정합 요약 ──")
        print(f"  산란점 CAM 활성도(coverage) = {mean_cov:.3f}")
        print(f"  전체 평균 CAM(기준선)        = {mean_base:.3f}")
        print(f"  비율 = {ratio:.2f}×  ({'>1 → 모델이 산란점에 더 집중' if ratio > 1 else '≤1 → 산란점 밖에 집중'})")
        print(f"  평균 IoU = {mean_iou:.3f}")
    return records


def run_xai_analysis(
    model_name: str = "smpl",
    checkpoint: Path | None = None,
    k: int = 5,
    n_samples: int = 10,
    save_dir: Path = RESULTS_DIR / "xai",
    log_scale: bool = True,
    dyn_range_db: float = 60.0,
    methods: tuple[str, ...] = ("occlusion", "smoothgrad_ig"),
) -> list[dict]:
    """
    우리 팀 개선 #3 (확장): 픽셀 단위 XAI로 산란점 정합 검증.

    Grad-CAM은 SMPL 마지막 conv(8×8) 해상도에 묶여 얇은 점 산란체를 못 짚음.
    Occlusion(인과적) + SmoothGrad-IG(공리적, 픽셀 단위)는 입력 공간에서 직접
    어트리뷰션을 계산 → 해상도 천장 없이 산란점과 IoU/coverage 비교.

    반환: 각 이미지·방법별 {file, method, iou, coverage, baseline}.
    """
    import torch as _torch
    from augmentation.precomputed_aug import _normalize_amplitude

    num_classes = len(CLASSES)
    model = get_model(model_name, num_classes)
    if checkpoint is None:
        default_ckpt = RESULTS_DIR / f"{model_name}_ph_aug.pth"
        if default_ckpt.exists():
            checkpoint = default_ckpt
    if checkpoint is not None and checkpoint.exists():
        try:
            model.load_state_dict(_torch.load(checkpoint, map_location="cpu"))
            print(f"Loaded checkpoint: {checkpoint}")
        except RuntimeError as e:
            print(f"⚠️  체크포인트 구조 불일치({checkpoint}): {e}\n"
                  "    랜덤 가중치 모델 사용 (결과 무의미). Cell 7a를 다시 실행해 재학습하세요.")
            model = get_model(model_name, num_classes)  # 부분 로드된 가중치 폐기
    else:
        print("⚠️  학습된 체크포인트 없음 — 결과 무의미. "
              "먼저 notebooks/colab_template.ipynb Cell 7a를 실행해 모델을 저장하세요.")
    model.eval()

    raw_files: list[Path] = []
    if _data_available():
        for files in _collect_raw_files(MSTAR_RAW_DIRS, CLASSES).values():
            raw_files.extend(files)
    if not raw_files:
        print("[XAI] No raw files found — skipping.")
        return []

    save_dir.mkdir(parents=True, exist_ok=True)
    method_fns = {
        "occlusion": lambda m, t: occlusion_sensitivity(m, t.unsqueeze(0), patch=8, stride=4),
        "smoothgrad_ig": lambda m, t: smoothgrad_ig(m, t.unsqueeze(0), steps=24, n_noise=6),
    }
    titles = {"occlusion": "Occlusion", "smoothgrad_ig": "SmoothGrad-IG"}
    records: list[dict] = []

    for p in raw_files[:n_samples]:
        try:
            amp = read_mstar_raw(p)
        except Exception as e:
            print(f"  Skip {p.name}: {e}")
            continue

        image_lin = amplitude_to_tensor(amp, center_crop=EXP_B_INPUT)  # [1,64,64] 선형
        amp64 = image_lin.squeeze(0).numpy()
        spatial_centers = extract_spatial_scattering_centers(amp64, k=k)
        # 모델 입력은 학습 전처리(log-amp 60dB)와 일치
        if log_scale:
            arr = _normalize_amplitude(amp64[None, ...], log_scale=True, dyn_range_db=dyn_range_db)
            image_t = _torch.from_numpy(arr[0]).unsqueeze(0)
        else:
            image_t = image_lin

        h, w = amp64.shape
        scatter_mask = centers_to_mask(spatial_centers, h, w, radius=6)

        maps = {name: method_fns[name](model, image_t) for name in methods}

        n_panels = 1 + len(methods)
        fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))
        axes[0].imshow(amp64, cmap="gray")
        for cy, cx in spatial_centers:
            axes[0].plot(cx, cy, "c+", markersize=10, markeredgewidth=2)
        axes[0].set_title("Amplitude + scatter pts"); axes[0].axis("off")

        for ax, name in zip(axes[1:], methods):
            attr = maps[name]
            thr = float(np.percentile(attr, 80))
            attr_bin = (attr > thr).astype(np.float32)
            iou_v = compute_iou(scatter_mask, attr_bin, threshold=0.5)
            center_vals = [float(attr[int(np.clip(cy, 0, h - 1)), int(np.clip(cx, 0, w - 1))])
                           for cy, cx in spatial_centers]
            coverage = float(np.mean(center_vals)) if center_vals else 0.0
            baseline = float(attr.mean())
            ax.imshow(amp64, cmap="gray")
            ax.imshow(attr, cmap="jet", alpha=0.5)
            for cy, cx in spatial_centers:
                ax.plot(cx, cy, "w+", markersize=10, markeredgewidth=2)
            ax.set_title(f"{titles[name]} (cov={coverage:.2f}, IoU={iou_v:.2f})")
            ax.axis("off")
            records.append({"file": p.name, "method": name, "iou": iou_v,
                            "coverage": coverage, "baseline": baseline})
            print(f"  {p.name:20s} {titles[name]:14s} coverage={coverage:.3f} "
                  f"(기준선 {baseline:.3f}) IoU={iou_v:.3f}")
        plt.tight_layout()
        fig.savefig(save_dir / f"{p.stem}_xai.png", dpi=150)
        plt.close(fig)

    if records:
        print(f"\n── 픽셀 단위 XAI 산란점 정합 요약 ──")
        for name in methods:
            rs = [r for r in records if r["method"] == name]
            mcov = np.mean([r["coverage"] for r in rs])
            mbase = np.mean([r["baseline"] for r in rs])
            miou = np.mean([r["iou"] for r in rs])
            ratio = mcov / mbase if mbase > 0 else 0.0
            print(f"  [{titles[name]}] coverage={mcov:.3f} / 기준선={mbase:.3f} "
                  f"/ 비율={ratio:.2f}× / 평균 IoU={miou:.3f}")
    return records


if __name__ == "__main__":
    # 학습은 notebooks/colab_template.ipynb Cell 7a(MATLAB .mat + precomputed_aug.py)에서 수행.
    # 이 CLI는 학습된 체크포인트에 대한 해석가능성 분석만 제공한다.
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="smpl", choices=["smpl", "resnet18"])
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--xai", action="store_true",
                        help="Occlusion/SmoothGrad-IG 픽셀 XAI 실행 (기본은 Grad-CAM)")
    args = parser.parse_args()

    if args.xai:
        run_xai_analysis(model_name=args.model, checkpoint=args.checkpoint)
    else:
        run_gradcam_analysis(model_name=args.model, checkpoint=args.checkpoint)
