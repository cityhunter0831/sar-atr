"""
Exp B — Phase History Interpolation Augmentation (논문 Section 2.1 재현)

논문 Table 3 재현:
  조건                | SMPL 5-class acc
  원본만 학습          | ~56.6%
  PH 보간 증강 추가   | ~96.6%

방법:
  1. Mixed Targets CD2에서 클래스별 이미지 pair 구성
  2. IFFT → PH 도메인 선형 보간 → FFT → 합성 이미지 생성
  3. 원본 + 합성 이미지로 SMPL 학습 후 테스트

우리 팀 개선 #3 (Grad-CAM 산란점 일치도 검증):
  run_gradcam_analysis() 참고

데이터: MSTAR Mixed Targets CD2 (Phoenix 헤더 확인됨)
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset

from augmentation.ph_extraction import (
    amplitude_to_tensor,
    extract_scattering_centers,
    extract_spatial_scattering_centers,
    interpolate_phase_history,
    read_mstar_complex,
    read_mstar_header,
    read_mstar_raw,
    visualize_scattering,
)
from core.interfaces import EvalResult, SARDataset, SARSample, TrainConfig
from core.mock_data import MockSARDataset
from core.models import get_model
from core.train import train_model
from gradcam.cam import GradCAM
from gradcam.scatter_overlap import centers_to_mask, iou as compute_iou

# 표준 MSTAR SOC(train 17° / test 15°) 재현을 위해 두 디스크를 모두 로드.
# 15°는 CD1에, 17°는 CD2에 있으므로 둘 다 필요. (진단 교차표로 확인됨)
MSTAR_RAW_DIRS = [
    Path("data/mstar/MSTAR_PUBLIC_MIXED_TARGETS_CD1"),
    Path("data/mstar/MSTAR_PUBLIC_MIXED_TARGETS_CD2"),
]
MSTAR_RAW_DIR = MSTAR_RAW_DIRS[1]  # 하위호환 (일부 코드에서 참조)
RESULTS_DIR = Path("results/exp_b")
CLASSES = ["2S1", "BRDM_2", "BTR_60", "D7", "T62", "ZIL131", "ZSU_23_4"]


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
    for root in roots:
        if not root.exists():
            continue
        for cls in classes:
            candidates: list[Path] = []
            for p in root.rglob(f"*{cls}*/*"):
                if p.is_file() and p.suffix.lstrip(".").isdigit() and len(p.suffix) >= 3:
                    candidates.append(p)
            if not candidates:
                for p in root.rglob("*"):
                    if p.is_file() and cls.lower() in str(p).lower():
                        if p.suffix.lstrip(".").isdigit():
                            candidates.append(p)
            result[cls].extend(p for p in candidates if _has_phoenix_header(p))
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


def _stratified_split(
    files_by_class: dict[str, list[Path]], seed: int, train_ratio: float = 0.8
) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    """클래스별 80/20 랜덤 분할. 각 클래스가 train/test 양쪽에 반드시 존재하도록 보장."""
    train_files: dict[str, list[Path]] = {}
    test_files: dict[str, list[Path]] = {}
    rng = random.Random(seed)
    for cls, files in files_by_class.items():
        shuffled = files[:]
        rng.shuffle(shuffled)
        if len(shuffled) >= 2:
            n_train = max(1, min(len(shuffled) - 1, int(len(shuffled) * train_ratio)))
        else:
            n_train = len(shuffled)  # 1개뿐이면 train에만
        train_files[cls] = shuffled[:n_train]
        test_files[cls] = shuffled[n_train:]
    return train_files, test_files


def _split_train_test(
    files_by_class: dict[str, list[Path]], seed: int
) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    """논문 Table 3 프로토콜: cross-depression split (헤더 `DesiredDepression` 기반).

    전역 상위 2개 부각을 train/test 부각으로 선택 (예: CD2 → 17°/30°).
    클래스별로 두 부각 파일을 각각 train/test에 배치.
    특정 클래스가 두 부각 중 하나만 가지면 → 그 클래스만 클래스 내 랜덤 80/20
    (전체 폴백 대신 나머지 클래스의 cross-depression 이점 유지).
    부각을 아예 못 읽으면 전체 stratified 80/20 폴백.
    """
    from collections import Counter

    # 파일당 부각 1회만 읽어 캐시 (중복 디스크 I/O 방지)
    dep_of: dict[Path, str] = {}
    for files in files_by_class.values():
        for p in files:
            dep_of[p] = _depression_angle(p)

    dep_counter: Counter = Counter(d for d in dep_of.values() if d != "unknown")
    top2 = [d for d, _ in dep_counter.most_common(2)]

    if len(top2) < 2:
        print("  ⚠️  헤더에서 2개 이상 부각을 못 찾음 — stratified 80/20 사용.")
        return _stratified_split(files_by_class, seed)

    train_dep, test_dep = top2[0], top2[1]
    print(f"  Cross-depression split: train={train_dep}°, test={test_dep}°")

    train_files: dict[str, list[Path]] = {}
    test_files: dict[str, list[Path]] = {}
    rng = random.Random(seed)

    for cls, files in files_by_class.items():
        tr = [p for p in files if dep_of[p] == train_dep]
        te = [p for p in files if dep_of[p] == test_dep]
        if tr and te:
            train_files[cls] = tr
            test_files[cls] = te
        else:
            # 이 클래스는 두 부각을 모두 갖지 않음 → 클래스 내 랜덤 80/20
            print(f"    ⚠️  {cls}: {train_dep}°={len(tr)} {test_dep}°={len(te)} "
                  f"— 클래스 내 랜덤 80/20 사용")
            shuffled = files[:]
            rng.shuffle(shuffled)
            if len(shuffled) >= 2:
                n_train = max(1, min(len(shuffled) - 1, int(len(shuffled) * 0.8)))
            else:
                n_train = len(shuffled)
            train_files[cls] = shuffled[:n_train]
            test_files[cls] = shuffled[n_train:]

    return train_files, test_files


class MSTARRawDataset(SARDataset):
    """MSTAR Mixed Targets raw binary 파일 → 이미지 Dataset."""

    def __init__(self, files_by_class: dict[str, list[Path]], class_names: list[str]):
        self._class_names = class_names
        self._samples: list[tuple[Path, int]] = []
        for idx, cls in enumerate(class_names):
            for p in files_by_class.get(cls, []):
                self._samples.append((p, idx))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> SARSample:
        path, label = self._samples[idx]
        try:
            amp = read_mstar_raw(path)
        except Exception:
            amp = np.zeros((128, 128), dtype=np.float32)
        t = amplitude_to_tensor(amp)
        return SARSample(image=t, label=label, meta={"source": str(path), "class_name": self._class_names[label]})

    @property
    def class_names(self) -> list[str]:
        return self._class_names


# ─── PH 보간 증강 데이터셋 ────────────────────────────────────────────────────

class PHAugmentedDataset(SARDataset):
    """
    원본 MSTAR 이미지 + PH 보간 합성 이미지를 합친 Dataset.

    각 클래스에서 이미지 pair를 구성하고, n_alphas개의 alpha 값으로
    합성 이미지를 생성해서 원본에 추가합니다.

    lazy loading: __init__에서 경로/메타 튜플만 저장, I/O는 __getitem__에서 수행.
    """

    def __init__(
        self,
        files_by_class: dict[str, list[Path]],
        class_names: list[str],
        n_alphas: int = 5,
        seed: int = 0,
    ):
        self._class_names = class_names
        # (kind, data, label) — kind: "original" | "synth"
        self._samples: list[tuple[str, object, int]] = []

        alphas = [i / (n_alphas + 1) for i in range(1, n_alphas + 1)]

        for idx, cls in enumerate(class_names):
            files = files_by_class.get(cls, [])
            if not files:
                continue

            for p in files:
                self._samples.append(("original", p, idx))

            if len(files) >= 2:
                pairs = list(zip(files, files[1:] + files[:1]))
                for p_a, p_b in pairs:
                    for alpha in alphas:
                        self._samples.append(("synth", (p_a, p_b, alpha), idx))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> SARSample:
        kind, data, label = self._samples[idx]
        if kind == "original":
            try:
                amp = read_mstar_raw(data)
            except Exception:
                amp = np.zeros((128, 128), dtype=np.float32)
        else:
            p_a, p_b, alpha = data
            try:
                amp = interpolate_phase_history(
                    read_mstar_complex(p_a), read_mstar_complex(p_b), alpha
                )
            except Exception:
                amp = np.zeros((128, 128), dtype=np.float32)
        return SARSample(image=amplitude_to_tensor(amp), label=label,
                         meta={"class_name": self._class_names[label]})

    @property
    def class_names(self) -> list[str]:
        return self._class_names


# ─── Main runners ─────────────────────────────────────────────────────────────

def _data_available() -> bool:
    return any(d.exists() and any(d.rglob("*")) for d in MSTAR_RAW_DIRS)


def run(
    model_name: str = "smpl",
    epochs: int = 60,
    n_interp: int = 5,
    seed: int = 0,
    save_dir: Path = RESULTS_DIR,
    use_mock: bool = False,
) -> dict:
    """
    논문 Table 3 재현: 원본 학습 vs PH 보간 증강 학습 정확도 비교.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    num_classes = len(CLASSES)

    if use_mock or not _data_available():
        if not use_mock:
            print(f"[Exp B] MSTAR Mixed Targets not found at {MSTAR_RAW_DIRS} — using mock data.")
        n = 60 * num_classes
        base_ds = MockSARDataset(n=n, num_classes=num_classes, seed=seed)
        aug_ds  = MockSARDataset(n=n * (n_interp + 1), num_classes=num_classes, seed=seed + 1)
        test_ds = MockSARDataset(n=20 * num_classes, num_classes=num_classes, seed=seed + 2)
    else:
        print(f"[Exp B] Loading MSTAR Mixed Targets from CD1+CD2 ...")
        files_by_class = _collect_raw_files(MSTAR_RAW_DIRS, CLASSES)
        for cls, files in files_by_class.items():
            print(f"  {cls}: {len(files)} files")

        # 논문 Table 3 프로토콜: cross-elevation split (헤더 부각 기반).
        # 확장자(.000/.001)는 앙각이 아니라 일련번호이므로 헤더에서 부각을 읽음.
        train_files, test_files = _split_train_test(files_by_class, seed)
        n_train_total = sum(len(v) for v in train_files.values())
        n_test_total = sum(len(v) for v in test_files.values())
        print(f"  → train {n_train_total}개 / test {n_test_total}개")
        for cls in CLASSES:
            print(f"     {cls}: train={len(train_files[cls])} test={len(test_files[cls])}")

        base_ds = MSTARRawDataset(train_files, CLASSES)
        aug_ds  = PHAugmentedDataset(train_files, CLASSES, n_alphas=n_interp, seed=seed)
        test_ds = MSTARRawDataset(test_files, CLASSES)

    results = {}

    # Condition 1: 원본만 학습
    print("\n── Condition 1: 원본 이미지만 학습 ────────────────────────────────")
    model1 = get_model(model_name, num_classes)
    cfg = TrainConfig(model_name=model_name, num_classes=num_classes, epochs=epochs, seed=seed)
    model1, r1 = train_model(model1, base_ds, test_ds, cfg)
    results["no_aug_acc"] = r1.accuracy
    print(f"  → 테스트 정확도: {r1.accuracy * 100:.1f}%")

    # Condition 2: PH 보간 증강 추가
    print("\n── Condition 2: PH 보간 증강 추가 학습 ────────────────────────────")
    model2 = get_model(model_name, num_classes)
    model2, r2 = train_model(model2, aug_ds, test_ds, cfg)
    results["ph_aug_acc"] = r2.accuracy
    print(f"  → 테스트 정확도: {r2.accuracy * 100:.1f}%")

    print("\n── Exp B 결과 (Table 3 재현) ───────────────────────────────────────")
    print(f"  원본만:      {results['no_aug_acc'] * 100:.1f}%  (논문: 56.6%)")
    print(f"  PH 보간 추가: {results['ph_aug_acc'] * 100:.1f}%  (논문: 96.6%)")

    import json
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Grad-CAM 분석(개선 #3)에서 재사용할 학습된 모델 저장
    if not use_mock:
        torch.save(model2.state_dict(), save_dir / f"{model_name}_ph_aug.pth")
        torch.save(model1.state_dict(), save_dir / f"{model_name}_no_aug.pth")
        print(f"  체크포인트 저장: {save_dir / f'{model_name}_ph_aug.pth'}")

    return results


def run_gradcam_analysis(
    model_name: str = "smpl",
    checkpoint: Path | None = None,
    k: int = 5,
    n_samples: int = 10,
    save_dir: Path = RESULTS_DIR / "gradcam",
) -> list[dict]:
    """
    우리 팀 개선 #3: Grad-CAM 히트맵과 공간 산란점 좌표 IoU 비교.
    모델이 실제 산란점 위치를 보고 분류하는지 물리적 신뢰도 검증.
    """
    num_classes = len(CLASSES)
    model = get_model(model_name, num_classes)

    # checkpoint 미지정 시 run()이 저장한 PH 보간 모델을 기본 사용
    if checkpoint is None:
        default_ckpt = RESULTS_DIR / f"{model_name}_ph_aug.pth"
        if default_ckpt.exists():
            checkpoint = default_ckpt

    if checkpoint is not None and checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        print(f"Loaded checkpoint: {checkpoint}")
    else:
        print("⚠️  학습된 체크포인트 없음 — 랜덤 가중치 모델 사용 (결과 무의미). "
              "먼저 run()을 실행해 모델을 저장하세요.")
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

        image_t = amplitude_to_tensor(amp)          # [1,128,128] (리사이즈됨)
        amp128 = image_t.squeeze(0).numpy()         # CAM과 동일 좌표계(128×128)
        # 산란점·마스크는 CAM과 같은 128×128에서 추출해야 IoU 계산 가능
        spatial_centers = extract_spatial_scattering_centers(amp128, k=k)
        ph_map = extract_scattering_centers(amp128, k=k)

        gcam = GradCAM(model)
        cam = gcam(image_t.unsqueeze(0))            # [128,128]
        gcam.remove()

        h, w = amp128.shape                          # 128, 128 — cam과 일치
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="smpl", choices=["smpl", "resnet18"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--n-interp", type=int, default=5)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--gradcam", action="store_true", help="Grad-CAM 분석 실행")
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()

    if args.gradcam:
        run_gradcam_analysis(
            model_name=args.model,
            checkpoint=args.checkpoint,
        )
    else:
        run(
            model_name=args.model,
            epochs=args.epochs,
            n_interp=args.n_interp,
            use_mock=args.mock,
        )
