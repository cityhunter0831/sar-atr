"""
Exp B — Phase History Interpolation Augmentation (논문 Section 2.1, Table 3)

⚠️ 현재 구현은 논문과 불일치 — 재설계 대상. 정확한 설계는 docs/PAPER_SPEC.md 참조.

논문 Table 3 (원문): 5클래스(2S1,BMP2,BTR70,T72,ZSU23), train El17°/test El15°,
  baseline = 클래스당 24~32장(총 136장, few-shot) → SMPL/AT 56.6%
  PH 보간 증강(Aug1, 1088장) → 96.4%. 입력 64×64 crop. 손실 AT(ε=2)/LSM.
  핵심: baseline이 낮은 이유는 부각 차이가 아니라 "학습 샘플이 136장뿐"이기 때문.

현재 코드: 7개 Mixed Targets 클래스 전체(2049장) 학습 → 98% (few-shot 아님).
  → 논문 재현하려면 5클래스 + 136장 baseline + azimuth 보간 + 64×64로 수정 필요.

우리 팀 개선 #3 (Grad-CAM 산란점 일치도 검증): run_gradcam_analysis() 참고
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
    read_azimuth,
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
# few-shot baseline 샘플 수 (논문 Table 2, MSTAR-R): 총 136장
FEWSHOT_COUNTS = {"2S1": 32, "BMP2": 24, "BTR70": 24, "T72": 24, "ZSU23": 32}
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

    # 논문 SOC = train 17° / test 15°. 둘 다 존재하면 명시적으로 사용 (Table 2/3).
    if dep_counter.get("17", 0) > 0 and dep_counter.get("15", 0) > 0:
        train_dep, test_dep = "17", "15"
    else:
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
    """MSTAR raw binary 파일 → 이미지 Dataset. crop_size 지정 시 center-crop (논문 64×64)."""

    def __init__(self, files_by_class: dict[str, list[Path]], class_names: list[str],
                 crop_size: int | None = None):
        self._class_names = class_names
        self._crop = crop_size
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
        t = amplitude_to_tensor(amp, center_crop=self._crop)
        return SARSample(image=t, label=label, meta={"source": str(path), "class_name": self._class_names[label]})

    @property
    def class_names(self) -> list[str]:
        return self._class_names


# ─── PH 보간 증강 데이터셋 ────────────────────────────────────────────────────

class PHAugmentedDataset(SARDataset):
    """
    few-shot 원본 + PH 보간 합성 이미지 Dataset (논문 MSTAR-Aug).

    논문 방식(T4): 각 원본 이미지를 **방위각(azimuth) 이웃**과 PH 도메인 보간해 합성.
    실이미지 Az=θ+Δ에서 Az=θ 합성 → 방위각을 조밀하게 채움.
    azimuth를 못 읽으면 파일 순서 이웃으로 폴백.

    lazy loading: __init__에서 경로/메타 튜플만 저장, I/O는 __getitem__에서 수행.
    """

    def __init__(
        self,
        files_by_class: dict[str, list[Path]],
        class_names: list[str],
        n_alphas: int = 5,
        seed: int = 0,
        crop_size: int | None = None,
    ):
        self._class_names = class_names
        self._crop = crop_size
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
                # 방위각 기준 정렬 후 이웃 pairing (azimuth 이웃 보간)
                az = {p: read_azimuth(p) for p in files}
                if all(not np.isnan(v) for v in az.values()):
                    ordered = sorted(files, key=lambda p: az[p])
                else:
                    ordered = files  # 폴백: 파일 순서
                pairs = list(zip(ordered, ordered[1:] + ordered[:1]))
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
        return SARSample(image=amplitude_to_tensor(amp, center_crop=self._crop), label=label,
                         meta={"class_name": self._class_names[label]})

    @property
    def class_names(self) -> list[str]:
        return self._class_names


# ─── Main runners ─────────────────────────────────────────────────────────────

def _data_available() -> bool:
    return any(d.exists() and any(d.rglob("*")) for d in MSTAR_RAW_DIRS)


def _fewshot_subsample(
    train_files: dict[str, list[Path]], counts: dict[str, int], seed: int
) -> dict[str, list[Path]]:
    """논문 MSTAR-R: 클래스별로 counts만큼만 무작위 선택 (few-shot baseline, 총 136장)."""
    rng = random.Random(seed)
    out: dict[str, list[Path]] = {}
    for cls, files in train_files.items():
        shuffled = files[:]
        rng.shuffle(shuffled)
        out[cls] = shuffled[: counts.get(cls, len(shuffled))]
    return out


def run(
    model_name: str = "smpl",
    epochs: int = 60,
    n_interp: int = 5,
    seed: int = 0,
    save_dir: Path = RESULTS_DIR,
    use_mock: bool = False,
    loss_type: str = "at",          # 논문 headline: AT(ε=2). "lsm"도 가능
    few_shot: bool = True,          # 논문 Table 3: few-shot(136장) baseline
    input_size: int = EXP_B_INPUT,  # 논문: 64×64 center-crop
) -> dict:
    """
    논문 Table 3 재현: few-shot 원본(MSTAR-R) vs PH 보간 증강(MSTAR-Aug) 정확도 비교.
    - few_shot=True: baseline을 클래스당 24~32장(총 136장)으로 제한 (논문 핵심)
    - loss_type: 'at'(ε=2, headline 56.6→96.4) 또는 'lsm'(61.3→97.6)
    - input_size: 64 center-crop
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    num_classes = len(CLASSES)

    if use_mock or not _data_available():
        if not use_mock:
            print(f"[Exp B] MSTAR raw not found at {MSTAR_RAW_DIRS} — using mock data.")
        n = 30 * num_classes if few_shot else 60 * num_classes
        base_ds = MockSARDataset(n=n, num_classes=num_classes, seed=seed)
        aug_ds  = MockSARDataset(n=n * (n_interp + 1), num_classes=num_classes, seed=seed + 1)
        test_ds = MockSARDataset(n=20 * num_classes, num_classes=num_classes, seed=seed + 2)
    else:
        print(f"[Exp B] Loading MSTAR raw (Targets+Mixed) for 5 classes {CLASSES} ...")
        files_by_class = _collect_raw_files(MSTAR_RAW_DIRS, CLASSES)
        for cls, files in files_by_class.items():
            print(f"  {cls}: {len(files)} files")

        # train El17° / test El15° (헤더 부각 기반)
        train_files, test_files = _split_train_test(files_by_class, seed)

        # 논문 MSTAR-R: few-shot 제한 (클래스당 24~32장)
        if few_shot:
            train_files = _fewshot_subsample(train_files, FEWSHOT_COUNTS, seed)
        n_train_total = sum(len(v) for v in train_files.values())
        n_test_total = sum(len(v) for v in test_files.values())
        print(f"  → train {n_train_total}개 (few_shot={few_shot}) / test {n_test_total}개, "
              f"입력 {input_size}×{input_size}, loss={loss_type}")
        for cls in CLASSES:
            print(f"     {cls}: train={len(train_files[cls])} test={len(test_files[cls])}")

        base_ds = MSTARRawDataset(train_files, CLASSES, crop_size=input_size)
        aug_ds  = PHAugmentedDataset(train_files, CLASSES, n_alphas=n_interp,
                                     seed=seed, crop_size=input_size)
        test_ds = MSTARRawDataset(test_files, CLASSES, crop_size=input_size)

    results = {"loss_type": loss_type, "few_shot": few_shot}

    cfg = TrainConfig(model_name=model_name, num_classes=num_classes, epochs=epochs,
                      seed=seed, loss_type=loss_type)

    # Condition 1: MSTAR-R (few-shot 원본만)
    print("\n── Condition 1: few-shot 원본만 학습 (MSTAR-R) ────────────────────")
    model1 = get_model(model_name, num_classes)
    model1, r1 = train_model(model1, base_ds, test_ds, cfg)
    results["no_aug_acc"] = r1.accuracy
    print(f"  → 테스트 정확도: {r1.accuracy * 100:.1f}%")

    # Condition 2: MSTAR-Aug (PH 보간 증강)
    print("\n── Condition 2: PH 보간 증강 학습 (MSTAR-Aug) ─────────────────────")
    model2 = get_model(model_name, num_classes)
    model2, r2 = train_model(model2, aug_ds, test_ds, cfg)
    results["ph_aug_acc"] = r2.accuracy
    print(f"  → 테스트 정확도: {r2.accuracy * 100:.1f}%")

    ref_base = "56.6" if loss_type == "at" else "61.3"
    ref_aug = "96.4" if loss_type == "at" else "97.6"
    print("\n── Exp B 결과 (Table 3 재현, SMPL/%s) ──────────────────────────" % loss_type.upper())
    print(f"  MSTAR-R(few-shot): {results['no_aug_acc'] * 100:.1f}%  (논문 SMPL: {ref_base}%)")
    print(f"  MSTAR-Aug(PH증강): {results['ph_aug_acc'] * 100:.1f}%  (논문 SMPL: {ref_aug}%)")

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
        model.load_state_dict(_torch.load(checkpoint, map_location="cpu"))
        print(f"Loaded checkpoint: {checkpoint}")
    else:
        print("⚠️  학습된 체크포인트 없음 — 결과 무의미. 먼저 run()으로 모델을 저장하세요.")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="smpl", choices=["smpl", "resnet18"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--n-interp", type=int, default=5)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--gradcam", action="store_true", help="Grad-CAM 분석 실행")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--loss", default="at", choices=["at", "lsm"], help="논문: at(56.6→96.4) / lsm(61.3→97.6)")
    parser.add_argument("--full-data", action="store_true", help="few-shot 해제(전체 데이터, 논문 아님)")
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
            loss_type=args.loss,
            few_shot=not args.full_data,
        )
