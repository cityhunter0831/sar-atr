"""
Exp D — OOD Detection (권승주)

Setup (논문 Section 4.4, 원문 페이지 이미지 대조로 확정):
    - In-distribution (ID): SAMPLE 10-class, K=0.1(실측 10%+합성 90% 혼합, make_k_mixed_datasets)
    - Holdout: J=1,2,3 — HLD1={M1}, HLD2={M35,M548}, HLD3={M1,M35,M548}
    - Outlier Exposure (OE): SAR-ship (+MiniSAR, 비공개라 미포함) — 학습 재료일 뿐 OOD 테스트 아님
    - Figure 9 재현의 실제 OOD 테스트 3종은 Holdout/MSTAR-O/MSTAR-P
      (SAR-ship을 far-OOD로 같이 보는 건 우리가 추가한 보조 검증, 논문 Figure 9엔 없음)

Comparison:
    ODIN  vs  Mahalanobis
    → AUROC and TNR@95TPR

Expected structure:
    data/sample/png_images/decibel/{real,synth}/<class>/   (SAMPLE, exp_c와 공유)
    data/sarship/                              (SAR-ship images, OE 학습 재료)
    results/exp_a/<model>_seed0_j<j>.pth       (Exp D 자체 체크포인트 — exp_a와 폴더만 공유)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from core.evaluate import evaluate, evaluate_ood
from core.interfaces import EvalResult, SARDataset, SARSample, TrainConfig
from core.mock_data import MockSARDataset
from core.models import get_model
from core.train import train_model

RESULTS_DIR = Path("results/exp_d")
SARSHIP_DIR = Path("data/sarship")

# T6 재설계: 논문은 ID=SAMPLE 10클래스, OE=SAR-ship(+MiniSAR 비공개), OOD=Holdout+MSTAR-O/P.
# SAMPLE 클래스 #0~#9 (논문 Figure 10 순서). exp_c의 SampleDataset·클래스 리스트를 그대로 재사용
# — 리스트를 이 파일에 따로 하드코딩하면 두 파일이 드리프트할 수 있어(과거에 실제로 순서가 어긋난 적 있음)
# 단일 소스(exp_c.SAMPLE_CLASSES)를 공유한다.
from experiments.exp_c_contrast_optuna import (  # noqa: E402
    SampleDataset, SAMPLE_ROOT, SAMPLE_CLASSES, make_k_mixed_datasets,
)
ALL_CLASSES = SAMPLE_CLASSES

# MSTAR-O (논문 Section 1/4.4.1, 원문 이미지 확정): BRDM2/BTR60/D7/T62/ZIL131 —
# Exp A/B 5클래스와 다른 Mixed Targets 나머지 5클래스. 공식 raw에서 MATLAB 없이도
# amplitude만 뽑으면 재현 가능(논문: "self-generated with Matlab based on the official
# MSTAR raw data" — 우리는 단순 진폭 렌더링만 하므로 완전히 동일하진 않을 수 있음).
from experiments.exp_b_ph_scattering import _collect_raw_files, MSTAR_RAW_DIRS  # noqa: E402
from augmentation.ph_extraction import read_mstar_raw, amplitude_to_tensor  # noqa: E402

MSTAR_O_CLASSES = ["BRDM2", "BTR60", "D7", "T62", "ZIL131"]
MSTAR_O_ALIASES = {
    "BRDM2": ["BRDM2", "BRDM_2", "brdm2", "brdm_2"],
    "BTR60": ["BTR60", "BTR_60", "btr60", "btr_60"],
    "D7":    ["D7", "d7"],
    "T62":   ["T62", "t62"],
    "ZIL131": ["ZIL131", "zil131"],
}

# Holdout combinations: J개 SAMPLE 클래스를 학습에서 제외 (near-OOD).
# 논문 Section 4.4.2 HLD1/2/3 (원문 페이지 이미지로 확정, Figure 11 본문
# "#5 (M35) and #6 (M548)... the only two trucks in the SAMPLE dataset"와
# Figure 10 캡션 "#0-9=2S1,BMP2,BTR70,M1,M2,M35,M548,M60,T72,ZSU23" 교차검증):
#   HLD1={#3:M1}, HLD2={#5:M35,#6:M548}, HLD3={#3:M1,#5:M35,#6:M548}
HOLDOUT_CONFIGS: dict[int, list[str]] = {
    1: ["m1"],
    2: ["m35", "m548"],
    3: ["m1", "m35", "m548"],
}


# ─── Dataset helpers ──────────────────────────────────────────────────────────

def _has_phoenix_header(path: Path) -> bool:
    """파일 앞 4KB만 읽어 Phoenix 헤더 존재 여부 확인."""
    try:
        with open(path, "rb") as f:
            chunk = f.read(4096)
        return b"PhoenixHeaderVer" in chunk
    except Exception:
        return False


class FolderDataset(SARDataset):
    """Generic image folder dataset compatible with SARDataset interface."""

    def __init__(self, roots: "Path | list[Path]", class_names: list[str], augmentation=None):
        self._class_names = class_names
        self._aug = augmentation
        self._samples: list[tuple[Path, int]] = []

        if isinstance(roots, Path):
            roots = [roots]

        for root in roots:
            for idx, cls in enumerate(class_names):
                # 직접 경로(Targets) 또는 COL/SCENE 중간 경로(Mixed) 모두 지원
                for p in root.rglob(f"{cls}/*"):
                    if not p.is_file():
                        continue
                    # 이미지 파일이거나 Phoenix 헤더가 있는 raw 파일만 포함
                    if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif"}:
                        self._samples.append((p, idx))
                    elif _has_phoenix_header(p):
                        self._samples.append((p, idx))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> SARSample:
        path, label = self._samples[idx]
        try:
            img = Image.open(path).convert("L").resize((128, 128))
            arr = np.array(img, dtype=np.float32) / 255.0
        except Exception:
            try:
                from augmentation.ph_extraction import read_mstar_raw
                import PIL.Image as _PILImage
                raw = read_mstar_raw(path)
                arr = (raw / (raw.max() + 1e-8)).astype(np.float32)
                arr = np.array(
                    _PILImage.fromarray((arr * 255).astype(np.uint8)).resize((128, 128)),
                    dtype=np.float32,
                ) / 255.0
            except Exception:
                arr = np.zeros((128, 128), dtype=np.float32)
        t = torch.from_numpy(arr).unsqueeze(0)
        meta = {"class_name": self._class_names[label], "source": str(path)}
        if self._aug is not None:
            t = self._aug(t, meta)
        return SARSample(image=t, label=label, meta=meta)

    @property
    def class_names(self) -> list[str]:
        return self._class_names


class SARShipDataset(SARDataset):
    """SAR-ship dataset used as OOD / Outlier Exposure set."""

    def __init__(self, root: Path, max_samples: int = 500):
        self._paths: list[Path] = []
        if root.exists():
            for p in sorted(root.rglob("*.png"))[:max_samples]:
                self._paths.append(p)
            for p in sorted(root.rglob("*.jpg"))[:max(0, max_samples - len(self._paths))]:
                self._paths.append(p)
        self._paths = self._paths[:max_samples]

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int) -> SARSample:
        p = self._paths[idx]
        img = Image.open(p).convert("L").resize((128, 128))
        arr = np.array(img, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).unsqueeze(0)
        return SARSample(image=t, label=-1, meta={"source": str(p), "class_name": "sarship"})

    @property
    def class_names(self) -> list[str]:
        return ["sarship"]


class MSTARODataset(SARDataset):
    """MSTAR-O — 논문 Figure 9의 far-OOD 테스트셋. BRDM2/BTR60/D7/T62/ZIL131
    5클래스를 Mixed Targets CD1/CD2 공식 raw에서 수집(MATLAB 없이 amplitude만 렌더링)."""

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
        t = amplitude_to_tensor(amp)  # 128×128, Exp D 다른 데이터셋과 동일 해상도
        return SARSample(image=t, label=label,
                         meta={"class_name": self._class_names[label], "source": str(path)})

    @property
    def class_names(self) -> list[str]:
        return self._class_names


def load_mstar_o() -> "SARDataset | None":
    """MSTAR-O 5클래스 raw 수집. 데이터 없으면 None(mock 대체 안 함 — sarship과 동일 원칙)."""
    files_by_class = _collect_raw_files(MSTAR_RAW_DIRS, MSTAR_O_CLASSES, aliases=MSTAR_O_ALIASES)
    counts = {c: len(v) for c, v in files_by_class.items()}
    total = sum(counts.values())
    if total == 0:
        print(f"[Exp D] MSTAR-O 없음 at {MSTAR_RAW_DIRS} — far-OOD(mstar_o) 비교를 건너뜁니다.")
        return None
    print(f"[Exp D] MSTAR-O 로드: {counts} (합계 {total}장, 논문 1290장)")
    return MSTARODataset(files_by_class, MSTAR_O_CLASSES)


class _SubsetWithNames(SARDataset):
    """torch random_split이 반환하는 Subset은 class_names가 없어서
    Mahalanobis(train_ds.class_names 참조)에서 터짐. 이를 노출하는 얇은 래퍼."""

    def __init__(self, subset, class_names: list[str]):
        self._subset = subset
        self._class_names = class_names

    def __len__(self) -> int:
        return len(self._subset)

    def __getitem__(self, idx: int) -> SARSample:
        return self._subset[idx]

    @property
    def class_names(self) -> list[str]:
        return self._class_names


# ─── Data loading ─────────────────────────────────────────────────────────────

def _data_available() -> bool:
    return SAMPLE_ROOT.exists() and any(SAMPLE_ROOT.iterdir())


def load_id_holdout(
    j: int = 1, class_names: list[str] = ALL_CLASSES, k: float = 0.1, seed: int = 0,
) -> tuple[SARDataset, SARDataset, SARDataset, "SARDataset | None"]:
    """
    T6 재설계: ID = SAMPLE 10클래스. Holdout = J개 SAMPLE 클래스(near-OOD),
    OE(=OOD 교차도메인) = SAR-ship.
    Returns (train_ds, test_id_ds, test_holdout_ds, oe_ds).
    SAMPLE 데이터 자체가 없으면 4개 모두 mock. SAR-ship만 없으면 oe_ds=None
    (mock으로 대체하지 않음 — 의미 없는 far-OOD 숫자가 결과에 섞이는 것을 방지).

    k: 논문 Figure 9/10/11 헤드라인 설정(K=0.1, 학습셋 중 실측 비율). K=0을 원하면
    k=0으로 호출 — 이 경우 기존과 동일하게 synth 전량/real 전량으로 학습/평가한다.
    """
    holdout = HOLDOUT_CONFIGS[j]
    known = [c for c in class_names if c not in holdout]

    if not _data_available():
        print(f"[Exp D] SAMPLE not found at {SAMPLE_ROOT} — using MockSARDataset.")
        return (
            MockSARDataset(n=300, num_classes=len(known), seed=0),
            MockSARDataset(n=100, num_classes=len(known), seed=1),
            MockSARDataset(n=50, num_classes=len(holdout), seed=2),
            MockSARDataset(n=100, num_classes=1, seed=3),  # mock OE
        )

    # ID 학습 = SAMPLE synth(known)+real 일부(K 비율 혼합), ID 테스트 = 학습에 안 쓴 나머지 real
    print(f"[Exp D] K={k} 혼합 데이터 구성 (known={len(known)}클래스)")
    train_ds, test_id_ds = make_k_mixed_datasets(known, k=k, seed=seed)
    # near-OOD = 학습 제외된 SAMPLE 클래스 (real)
    test_holdout_ds = SampleDataset(SAMPLE_ROOT, "real", holdout)

    sar_ship = SARShipDataset(SARSHIP_DIR)
    if len(sar_ship) == 0:
        print(f"[Exp D] SAR-ship not found at {SARSHIP_DIR} — far-OOD(sarship) 비교를 건너뜁니다.\n"
              "  (mock 데이터로 대체하면 진짜 결과처럼 보이는 의미 없는 숫자가 나오므로 skip)")
        oe_ds: SARDataset | None = None
    else:
        oe_ds = sar_ship  # far-OOD (cross-domain). 논문에선 OE 학습 재료로도 사용

    return train_ds, test_id_ds, test_holdout_ds, oe_ds


# ─── OOD experiment ───────────────────────────────────────────────────────────

def run_ood_experiment(
    model,
    train_ds: SARDataset,
    test_id_ds: SARDataset,
    holdout_ds: SARDataset,
    oe_ds: "SARDataset | None",
    j: int,
    mstar_o_ds: "SARDataset | None" = None,
) -> dict:
    """Run ODIN and Mahalanobis on holdout(near-OOD) + mstar_o(far-OOD, 논문 Figure 9
    재현) + sarship(far-OOD, 우리가 추가한 보조 검증 — 논문 Figure 9엔 없음).

    oe_ds/mstar_o_ds가 None(데이터 없음)이면 해당 비교는 건너뛰고 None으로 남긴다 —
    mock으로 대체해 의미 없는 숫자를 결과에 섞지 않기 위함.
    """
    model.eval()
    results = {"j": j}

    ood_targets = [("holdout", holdout_ds)]
    for name, ds, hint in [("mstar_o", mstar_o_ds, "Mixed Targets CD1/CD2 raw 확인"),
                            ("sarship", oe_ds, "data/sarship 확인")]:
        if ds is not None:
            ood_targets.append((name, ds))
        else:
            print(f"  [SKIP] {name} — 데이터 없음 ({hint})")
            for method in ["odin", "mahalanobis"]:
                results[f"{method}_{name}_auroc"] = None
                results[f"{method}_{name}_tnr95"] = None

    for ood_name, ood_ds in ood_targets:
        for method in ["odin", "mahalanobis"]:
            print(f"  [{method.upper()}] ID vs OOD={ood_name} ...", end=" ", flush=True)
            try:
                r: EvalResult = evaluate_ood(
                    model,
                    id_ds=test_id_ds,
                    ood_ds=ood_ds,
                    method=method,
                    train_ds=train_ds if method == "mahalanobis" else None,
                )
                results[f"{method}_{ood_name}_auroc"] = r.auroc
                results[f"{method}_{ood_name}_tnr95"] = r.tnr_at_95tpr
                print(f"AUROC={r.auroc:.3f}  TNR@95={r.tnr_at_95tpr:.3f}")
            except Exception as e:
                print(f"ERROR: {e}")
                results[f"{method}_{ood_name}_auroc"] = None
                results[f"{method}_{ood_name}_tnr95"] = None

    return results


# ─── Main runner ──────────────────────────────────────────────────────────────

def run(
    model_name: str = "smpl",
    checkpoint_dir: Path = Path("results/exp_a"),
    epochs: int = 60,
    seed: int = 0,
    j_list: list[int] = [1, 2, 3],
    save_dir: Path = RESULTS_DIR,
    k: float = 0.1,
) -> list[dict]:
    """k: 논문 헤드라인 설정(Figure 9/10/11) = 0.1. K=0(100% synthetic)으로 돌리려면 k=0."""
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    mstar_o_ds = load_mstar_o()  # J와 무관 — 한 번만 로드

    for j in j_list:
        print(f"\n── J={j} holdout classes: {HOLDOUT_CONFIGS[j]} ────────────────")
        holdout = HOLDOUT_CONFIGS[j]
        known = [c for c in ALL_CLASSES if c not in holdout]

        train_ds, test_id_ds, holdout_ds, oe_ds = load_id_holdout(j, k=k, seed=seed)

        # Load or train model
        ckpt = checkpoint_dir / f"{model_name}_seed{seed}_j{j}.pth"
        model = get_model(model_name, num_classes=len(known))

        loaded = False
        if ckpt.exists():
            try:
                model.load_state_dict(torch.load(ckpt, map_location="cpu"))
                print(f"  Loaded checkpoint: {ckpt}")
                loaded = True
            except RuntimeError:
                print(f"  Checkpoint 구조 불일치 — 처음부터 학습합니다: {ckpt}")
                model = get_model(model_name, num_classes=len(known))  # 부분 로드된 가중치 폐기, 새 모델로 재시작

        if not loaded:
            print(f"  Training {model_name} (J={j}, seed={seed}) ...")
            config = TrainConfig(
                model_name=model_name,
                num_classes=len(known),
                epochs=epochs,
                seed=seed,
            )
            model, train_result = train_model(model, train_ds, test_id_ds, config)
            print(f"  Train → Test ID accuracy: {train_result.accuracy * 100:.1f}%")
            torch.save(model.state_dict(), ckpt)

        rec = run_ood_experiment(model, train_ds, test_id_ds, holdout_ds, oe_ds, j,
                                 mstar_o_ds=mstar_o_ds)
        all_results.append(rec)

    with open(save_dir / "metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)

    _print_summary(all_results)
    return all_results


def _print_summary(results: list[dict]):
    print("\n── Exp D Summary ─────────────────────────────────────────────────────")
    header = f"{'J':<4} {'Method':<14} {'OOD':<10} {'AUROC':<8} {'TNR@95':<8}"
    print(header)
    print("-" * len(header))
    for rec in results:
        j = rec["j"]
        for method in ["odin", "mahalanobis"]:
            for ood in ["holdout", "mstar_o", "sarship"]:
                auroc = rec.get(f"{method}_{ood}_auroc")
                tnr = rec.get(f"{method}_{ood}_tnr95")
                auroc_s = f"{auroc:.3f}" if auroc is not None else "  N/A"
                tnr_s = f"{tnr:.3f}" if tnr is not None else "  N/A"
                print(f"{j:<4} {method:<14} {ood:<10} {auroc_s:<8} {tnr_s:<8}")
    print("──────────────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="smpl", choices=["smpl", "resnet18"])
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("results/exp_a"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--j", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--k", type=float, default=0.1, help="논문 헤드라인=0.1, K=0(100%% synthetic)이면 0")
    args = parser.parse_args()
    run(
        model_name=args.model,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        seed=args.seed,
        j_list=args.j,
        k=args.k,
    )
