"""
Exp C — Contrast Balance + Optuna (박승준, Figure 1 재현)

Reproduces and improves the cross-elevation accuracy drop shown in Figure 1:
    Train El=17° → Test El=17°: ~97.2%   (baseline)
    Train El=17° → Test El=30°: ~65.3%   (degradation without contrast balance)
    Train El=17° → Test El=30° + ContrastBalance: target ≥88.5%

Data required:
    data/mstar/mixed_targets/  (MSTAR/IU Mixed Targets package)
    Classes: 2S1, BRDM2, ZSU23-4  at elevations 17° and 30°

Falls back to MockSARDataset when data is absent.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import optuna
import torch

from augmentation.contrast_balance import (
    ContrastBalance, ContrastJitter,
    make_optuna_objective, make_contrast_optuna_objective)
from core.evaluate import evaluate
from core.interfaces import EvalResult, SARDataset, SARSample, TrainConfig
from core.mock_data import MockSARDataset
from core.models import get_model
from core.train import train_model

RESULTS_DIR = Path("results/exp_c")
DATA_ROOT = Path("data/mstar/MSTAR_PUBLIC_MIXED_TARGETS_CD2")
SAMPLE_ROOT = Path("data/sample/png_images/decibel")
SAMPLE_MAT_ROOT = Path("data/sample/mat_files")  # 복소 원본 — sample_mat.py 전용, PNG와 다른 폴더
FIGURE1_CLASSES = ["2S1", "BRDM_2", "ZSU_23_4"]  # 실제 폴더명 (언더스코어)
# SAMPLE dataset 클래스 (BMP2, BTR70, T72 등 MSTAR와 동일)
# 논문 Figure 10 캡션 순서와 동일(#0~#9 = 2S1,BMP2,BTR70,M1,M2,M35,M548,M60,T72,ZSU23).
# exp_d_ood.py의 ALL_CLASSES와도 반드시 이 순서로 일치시킬 것.
SAMPLE_CLASSES = ["2s1", "bmp2", "btr70", "m1", "m2", "m35", "m548", "m60", "t72", "zsu23"]


# ─── Dataset helpers ──────────────────────────────────────────────────────────

class ElevationFilteredDataset(SARDataset):
    """
    Wraps a directory-based dataset and filters by elevation angle.

    Expected structure:
        <root>/<elevation>/<class>/*.png    e.g. mixed_targets/017/2S1/HB03344.017
    """

    def __init__(
        self,
        root: Path,
        elevation: int,
        class_names: list[str],
        augmentation=None,
    ):
        self._class_names = class_names
        self._augmentation = augmentation
        self._samples: list[tuple[Path, int]] = []

        el_str = str(elevation).zfill(3)
        el_dir = root / el_str
        if not el_dir.exists():
            # Try alternative naming: 'el17', '17deg', etc.
            for d in sorted(root.iterdir()):
                if d.is_dir() and str(elevation) in d.name:
                    el_dir = d
                    break

        for idx, cls in enumerate(class_names):
            # 직접 경로 또는 COL<n>/SCENE<n>/<cls> 중간 경로 모두 지원
            for p in el_dir.rglob(f"{cls}/*"):
                if p.is_file():
                    self._samples.append((p, idx))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> SARSample:
        from PIL import Image
        path, label = self._samples[idx]
        try:
            img = Image.open(path).convert("L").resize((128, 128))
            arr = np.array(img, dtype=np.float32) / 255.0
        except Exception:
            # MSTAR raw fallback (크기 제각각 → 128×128 리사이즈 필수)
            from augmentation.ph_extraction import read_mstar_raw, amplitude_to_tensor
            arr_raw = read_mstar_raw(path)
            arr = amplitude_to_tensor(arr_raw).squeeze(0).numpy()  # [128,128]

        t = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)
        meta = {"class_name": self._class_names[label], "elevation": 0, "source": str(path)}
        if self._augmentation is not None:
            t = self._augmentation(t, meta)
        return SARSample(image=t, label=label, meta=meta)

    @property
    def class_names(self) -> list[str]:
        return self._class_names


class GaussianNoiseAug:
    """논문 Section 4.3 'Gaus' 레시피 — 학습 시 입력에 가우시안 노이즈를 얹는
    train-only 정규화. AugmentedWrapper와 함께 쓴다(__getitem__마다 새로 샘플링돼
    epoch마다 다른 노이즈가 적용됨)."""

    def __init__(self, std: float):
        self.std = std

    def __call__(self, image: torch.Tensor, meta: dict) -> torch.Tensor:
        if self.std <= 0:
            return image
        return (image + torch.randn_like(image) * self.std).clamp(0.0, 1.0)


class AugmentedWrapper(SARDataset):
    """Apply a ContrastBalance augmentation on top of any SARDataset."""

    def __init__(self, ds: SARDataset, aug: ContrastBalance):
        self._ds = ds
        self._aug = aug

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> SARSample:
        s = self._ds[idx]
        s.image = self._aug(s.image, s.meta)
        return s

    @property
    def class_names(self) -> list[str]:
        return self._ds.class_names


class IndexSubset(SARDataset):
    """SARDataset의 부분집합 (인덱스 목록). test → val/test 분할용."""

    def __init__(self, ds: SARDataset, indices: list[int]):
        self._ds, self._indices = ds, list(indices)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> SARSample:
        return self._ds[self._indices[idx]]

    @property
    def class_names(self) -> list[str]:
        return self._ds.class_names


def _split_val_test(ds: SARDataset, val_frac: float = 0.2, seed: int = 0
                    ) -> tuple[SARDataset, SARDataset]:
    """real 평가셋을 val/test로 무작위 분할 (Optuna는 val로만 튜닝 → test 누수 방지)."""
    n = len(ds)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_frac))
    val_idx, test_idx = perm[:n_val].tolist(), perm[n_val:].tolist()
    return IndexSubset(ds, val_idx), IndexSubset(ds, test_idx)


class RepeatAugmentedDataset(SARDataset):
    """
    원본을 `times`배로 확장하고 각 접근마다 augmentation을 적용.
    논문의 "이미지당 3개 대비 레벨 생성(806×3=2418)" 재현용 — times=3, aug=ContrastJitter.
    무작위 대비이므로 접근마다 다른 대비 버전이 나와 3배 증강 효과.
    """

    def __init__(self, ds: SARDataset, aug, times: int = 3):
        self._ds, self._aug, self._times = ds, aug, times

    def __len__(self) -> int:
        return len(self._ds) * self._times

    def __getitem__(self, idx: int) -> SARSample:
        s = self._ds[idx % len(self._ds)]
        s.image = self._aug(s.image, s.meta)
        return s

    @property
    def class_names(self) -> list[str]:
        return self._ds.class_names


# ─── SAMPLE dataset loader (v3 주 데이터) ─────────────────────────────────────

class SampleDataset(SARDataset):
    """
    SAMPLE dataset (benjaminlewis-afrl/SAMPLE_dataset_public) loader.

    구조: data/sample/png_images/decibel/real/<class_name>/*.png
              data/sample/png_images/decibel/synth/<class_name>/*.png
    split: "real" | "synth"
    """

    def __init__(self, root: Path, split: str, class_names: list[str]):
        assert split in ("real", "synth"), f"split must be 'real' or 'synth', got {split!r}"
        self._class_names = class_names
        self._split = split
        self._samples: list[tuple[Path, int]] = []

        for idx, cls in enumerate(class_names):
            cls_dir = root / split / cls
            if not cls_dir.exists():
                # 대소문자 차이 허용
                split_dir = root / split
                if split_dir.exists():
                    for d in sorted(split_dir.iterdir()):
                        if d.name.lower() == cls.lower():
                            cls_dir = d
                            break
            if not cls_dir.exists():
                continue
            for p in sorted(cls_dir.iterdir()):
                if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    self._samples.append((p, idx))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> SARSample:
        from PIL import Image
        path, label = self._samples[idx]
        img = Image.open(path).convert("RGB").convert("L").resize((128, 128))
        arr = np.array(img, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).unsqueeze(0)
        meta = {
            "class_name": self._class_names[label],
            "split": self._split,
            "source": str(path),
        }
        return SARSample(image=t, label=label, meta=meta)

    @property
    def class_names(self) -> list[str]:
        return self._class_names


def _sample_available() -> bool:
    return SAMPLE_ROOT.exists() and any(SAMPLE_ROOT.iterdir())


def load_sample(
    class_names: list[str] = SAMPLE_CLASSES,
) -> tuple[SARDataset, SARDataset]:
    """
    Returns (train_synth, test_real).
    SAMPLE의 synth로 훈련 → real로 테스트 = 논문 Figure 1 원본 방법.
    Falls back to mock if data is absent.
    """
    if not _sample_available():
        print(f"[Exp C] SAMPLE data not found at {SAMPLE_ROOT} — using MockSARDataset.")
        print("  → git clone https://github.com/benjaminlewis-afrl/SAMPLE_dataset_public")
        return (
            MockSARDataset(n=200, num_classes=len(class_names), seed=0),
            MockSARDataset(n=60,  num_classes=len(class_names), seed=1),
        )

    train_ds = SampleDataset(SAMPLE_ROOT, "synth", class_names)
    test_ds  = SampleDataset(SAMPLE_ROOT, "real",  class_names)
    print(f"[Exp C] SAMPLE loaded: train(synthetic)={len(train_ds)}, test(measured)={len(test_ds)}")

    if len(train_ds) == 0 or len(test_ds) == 0:
        print(f"[Exp C] 데이터 0개 — SAMPLE_ROOT={SAMPLE_ROOT} 경로 확인 필요. MockSARDataset으로 대체.")
        return (
            MockSARDataset(n=200, num_classes=len(class_names), seed=0),
            MockSARDataset(n=60,  num_classes=len(class_names), seed=1),
        )

    return train_ds, test_ds


class _ConcatSARDataset(SARDataset):
    """여러 SARDataset을 이어붙임. class_names는 모두 동일하다고 가정(첫 데이터셋 기준)."""

    def __init__(self, datasets: list[SARDataset]):
        self._datasets = list(datasets)
        self._lens = [len(d) for d in self._datasets]
        self._class_names = self._datasets[0].class_names

    def __len__(self) -> int:
        return sum(self._lens)

    def __getitem__(self, idx: int) -> SARSample:
        for ds, n in zip(self._datasets, self._lens):
            if idx < n:
                return ds[idx]
            idx -= n
        raise IndexError(idx)

    @property
    def class_names(self) -> list[str]:
        return self._class_names


def split_real_by_k(
    class_names: list[str] = SAMPLE_CLASSES,
    k: float = 0.1,
    seed: int = 0,
    n_synth_per_class: dict[str, int] | None = None,
) -> tuple[SARDataset, SARDataset]:
    """
    논문 Section 4.3 K 정의(학습셋 중 실측 비율)에 맞춰 real 풀을 클래스별로
    train(K 비율)/test(나머지)로 나눈다. Exp C(Ori/Aug 조건 공용) + Exp D가
    "같은 real 분할"을 쓸 수 있도록 분리해 노출 — synth 쪽(PNG vs 복소 mat
    렌더링)이 조건마다 달라도 real 분할은 재현 가능하게 동일 seed로 고정.

    n_synth_per_class 생략 시 SampleDataset(synth)의 클래스별 장수를 사용
    (기본 K=0 baseline 규모 기준 K/(1-K) 비율 계산).

    Returns (train_real_ds, test_real_ds). k<=0이면 (빈 데이터셋, real_ds 전체).
    """
    real_ds = SampleDataset(SAMPLE_ROOT, "real", class_names)
    if k <= 0:
        return IndexSubset(real_ds, []), real_ds

    if n_synth_per_class is None:
        synth_ds = SampleDataset(SAMPLE_ROOT, "synth", class_names)
        n_synth_per_class = {}
        for _, label in synth_ds._samples:
            cls = class_names[label]
            n_synth_per_class[cls] = n_synth_per_class.get(cls, 0) + 1

    rng = np.random.default_rng(seed)
    real_by_class: dict[int, list[int]] = {}
    for i, (_, label) in enumerate(real_ds._samples):
        real_by_class.setdefault(label, []).append(i)

    train_real_idx: list[int] = []
    test_real_idx: list[int] = []
    for label, idxs in real_by_class.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        n_synth_cls = n_synth_per_class.get(class_names[label], 0)
        n_real_train = min(len(idxs), max(1, round(k / (1 - k) * n_synth_cls))) if n_synth_cls > 0 else 0
        train_real_idx.extend(idxs[:n_real_train])
        test_real_idx.extend(idxs[n_real_train:])

    return IndexSubset(real_ds, train_real_idx), IndexSubset(real_ds, test_real_idx)


def make_k_mixed_datasets(
    class_names: list[str] = SAMPLE_CLASSES,
    k: float = 0.1,
    seed: int = 0,
    use_contrast: bool = False,
) -> tuple[SARDataset, SARDataset]:
    """
    논문 Section 4.3 K 정의 재현: K = 학습셋 중 실측(measured) 샘플의 비율
    (K=0 → 100% synthetic, K=1 → 100% measured). Exp C/D 공용.

    synth은 전량(K=0 baseline과 동일 규모) 사용. real은 클래스별로 K/(1-K) 비율만큼
    무작위로 뽑아 학습에 섞고, 학습에 쓰지 않은 나머지 real만 평가(test_id_ds)에
    쓴다 — 같은 real 이미지가 학습·평가 양쪽에 들어가는 누수를 방지.

    use_contrast=True면 synth을 PNG 대신 복소 mat 기반 3단계 고정 대비
    (SampleContrastDataset, augmentation/sample_mat.py)로 로드한다 — 논문
    Section 3가 "Standard 시나리오(Exp D)도 대비 증강을 쓴다"고 명시한 부분 재현.
    mat_files가 없으면 예외가 나므로, 없을 가능성이 있는 호출부는 try/except로 감쌀 것.

    K=0이면 (synth_ds, real_ds) 그대로 반환해 기존 K=0 경로와 동일하게 동작.
    """
    if use_contrast:
        from augmentation.sample_mat import SampleContrastDataset
        synth_ds: SARDataset = SampleContrastDataset(SAMPLE_MAT_ROOT, "synth", class_names)
        if len(synth_ds) == 0:
            raise FileNotFoundError(
                f"use_contrast=True인데 {SAMPLE_MAT_ROOT}/synth에서 .mat을 못 찾음 — "
                "mat_files 미다운로드 가능성. 조용히 빈 데이터셋으로 진행하지 않고 예외를 던짐."
            )
    else:
        synth_ds = SampleDataset(SAMPLE_ROOT, "synth", class_names)

    if k <= 0:
        real_ds = SampleDataset(SAMPLE_ROOT, "real", class_names)
        return synth_ds, real_ds

    # real/synth 비율 계산은 항상 '실제 synth 원본 장수'(PNG 기준) — use_contrast=True여도
    # 3배 뻥튀기된 개수가 아니라 원본 806장 규모를 기준으로 K를 정의해야 논문 정의와 맞음.
    n_synth_per_class = None
    if use_contrast:
        plain_synth = SampleDataset(SAMPLE_ROOT, "synth", class_names)
        n_synth_per_class = {}
        for _, label in plain_synth._samples:
            cls = class_names[label]
            n_synth_per_class[cls] = n_synth_per_class.get(cls, 0) + 1

    train_real_ds, test_id_ds = split_real_by_k(class_names, k, seed, n_synth_per_class)
    train_mixed = _ConcatSARDataset([synth_ds, train_real_ds])
    print(f"  K={k} (contrast={use_contrast}): synth {len(synth_ds)}장 + real(train) {len(train_real_ds)}장 "
          f"= 학습 {len(train_mixed)}장, 평가용 real(test) {len(test_id_ds)}장")
    return train_mixed, test_id_ds


# ─── Data loading (MSTAR El ablation — 보조) ──────────────────────────────────

def _data_available() -> bool:
    return DATA_ROOT.exists() and any(DATA_ROOT.iterdir())


def load_el17_el30(
    class_names: list[str] = FIGURE1_CLASSES,
) -> tuple[SARDataset, SARDataset, SARDataset]:
    """
    Returns (train_el17, test_el17, test_el30).
    Falls back to mock if real data is absent.
    """
    if not _data_available():
        print(f"[Exp C] Real data not found at {DATA_ROOT} — using MockSARDataset.")
        n = 150
        return (
            MockSARDataset(n=n, num_classes=len(class_names), seed=0),
            MockSARDataset(n=50, num_classes=len(class_names), seed=1),
            MockSARDataset(n=50, num_classes=len(class_names), seed=2),
        )

    train_el17 = ElevationFilteredDataset(DATA_ROOT, elevation=17, class_names=class_names)
    test_el17 = ElevationFilteredDataset(DATA_ROOT, elevation=17, class_names=class_names)
    test_el30 = ElevationFilteredDataset(DATA_ROOT, elevation=30, class_names=class_names)
    return train_el17, test_el17, test_el30


# ─── Runner ───────────────────────────────────────────────────────────────────

def run(
    model_name: str = "resnet18",
    n_optuna_trials: int = 20,
    epochs_full: int = 60,
    epochs_trial: int = 10,
    seed: int = 0,
    save_dir: Path = RESULTS_DIR,
    class_names: list[str] = SAMPLE_CLASSES,
    paper_contrast: float = 0.5,
    paper_levels: int = 3,
) -> dict:
    """
    Exp C: SAMPLE dataset (synthetic → measured), 재현 + 개선 2단 구조.

    Step 1  baseline           : synth 학습(증강 없음) → real 평가 (논문 SAMPLE-Ori, RN18 91.9%)
    Step 2  논문 대비증강 재현   : ColorJitter(contrast=0.5) ×3 로 synth 증강 → real 평가
                                  (논문 SAMPLE-Aug, RN18 94.5%). 논문 실제 방법 그대로.
    Step 3  우리 개선 #2        : 대비 강도·레벨을 Optuna로 자동 탐색 → 최적값으로 재학습
                                  (매직넘버 0.5·3레벨 고정 → 근거 있는 최적값). real 평가.

    핵심: 대비 증강은 **train-only 데이터 증강**(무작위 대비), 평가는 실측 real 원본 그대로.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    n_classes = len(class_names)

    train_ds, test_full = load_sample(class_names)
    # real 평가셋을 val/test로 분할 — Optuna는 real_val로만 튜닝, 최종 수치는 못 본 real_test
    real_val, real_test = _split_val_test(test_full, val_frac=0.2, seed=seed)
    print(f"real 평가셋 분할: val={len(real_val)} / test={len(real_test)} (Optuna 누수 방지)")

    base_config = TrainConfig(
        model_name=model_name,
        num_classes=n_classes,
        epochs=epochs_full,
        seed=seed,
    )

    # ── Step 1: baseline (synthetic → measured, 증강 없음) ──
    print("Step 1: Train synthetic (no aug) → Test measured")
    model_base = get_model(model_name, n_classes)
    model_base, _ = train_model(model_base, train_ds, real_test, base_config)
    acc_no_aug = evaluate(model_base, real_test).accuracy * 100
    print(f"  synth → measured (no aug): {acc_no_aug:.1f}%  (논문 RN18 SAMPLE-Ori 91.9%)")

    # ── Step 2: 논문 대비증강 재현 — ColorJitter(contrast=0.5) ×3 (train-only) ──
    print(f"\nStep 2: 논문 재현 — ColorJitter(contrast={paper_contrast}) ×{paper_levels}")
    paper_aug = ContrastJitter(strength=paper_contrast)
    paper_train = RepeatAugmentedDataset(train_ds, paper_aug, times=paper_levels)
    model_paper = get_model(model_name, n_classes)
    model_paper, _ = train_model(model_paper, paper_train, real_test, base_config)
    acc_paper = evaluate(model_paper, real_test).accuracy * 100
    print(f"  synth+대비증강(논문 0.5×{paper_levels}) → measured: {acc_paper:.1f}%  "
          f"(논문 RN18 SAMPLE-Aug 94.5%)")

    # ── Step 3: 우리 개선 — 대비 강도·레벨 Optuna 자동 탐색 (real_val로 선택) ──
    print(f"\nStep 3: 개선 #2 — Optuna 대비 자동탐색 ({n_optuna_trials} trials, "
          f"{epochs_trial} epochs each) — 선택은 real_val, 보고는 real_test")
    objective = make_contrast_optuna_objective(
        train_ds, real_val, base_config, n_epochs_trial=epochs_trial,
        repeat_dataset_fn=lambda ds, aug, t: RepeatAugmentedDataset(ds, aug, times=t))
    study = optuna.create_study(direction="maximize", study_name="exp_c_contrast_auto")
    study.optimize(objective, n_trials=n_optuna_trials, show_progress_bar=True)
    best = study.best_params
    print(f"\n  최적 대비 파라미터: {best}  (Optuna best val {study.best_value*100:.1f}%)")

    # 최적값으로 full-epoch 재학습 → 못 본 real_test로 최종 보고
    best_aug = ContrastJitter(strength=best["strength"])
    best_train = RepeatAugmentedDataset(train_ds, best_aug, times=best["levels"])
    model_auto = get_model(model_name, n_classes)
    model_auto, _ = train_model(model_auto, best_train, real_test, base_config)
    acc_auto = evaluate(model_auto, real_test).accuracy * 100
    print(f"  synth+대비증강(Optuna 최적) → measured: {acc_auto:.1f}%  (못 본 real_test)")

    results = {
        "model": model_name,
        "dataset": "SAMPLE",
        "acc_no_aug": acc_no_aug,
        "acc_paper_contrast": acc_paper,
        "acc_auto_contrast": acc_auto,
        "paper_params": {"contrast": paper_contrast, "levels": paper_levels},
        "best_params": best,
        "optuna_best_trial_acc": study.best_value * 100,
        "criterion_met": acc_auto >= 94.5,
    }
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    torch.save(model_auto.state_dict(), save_dir / f"{model_name}_contrast_best.pth")

    print("\n── Exp C 재현+개선 요약 (SAMPLE Table 6, K=0, RN18 91.9→94.5) ──")
    print(f"  ① no-aug            : {acc_no_aug:5.1f}%  (논문 Ori 91.9%)")
    print(f"  ② 논문 대비증강 재현 : {acc_paper:5.1f}%  (논문 Aug 94.5%)")
    print(f"  ③ Optuna 자동탐색   : {acc_auto:5.1f}%  (best {best})")
    print(f"  개선 효과(③−②): {acc_auto-acc_paper:+.1f}%p")
    print("──────────────────────────────────────────────────")
    return results


def run_el_ablation(
    model_name: str = "resnet18",
    n_optuna_trials: int = 20,
    epochs_full: int = 60,
    epochs_trial: int = 10,
    seed: int = 0,
    save_dir: Path = RESULTS_DIR / "el_ablation",
) -> dict:
    """
    보조 실험: MSTAR El=17° → El=30° 교차 elevation 정확도 하락 재현.
    MSTAR Mixed Targets 데이터 필요 (SDMS 승인 필요).
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    n_classes = len(FIGURE1_CLASSES)

    train_el17, test_el17, test_el30 = load_el17_el30()

    base_config = TrainConfig(
        model_name=model_name,
        num_classes=n_classes,
        epochs=epochs_full,
        seed=seed,
    )

    print("El Ablation Step 1: Train El17° → Test El17° (baseline)")
    model_base = get_model(model_name, n_classes)
    model_base, _ = train_model(model_base, train_el17, test_el17, base_config)
    acc_el17 = evaluate(model_base, test_el17).accuracy * 100
    acc_el30_no_aug = evaluate(model_base, test_el30).accuracy * 100
    print(f"  El17 → El17 : {acc_el17:.1f}%  (paper: ~97.2%)")
    print(f"  El17 → El30 : {acc_el30_no_aug:.1f}%  (paper: ~65.3%)")

    print(f"\nEl Ablation Step 2: Optuna ({n_optuna_trials} trials)")
    objective = make_optuna_objective(train_el17, test_el30, base_config, n_epochs_trial=epochs_trial)
    study = optuna.create_study(direction="maximize", study_name="exp_c_el_ablation")
    study.optimize(objective, n_trials=n_optuna_trials, show_progress_bar=True)

    best = study.best_params
    aug = ContrastBalance(
        clip_limit=best["clip_limit"],
        tile_grid_size=(best["tile_grid_size"], best["tile_grid_size"]),
        global_norm=best["global_norm"],
    )
    aug_train = AugmentedWrapper(train_el17, aug)
    aug_test_el30 = AugmentedWrapper(test_el30, aug)

    model_aug = get_model(model_name, n_classes)
    model_aug, _ = train_model(model_aug, aug_train, aug_test_el30, base_config)
    acc_el30_aug = evaluate(model_aug, aug_test_el30).accuracy * 100
    print(f"  El17 → El30 + ContrastBalance: {acc_el30_aug:.1f}%  (target: ≥88.5%)")

    results = {
        "model": model_name,
        "dataset": "MSTAR_mixed_targets",
        "acc_el17_baseline": acc_el17,
        "acc_el30_no_aug": acc_el30_no_aug,
        "acc_el30_with_aug": acc_el30_aug,
        "best_params": best,
        "criterion_met": acc_el30_aug >= 88.5,
    }
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    _print_figure1(acc_el30_no_aug, acc_el30_aug,
                   ref_base="65.3", ref_aug="88.5", target=88.5)  # MSTAR Figure 1
    return results


def _print_figure1(acc_no_aug: float, acc_aug: float,
                   ref_base: str = "91.9", ref_aug: str = "94.5", target: float = 94.5):
    """SAMPLE 주 실험(Table 6, K=0) 기준 출력.
    논문 RN18: SAMPLE(Ori) 91.9% → SAMPLE(Aug) 94.5% (K=0, 100% synthetic→measured).
    run_el_ablation은 MSTAR Figure 1 수치(65.3→88.5)를 넘겨 호출."""
    print("\n── Exp C 재현 (SAMPLE Table 6, K=0) ──────────────")
    print(f"  synthetic→measured (no aug):  {acc_no_aug:5.1f}%  (논문 RN18: {ref_base}%)")
    print(f"  synthetic→measured (대비증강): {acc_aug:5.1f}%  (논문 RN18: {ref_aug}%)")
    status = "달성 ✓" if acc_aug >= target else "미달 ✗"
    print(f"  목표({target}%): {status}")
    print("──────────────────────────────────────────────────")


def run_paper_faithful(
    model_names: list[str] = ("smpl_paper", "aconv_paper", "resnet18_paper", "heiligers_paper"),
    k_values: list[float] = (0.0, 0.05, 0.1),
    seeds: list[int] = (0, 1, 2),
    epochs: int = 60,
    class_names: list[str] = SAMPLE_CLASSES,
    save_dir: Path = RESULTS_DIR / "paper_faithful",
) -> dict:
    """
    논문 Table 6 재현 시도: 4모델 × K(0/0.05/0.1) × {Ori, Aug} × 여러 seed.

    Ori = synth(PNG) 전량 + real(train, K비율) — 대비 증강 없음.
    Aug = synth **복소 mat 3단계 고정 대비**(SampleContrastDataset, 806→2418) +
          동일 real(train, Ori와 같은 seed로 분할해 공정 비교) + Gaussian noise 주입.
    두 조건 모두 같은 real(test) 분할로 평가.

    이 함수는 기존 run()(ColorJitter+Optuna, 우리 팀 개선 #2)과 별개이며,
    "논문을 최대한 그대로 재현"하는 경로다 — 개선 #2는 그대로 유지.

    ⚠️ Colab에서만 검증 가능(SAMPLE mat_files 필요, 로컬 개발 환경엔 실데이터 없음).
    """
    from core.models_paper import get_paper_model, PAPER_RECIPE
    from augmentation.sample_mat import SampleContrastDataset

    save_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    n_classes = len(class_names)

    for model_name in model_names:
        recipe = PAPER_RECIPE[model_name]
        results[model_name] = {}
        for k in k_values:
            ori_accs: list[float] = []
            aug_accs: list[float] = []
            for seed in seeds:
                train_real_ds, test_id_ds = split_real_by_k(class_names, k, seed)

                synth_png_ds = SampleDataset(SAMPLE_ROOT, "synth", class_names)
                ori_train = _ConcatSARDataset([synth_png_ds, train_real_ds]) if k > 0 else synth_png_ds

                synth_mat_ds = SampleContrastDataset(SAMPLE_MAT_ROOT, "synth", class_names)
                aug_base = _ConcatSARDataset([synth_mat_ds, train_real_ds]) if k > 0 else synth_mat_ds
                aug_train = AugmentedWrapper(aug_base, GaussianNoiseAug(recipe["gaus"]))

                cfg = TrainConfig(model_name=model_name, num_classes=n_classes, epochs=epochs,
                                  seed=seed, label_smoothing=recipe["lblsm"])

                for train_ds, acc_list, tag in [(ori_train, ori_accs, "Ori"), (aug_train, aug_accs, "Aug")]:
                    model = get_paper_model(model_name, n_classes, drop_prob=recipe["drop"])
                    model, _ = train_model(model, train_ds, test_id_ds, cfg)
                    acc = evaluate(model, test_id_ds).accuracy * 100
                    acc_list.append(acc)
                    print(f"  [{model_name}] K={k} seed={seed} {tag}: {acc:.1f}%")

            def _stat(accs: list[float]) -> dict:
                return {"min": float(np.min(accs)), "max": float(np.max(accs)),
                       "avg": float(np.mean(accs)), "std": float(np.std(accs))}
            results[model_name][f"K={k}"] = {"Ori": _stat(ori_accs), "Aug": _stat(aug_accs)}

    with open(save_dir / "table6_results.json", "w") as f:
        json.dump(results, f, indent=2)
    _print_table6(results, k_values)
    return results


def _print_table6(results: dict, k_values: list[float]):
    print("\n── Exp C 논문 충실 재현 (Table 6 형식) ─────────────────────────")
    for model_name, by_k in results.items():
        print(f"\n[{model_name}]")
        for k in k_values:
            rec = by_k[f"K={k}"]
            o, a = rec["Ori"], rec["Aug"]
            print(f"  K={k:<5} Ori min={o['min']:.1f} max={o['max']:.1f} avg={o['avg']:.1f}±{o['std']:.1f}"
                  f"   Aug min={a['min']:.1f} max={a['max']:.1f} avg={a['avg']:.1f}±{a['std']:.1f}")
    print("─────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet18", choices=["smpl", "resnet18"])
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--epochs-trial", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--el-ablation", action="store_true",
                        help="Run MSTAR El=17→30 ablation instead of SAMPLE experiment")
    parser.add_argument("--paper-faithful", action="store_true",
                        help="논문 충실 재현(mat_files 3단계 대비 + 4모델 + K=0/0.05/0.1) 실행")
    args = parser.parse_args()
    if args.paper_faithful:
        run_paper_faithful(epochs=args.epochs)
        raise SystemExit(0)
    fn = run_el_ablation if args.el_ablation else run
    fn(
        model_name=args.model,
        n_optuna_trials=args.n_trials,
        epochs_full=args.epochs,
        epochs_trial=args.epochs_trial,
        seed=args.seed,
    )
