"""Exp B 파일 수집 진단 — Targets 패키지에서 BMP2/BTR70/T72를 왜 못 찾는지 확인.

Colab에서:  !python scripts/diag_exp_b_files.py
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.exp_b_ph_scattering import (  # noqa: E402
    MSTAR_RAW_DIRS, CLASSES, CLASS_ALIASES, _collect_raw_files,
)

print("=" * 70)
print("코드 버전 확인: _collect_raw_files 소스에 '_match_class' 있으면 최신")
import inspect  # noqa: E402
src = inspect.getsource(_collect_raw_files)
print("  최신 코드?", "_match_class" in src, "(_match_class 포함 여부)")
print("=" * 70)

# 1) Targets 패키지 루트 존재 확인
targets_root = MSTAR_RAW_DIRS[0]
print(f"\n[1] Targets 패키지 경로: {targets_root}")
print(f"    존재? {targets_root.exists()}")
if not targets_root.exists():
    # 혹시 다른 이름으로 풀렸는지 data/mstar 아래를 훑음
    mstar = Path("data/mstar")
    print(f"    → data/mstar 아래 실제 폴더들:")
    if mstar.exists():
        for d in sorted(mstar.iterdir()):
            print(f"        {d.name}")

# 2) BMP2/BTR70/T72 폴더가 실제로 어디 있는지 (이름 부분일치, 대소문자 무시)
print("\n[2] BMP2/BTR70/T72/2S1/ZSU 폴더를 전체 트리에서 탐색:")
for root in MSTAR_RAW_DIRS:
    if not root.exists():
        print(f"    (없음) {root}")
        continue
    print(f"    ── {root.name} ──")
    hits = {}
    for p in root.rglob("*"):
        if p.is_dir():
            nl = p.name.lower()
            for key in ["bmp2", "btr70", "t72", "2s1", "zsu"]:
                if nl.startswith(key):
                    hits.setdefault(key, []).append(p)
    for key, dirs in hits.items():
        print(f"      '{key}*' 폴더 {len(dirs)}개. 예:")
        for d in dirs[:2]:
            # 상대경로로 표시
            print(f"          {d.relative_to(root)}")
            # 그 폴더 안 파일 샘플
            files = [x for x in d.rglob("*") if x.is_file()][:2]
            for f in files:
                print(f"            └ 파일: {f.relative_to(d)}  (확장자 {f.suffix})")
    if not hits:
        print("      ⚠️  BMP2/BTR70/T72/2S1/ZSU 로 시작하는 폴더 없음")
        # 대신 17_DEG 아래 실제 구조 보여주기
        for deg in root.rglob("17_DEG"):
            print(f"      17_DEG 발견: {deg.relative_to(root)}")
            for child in sorted(deg.iterdir())[:8]:
                print(f"          내부: {child.name}  ({'dir' if child.is_dir() else 'file'})")
            break

# 3) 현재 _collect_raw_files 결과
print("\n[3] _collect_raw_files 실제 수집 결과:")
fbc = _collect_raw_files(MSTAR_RAW_DIRS, CLASSES)
for cls in CLASSES:
    print(f"    {cls}: {len(fbc[cls])}개")
