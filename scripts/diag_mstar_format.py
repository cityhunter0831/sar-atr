"""MSTAR 파일 포맷 진단 스크립트.

용도:
  1. 확장자 분포 확인 (확장자가 앙각 코드인지 일련번호인지 판별)
  2. Phoenix 헤더 전체 덤프 (앙각 필드명 확인 — DesiredDepression 등)
  3. 헤더의 부각(depression) 값 분포 집계 → cross-elevation split 가능 여부 판단
"""
from pathlib import Path
from collections import Counter
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from augmentation.ph_extraction import read_mstar_header  # noqa: E402

check_dirs = [
    Path("data/mstar/MSTAR_PUBLIC_MIXED_TARGETS_CD1"),
    Path("data/mstar/MSTAR_PUBLIC_MIXED_TARGETS_CD2"),
]

PHOENIX = b"PhoenixHeaderVer"

# 헤더에서 부각을 담을 수 있는 후보 필드명들
DEPRESSION_KEYS = [
    "DesiredDepression", "MeasuredDepression", "Depression",
    "DepressionAngle", "TargetElevation", "Elevation",
]


def _raw_files(d: Path) -> list[Path]:
    return [
        p for p in d.rglob("*")
        if p.is_file() and p.suffix.lstrip(".").isdigit() and len(p.suffix) >= 3
    ]


for d in check_dirs:
    if not d.exists():
        print(d.name, "not found, skipping")
        continue

    raw_files = _raw_files(d)
    print("=" * 70)
    print(d.name, ":", len(raw_files), "raw files")
    if not raw_files:
        continue

    ext_count = Counter(p.suffix for p in raw_files)
    print("  확장자 분포:", dict(ext_count.most_common(10)))

    # (1) 첫 파일 헤더 전체 덤프
    print("\n  ── 샘플 파일 헤더 전체 (앙각 필드 확인용) ──")
    sample = raw_files[0]
    try:
        hdr = read_mstar_header(sample)
        print(f"  파일: {sample.name}  (필드 {len(hdr)}개)")
        for k, v in hdr.items():
            print(f"    {k} = {v}")
    except Exception as e:
        print("  헤더 파싱 실패:", e)

    # (2) 부각 필드 탐지 + 분포 집계
    print("\n  ── 부각(depression) 분포 집계 (전체 파일) ──")
    found_key = None
    for key in DEPRESSION_KEYS:
        if key in hdr:
            found_key = key
            break

    if found_key is None:
        print(f"  ⚠️  후보 필드 {DEPRESSION_KEYS} 중 헤더에 없음.")
        print("      → 위 헤더 덤프에서 앙각으로 보이는 필드명을 확인하세요.")
    else:
        print(f"  사용 필드: {found_key}")
        dep_count: Counter = Counter()
        for p in raw_files:
            try:
                h = read_mstar_header(p)
                val = h.get(found_key, "?")
                try:
                    val = str(int(round(float(val))))
                except ValueError:
                    pass
                dep_count[val] += 1
            except Exception:
                dep_count["error"] += 1
        print("  부각 분포:", dict(dep_count.most_common(10)))
        if len(dep_count) >= 2:
            print("  ✅ cross-elevation split 가능 (2개 이상 부각 존재)")
        else:
            print("  ⚠️  부각이 1종류뿐 — cross-elevation 불가, stratified split 사용")
