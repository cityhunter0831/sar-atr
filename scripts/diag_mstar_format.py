"""MSTAR 파일 포맷 진단 스크립트."""
from pathlib import Path
from collections import Counter

check_dirs = [
    Path("data/mstar/MSTAR_PUBLIC_MIXED_TARGETS_CD1"),
    Path("data/mstar/MSTAR_PUBLIC_MIXED_TARGETS_CD2"),
]

CRLF = b"\r\n"
PHOENIX = b"PhoenixHeaderVer"

for d in check_dirs:
    if not d.exists():
        print(d.name, "not found, skipping")
        continue
    raw_files = [
        p for p in d.rglob("*")
        if p.is_file() and p.suffix.lstrip(".").isdigit() and len(p.suffix) >= 3
    ]
    print(d.name, ":", len(raw_files), "raw files")
    ext_count = Counter(p.suffix for p in raw_files)
    print("  Extensions:", dict(ext_count.most_common(10)))
    for p in raw_files[:3]:
        with open(p, "rb") as fh:
            chunk = fh.read(300)
        has_phoenix = PHOENIX in chunk
        has_crlf = CRLF in chunk[:200]
        print(" ", p.name, "  phoenix =", has_phoenix, "  crlf =", has_crlf)
        print("    first 80B:", repr(chunk[:80]))
