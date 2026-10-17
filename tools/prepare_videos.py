#!/usr/bin/env python3
"""Unpack the organisers' open.zip into a flat folder with the names eval.py expects (file_in_zip in the CSV).

The zip mixes Latin "X-" and Cyrillic "Х-" file names, and different unzip tools write the Cyrillic ones differently
("Х-3.mov" or "#U0425-3.mov"), so a plain `unzip` may not match data/train_annotations.csv. This extracts only the
.mov files under the CSV's names and checks every file against data/train_md5.json.

    python3 tools/prepare_videos.py --zip ~/open.zip --out var/videos
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def latin(name: str) -> str:
    return name.replace("Х", "X").replace("х", "x").replace("#U0425", "X")


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="open.zip from the organisers")
    ap.add_argument("--out", required=True, help="output folder, e.g. var/videos")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(ROOT / "data" / "train_annotations.csv", encoding="utf-8") as f:
        files = {r["clip"]: r["file_in_zip"] for r in csv.DictReader(f)}
    # the CSV keeps the names one unzip produced (some "#U0425-3.mov"); write each clip under exactly that name
    csv_name = {latin(name): name for name in files.values()}
    with zipfile.ZipFile(args.zip) as z:
        for info in z.infolist():
            if info.is_dir() or not info.filename.lower().endswith(".mov"):
                continue
            base = latin(Path(info.filename).name)
            dst = out / csv_name.get(base, base)
            if dst.exists() and dst.stat().st_size == info.file_size:
                continue
            with z.open(info) as src, open(dst, "wb") as f:
                while chunk := src.read(1 << 20):
                    f.write(chunk)

    expected = json.loads((ROOT / "data" / "train_md5.json").read_text(encoding="utf-8"))
    missing, wrong = [], []
    for clip, name in files.items():
        p = out / name
        if not p.exists():
            missing.append(name)
        elif clip in expected and md5(p) != expected[clip]:
            wrong.append(name)
    print(f"{len(files) - len(missing) - len(wrong)}/{len(files)} clips ready in {out}")
    if missing:
        print("missing:", ", ".join(missing))
    if wrong:
        print("checksum mismatch:", ", ".join(wrong))
    return 1 if missing or wrong else 0


if __name__ == "__main__":
    sys.exit(main())
