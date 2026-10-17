#!/usr/bin/env python3
"""Bring our own phone clips to the jury's format and write labels eval.py can score.

Input: a folder with clips renamed to the shot-log names (OWN-01a.mov, OWN-01b.mp4, ...; any video extension) and the
shot log as CSV with the header  clip,scene,label,second,type,note  (label: violation | no_violation; second and type
empty for clean clips). Output in --out: <clip>.mov at the training clips' size (long side 848 px, portrait kept,
H.264, 30 fps, no audio), plus labels.csv (all clips), labels_a.csv and labels_b.csv (per phone, by the last letter).

    python3 tools/prepare_own.py --src var/own_raw --log var/own_raw/shotlog.csv --out var/own
    python3 tools/eval.py --local pipeline --videos var/own --labels var/own/labels_a.csv --out var/runs/own-a
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cashaudit import kb  # noqa: E402

FIELDS = ["clip", "file_in_zip", "scene", "label", "label_ts_sec", "type_id", "type_name", "note"]
VIDEO_EXT = {".mov", ".mp4", ".m4v", ".mkv", ".avi", ".3gp"}


def norm_label(x: str) -> str:
    x = (x or "").strip().lower().replace("-", "_").replace(" ", "_")
    if x in ("violation", "v", "yes", "1", "нарушение"):
        return "violation"
    if x in ("no_violation", "clean", "none", "no", "0", "n", "нет"):
        return "no_violation"
    raise ValueError(f"label must be violation or no_violation, got {x!r}")


def convert(src: Path, dst: Path) -> None:
    # long side 848 px like the training clips (464x848), aspect and orientation kept; H.264 so every decoder reads it
    vf = "scale='if(gt(iw,ih),848,-2)':'if(gt(iw,ih),-2,848)',fps=30"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vf", vf, "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "23", "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", str(dst)]
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder with OWN-xxa/b clips")
    ap.add_argument("--log", required=True, help="shot log CSV: clip,scene,label,second,type,note")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = {p.stem: p for p in src.iterdir() if p.suffix.lower() in VIDEO_EXT}
    with open(args.log, encoding="utf-8-sig") as f:
        log = [{k.strip().lower(): (v or "").strip() for k, v in r.items() if k} for r in csv.DictReader(f)]

    rows, problems = [], []
    for r in log:
        clip = r.get("clip", "")
        if not clip:
            continue
        try:
            label = norm_label(r.get("label", ""))
        except ValueError as e:
            problems.append(f"{clip}: {e}")
            continue
        tid = (r.get("type") or "").strip().upper()
        sec = r.get("second", "")
        if label == "violation" and (not tid or not sec):
            problems.append(f"{clip}: violation needs type and second")
            continue
        if tid and tid not in kb.type_ids():
            problems.append(f"{clip}: unknown type {tid}")
            continue
        if clip not in files:
            problems.append(f"{clip}: no video file named {clip}.* in {src}")
            continue
        dst = out / f"{clip}.mov"
        if not dst.exists():
            convert(files[clip], dst)
        rows.append({"clip": clip, "file_in_zip": dst.name, "scene": (r.get("scene") or "").upper(), "label": label,
                     "label_ts_sec": sec if label == "violation" else "", "type_id": tid if label == "violation" else "",
                     "type_name": kb.type_name(tid) if label == "violation" else "", "note": r.get("note", "")})

    def write(name, subset):
        with open(out / name, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(subset)
        v = sum(1 for r in subset if r["label"] == "violation")
        print(f"{name}: {len(subset)} clips ({v} violation, {len(subset) - v} clean)")

    write("labels.csv", rows)
    for phone in ("a", "b"):
        write(f"labels_{phone}.csv", [r for r in rows if r["clip"].lower().endswith(phone)])
    unused = sorted(set(files) - {r["clip"] for r in rows})
    if unused:
        print("videos not in the shot log (skipped):", ", ".join(unused))
    for p in problems:
        print("PROBLEM", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
