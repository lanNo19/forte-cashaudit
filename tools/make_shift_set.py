#!/usr/bin/env python3
"""Make camera-shifted copies of the training clips to test robustness before the jury's new-angle video.

Each clip gets one geometric change (perspective tilt, zoom-crop, mirror or rotation) plus one photometric
change (brightness/contrast/colour, blur, heavier compression), chosen deterministically from the clip
name so every teammate generates the same set. Timestamps do not change, so train_annotations.csv still
applies: run tools/eval.py with --videos pointing at the output folder.

    python tools/make_shift_set.py --videos /path/open/videos --out /path/shifted
    python tools/eval.py --local pipeline --videos /path/shifted --out runs/shift

This is a proxy. Real footage from a different angle (FAQ 7 allows own recordings) is a better test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# W=464, H=848 in the training data; expressions use iw/ih so other sizes work too.
GEOMETRIC = {
    "tilt_left": "perspective=x0=0.08*W:y0=0.04*H:x1=W:y1=0:x2=0:y2=H:x3=0.92*W:y3=0.96*H:sense=destination,scale=iw:ih",
    "tilt_right": "perspective=x0=0:y0=0:x1=0.92*W:y1=0.05*H:x2=0.06*W:y2=H:x3=W:y3=0.95*H:sense=destination,scale=iw:ih",
    "zoom_crop": "crop=iw*0.82:ih*0.82:iw*0.09:ih*0.05,scale=464:848",
    "mirror": "hflip",
    "rotate": "rotate=7*PI/180:fillcolor=black,crop=iw*0.88:ih*0.88,scale=464:848",
    "wide": "scale=iw*0.8:ih*0.8,pad=464:848:(ow-iw)/2:(oh-ih)/2:color=gray",
}
PHOTOMETRIC = {
    "darker": "eq=brightness=-0.08:contrast=0.9",
    "warmer": "colorbalance=rs=0.08:bs=-0.06,eq=saturation=1.15",
    "cooler": "colorbalance=rs=-0.06:bs=0.08,eq=brightness=0.04",
    "soft": "gblur=sigma=1.2",
    "none": "null",
}


def pick(name: str, options: dict) -> str:
    keys = sorted(options)
    h = int(hashlib.md5((name + str(len(keys))).encode()).hexdigest(), 16)
    return keys[h % len(keys)]


def make(src: Path, dst: Path, geo: str, photo: str, crf: int) -> None:
    vf = f"{GEOMETRIC[geo]},{PHOTOMETRIC[photo]},format=yuv420p"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", vf, "-an", "-c:v", "libx264",
                    "-preset", "veryfast", "-crf", str(crf), "-movflags", "+faststart", str(dst)], check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clips", help="comma-separated subset")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(ROOT / "data" / "train_annotations.csv", encoding="utf-8")))
    if args.clips:
        keep = set(args.clips.split(","))
        rows = [r for r in rows if r["clip"] in keep]

    plan = []
    for r in rows:
        geo, photo = pick(r["clip"], GEOMETRIC), pick(r["clip"] + "p", PHOTOMETRIC)
        crf = 23 + int(hashlib.md5(r["clip"].encode()).hexdigest(), 16) % 8  # 23-30: some heavier compression
        plan.append((Path(args.videos) / r["file_in_zip"], out / r["file_in_zip"], geo, photo, crf, r["clip"]))

    def run(p):
        make(*p[:5])
        return p

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for src, dst, geo, photo, crf, clip in ex.map(run, plan):
            print(f"  {clip:6s} {geo:10s} {photo:7s} crf {crf}", flush=True)
    with open(out / "shift_plan.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "geometric", "photometric", "crf"])
        for src, dst, geo, photo, crf, clip in plan:
            w.writerow([clip, geo, photo, crf])
    print(f"{len(plan)} clips written to {out}")


if __name__ == "__main__":
    main()
