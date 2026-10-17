#!/usr/bin/env python3
"""Cut our long block recordings into one clip per take, for both phones, lined up by the sync clap.

Input folder (e.g. var/own_long):
  A_a.mov, A_b.mov, B_a.mov, B_b.mov   one long recording per block (A, B) and phone (a, b); any video extension
  takes.csv                             take,block,label,type,start,key,end,note  (times = phone a's video, m:ss or s)

Each block starts with one loud clap. The script finds it in each phone's audio (the loudest moment in the first
`--search` seconds) and shifts phone b's times by the difference, so only phone a's times need writing down.
Override a detection with --clap A_b=12.4 if the printed clap time looks wrong.

Output (e.g. var/own_raw): OWN-01a.mov, OWN-01b.mov, ... and shotlog.csv for tools/prepare_own.py, where `second` is
the key moment counted from the start of the cut clip.

    python3 tools/cut_own.py --src var/own_long --out var/own_raw
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

VIDEO_EXT = {".mov", ".mp4", ".m4v", ".mkv", ".avi", ".3gp"}


def seconds(x: str) -> float:
    x = (x or "").strip()
    if not x:
        raise ValueError("empty time")
    parts = [float(p) for p in x.replace(",", ".").split(":")]
    s = 0.0
    for p in parts:
        s = s * 60 + p
    return s


def find_clap(path: Path, search: float) -> float:
    """Loudest 20 ms frame in the first `search` seconds of the audio."""
    sr = 8000
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-t", str(search), "-vn", "-ac", "1", "-ar", str(sr),
                          "-f", "s16le", "-"], capture_output=True, check=True).stdout
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if a.size < sr:
        raise RuntimeError(f"{path.name}: no audio track; give the clap time with --clap")
    hop = sr // 50
    n = a.size // hop
    energy = (a[: n * hop].reshape(n, hop) ** 2).mean(axis=1)
    return float(np.argmax(energy) * hop / sr)


def cut(src: Path, t0: float, t1: float, dst: Path) -> None:
    vf = "scale='if(gt(iw,ih),848,-2)':'if(gt(iw,ih),-2,848)',fps=30"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t0:.3f}", "-i", str(src), "-t", f"{t1 - t0:.3f}",
                    "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-an",
                    "-movflags", "+faststart", str(dst)], check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--search", type=float, default=120, help="seconds from the start to look for the clap")
    ap.add_argument("--clap", action="append", default=[], help="manual clap time, e.g. A_b=12.4 (repeatable)")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    videos = {p.stem.upper(): p for p in src.iterdir() if p.suffix.lower() in VIDEO_EXT}
    manual = {k.strip().upper(): float(v) for k, v in (c.split("=", 1) for c in args.clap)}
    with open(src / "takes.csv", encoding="utf-8-sig") as f:
        takes = [{k.strip().lower(): (v or "").strip() for k, v in r.items() if k} for r in csv.DictReader(f)]

    claps = {}
    for name in sorted(videos):
        if name in manual:
            claps[name] = manual[name]
        else:
            try:
                claps[name] = find_clap(videos[name], args.search)
            except Exception as e:
                print(f"PROBLEM {e}")
                continue
        print(f"clap {name}: {claps[name]:.2f} s{' (manual)' if name in manual else ''}")

    log, problems = [], []
    for r in takes:
        if not r.get("take"):
            continue
        try:
            take = int(float(r["take"]))
            block = r.get("block", "").upper()
            label = r.get("label", "").lower().replace("-", "_").replace(" ", "_")
            start, end = seconds(r.get("start")), seconds(r.get("end"))
            key = seconds(r["key"]) if r.get("key") else None
        except (ValueError, KeyError) as e:
            problems.append(f"take {r.get('take')}: {e}")
            continue
        if end <= start or (key is not None and not start <= key <= end):
            problems.append(f"take {take}: times must satisfy start <= key <= end")
            continue
        for phone in ("a", "b"):
            name = f"{block}_{phone.upper()}"
            if name not in videos or f"{block}_A" not in claps or name not in claps:
                problems.append(f"take {take}: no video or clap for {block}_{phone}")
                continue
            shift = claps[name] - claps[f"{block}_A"]  # phone b's clock = phone a's clock + shift
            clip = f"OWN-{take:02d}{phone}"
            cut(videos[name], max(0.0, start + shift), end + shift, out / f"{clip}.mov")
            log.append({"clip": clip, "scene": block, "label": label,
                        "second": f"{key - start:.1f}" if key is not None else "", "type": r.get("type", "").upper(),
                        "note": r.get("note", "")})
            print(f"{clip}: {start + shift:.1f}-{end + shift:.1f} s of {videos[name].name}")

    with open(out / "shotlog.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "scene", "label", "second", "type", "note"])
        w.writeheader()
        w.writerows(log)
    print(f"shotlog.csv: {len(log)} clips")
    for p in problems:
        print("PROBLEM", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
