#!/usr/bin/env python3
"""Build data/camera_ref.npz: ORB keypoints + descriptors of one early frame per recording session of the training
camera. No image is stored (positions and 256-bit codes only, nothing to rebuild a picture from); the service uses
them to register a new video to the training camera (cashaudit/cv/roi.py).

    python tools/build_camera_ref.py --videos /path/open/videos

Sessions were found by registering every clip to X-18 (the camera moved up to 66 px / 5 deg between sessions).
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cashaudit.cv.roi import N_FEATURES, orb_features  # noqa: E402

SESSIONS = {"s1": "3", "s2": "11", "s3": "18", "s4": "26", "s5": "35", "s6": "44"}  # one clip per session
AT_SEC = 0.5


def find(videos: str, num: str) -> str:
    hits = [p for p in glob.glob(os.path.join(videos, "*.mov")) if os.path.basename(p).split("-")[-1] == num + ".mov"]
    if not hits:
        raise SystemExit(f"clip {num} not found in {videos}")
    return hits[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", default=str(ROOT / "data" / "camera_ref.npz"))
    a = ap.parse_args()
    arrays = {}
    for name, num in SESSIONS.items():
        cap = cv2.VideoCapture(find(a.videos, num))
        cap.set(cv2.CAP_PROP_POS_MSEC, AT_SEC * 1000)
        ok, img = cap.read()
        if not ok:
            raise SystemExit(f"cannot read clip {num}")
        pts, desc = orb_features(img, N_FEATURES)
        arrays[f"{name}_pts"] = pts.astype(np.float32)  # normalized x, y
        arrays[f"{name}_desc"] = desc
        print(name, f"X-{num}", len(pts), "points")
    np.savez_compressed(a.out, names=np.array(list(SESSIONS)), **arrays)
    print("wrote", a.out, os.path.getsize(a.out) // 1024, "KB")


if __name__ == "__main__":
    main()
