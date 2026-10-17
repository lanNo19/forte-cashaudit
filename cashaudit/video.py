"""Video helpers: probing, frame sampling, evidence JPEGs, timestamp burn-in."""

from __future__ import annotations

import base64
import json
import subprocess
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class VideoInfo:
    duration: float
    fps: float
    width: int
    height: int
    frames: int


class BadVideo(Exception):
    pass


def probe(path: str) -> VideoInfo:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height,r_frame_rate,nb_frames:format=duration", "-of", "json", path],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout
        d = json.loads(out)
        s = d["streams"][0]
        num, den = (s.get("r_frame_rate") or "30/1").split("/")
        fps = float(num) / float(den or 1)
        dur = float(d.get("format", {}).get("duration") or 0)
        nb = int(s.get("nb_frames") or round(dur * fps))
        if dur <= 0 or s.get("width") is None:
            raise BadVideo("no video stream or zero duration")
        return VideoInfo(dur, fps, int(s["width"]), int(s["height"]), nb)
    except (subprocess.SubprocessError, KeyError, IndexError, ValueError, json.JSONDecodeError) as e:
        raise BadVideo(str(e)) from e


def sample_frames(path: str, fps: float, max_side: int = 0) -> List[Tuple[float, np.ndarray]]:
    """Decode sequentially and keep every k-th frame. Returns [(t_sec, BGR image)].
    Timestamps come from the frame index, so they line up with the jury's timecodes."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise BadVideo("cannot open video")
    native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(native / max(fps, 0.1))))
    out, idx = [], 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, img = cap.retrieve()
            if ok:
                if max_side and max(img.shape[:2]) > max_side:
                    scale = max_side / max(img.shape[:2])
                    img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
                out.append((idx / native, img))
        idx += 1
    cap.release()
    return out


def frame_at(path: str, t: float) -> np.ndarray:
    """Accurate single-frame grab with ffmpeg (seek before -i is frame-accurate in modern ffmpeg)."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{max(0.0, t):.3f}", "-i", path, "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
        capture_output=True, timeout=30,
    )
    if proc.returncode != 0 or not proc.stdout:
        raise BadVideo(f"cannot grab frame at {t:.2f}s")
    img = cv2.imdecode(np.frombuffer(proc.stdout, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise BadVideo(f"cannot decode frame at {t:.2f}s")
    return img


def jpeg_b64(img: np.ndarray, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise BadVideo("jpeg encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def burn_timestamp(img: np.ndarray, t: float) -> np.ndarray:
    """Write a small 't=12.5s' into the top-left corner. Kept small on purpose: research found large overlays hurt
    temporal grounding (NumPro), and in video mode Qwen3-VL's native timestamps are used instead."""
    out = img.copy()
    label = f"t={t:.1f}s"
    scale = max(0.4, out.shape[1] / 1100)
    th = 1
    (w, h), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    cv2.rectangle(out, (0, 0), (w + 12, h + base + 12), (0, 0, 0), -1)
    cv2.putText(out, label, (6, h + 6), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 255), th, cv2.LINE_AA)
    return out
