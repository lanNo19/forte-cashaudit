"""Per-video camera geometry (#7): no hard-coded pixels for a new camera, and it follows a moved camera.

Two steps, both on the first frames of the video:

1. Registration to the training camera. `data/camera_ref.npz` holds ORB points (positions + 256-bit codes, no image)
   of one frame per recording session (tools/build_camera_ref.py). If the video matches a session, a homography
   maps its points into that session's frame, where the hand-drawn polygons of `data/cameras.json` hold. This undoes
   shifts, zoom, rotation, tilt and mirroring. Matching each session separately avoids the parallax a physical camera
   move causes on the floor.
2. Anchors with OWLv2 (Apache-2.0, open vocabulary): safe, door handle, doorway, desk. The safe zone is always taken
   from here (the safe moves between layouts). If registration fails (a different camera), the door and the room
   floor are placed relative to the handle and the desk (`propose`), else the hand-drawn config stays.

`propose` and `register_points` work on plain arrays, so they are unit-tested without a model.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .camera import Camera

ROI_MODEL = os.getenv("CV_ROI_MODEL", "google/owlv2-large-patch14-ensemble")
REF_FILE = Path(__file__).resolve().parents[2] / "data" / "camera_ref.npz"
N_FEATURES = 1500
MIN_INLIERS = 40  # training clips match their session with 90+ inliers; a different room gets a handful
QUERIES = {"safe": "a safe", "handle": "a door handle", "doorway": "an open doorway", "desk": "an office desk"}
MIN_SCORE = {"safe": 0.15, "handle": 0.12, "doorway": 0.2, "desk": 0.3}

# Fallback geometry (no registration), relative to the handle's right/bottom edge, scaled by desk width.
REF_DESK_W = 0.57
FLOOR_LINE = [(-1.0, 0.10), (0.08, 0.10), (0.10, 0.065), (0.22, 0.025), (0.34, -0.045), (2.0, -0.045)]
DOOR_W = 0.30
DOOR_Y = (0.035, 0.19)

Box = Tuple[float, float, float, float]


def _clip01(p):
    return (float(min(1.0, max(0.0, p[0]))), float(min(1.0, max(0.0, p[1]))))


# ---------- registration ----------

def orb_features(img_bgr: np.ndarray, n: int = N_FEATURES) -> Tuple[np.ndarray, np.ndarray]:
    """-> normalized (x, y) points (N, 2), ORB descriptors (N, 32) uint8."""
    import cv2

    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kps, desc = cv2.ORB_create(n).detectAndCompute(g, None)
    if desc is None:
        return np.zeros((0, 2), np.float32), np.zeros((0, 32), np.uint8)
    h, w = g.shape
    return np.array([(k.pt[0] / w, k.pt[1] / h) for k in kps], np.float32), desc


def register_points(pts: np.ndarray, desc: np.ndarray, ref_pts: np.ndarray, ref_desc: np.ndarray) -> Tuple[Optional[np.ndarray], int]:
    """Homography (3x3, normalized coords) video -> reference, and its inlier count."""
    import cv2

    if len(desc) < 8 or len(ref_desc) < 8:
        return None, 0
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc, ref_desc, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < 8:
        return None, 0
    src = np.float32([pts[m.queryIdx] for m in good])
    dst = np.float32([ref_pts[m.trainIdx] for m in good])
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 0.008)  # ~4 px at 464 px width
    if H is None:
        return None, 0
    return H, int(mask.sum())


def load_refs(path: Path = REF_FILE) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    try:
        z = np.load(path)
    except OSError:
        return {}
    return {str(n): (z[f"{n}_pts"], z[f"{n}_desc"]) for n in z["names"]}


MIRROR = np.array([[-1.0, 0, 1.0], [0, 1.0, 0], [0, 0, 1.0]])


def register(img_bgr: np.ndarray, refs: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Tuple[Optional[np.ndarray], Dict]:
    """Best session match over the frame and its mirror. -> (H video->ref or None, info)."""
    import cv2

    best: Tuple[Optional[np.ndarray], int, str, bool] = (None, 0, "", False)
    for mirrored in (False, True):
        pts, desc = orb_features(cv2.flip(img_bgr, 1) if mirrored else img_bgr)
        for name, (rp, rd) in refs.items():
            H, n = register_points(pts, desc, rp, rd)
            if H is not None and n > best[1]:
                best = (H @ MIRROR if mirrored else H, n, name, mirrored)
    H, n, name, mirrored = best
    info = {"session": name, "inliers": n, "mirrored": mirrored}
    return (H if n >= MIN_INLIERS else None), info


# ---------- anchors (OWLv2) ----------

def safe_zone(safe: Box, s: float = 1.0) -> List[Tuple[float, float]]:
    """Floor where the cashier stands to work the safe: around and in front of the box."""
    x1, y1, x2, y2 = safe
    return [_clip01(p) for p in [(x1 - 0.02, y1 + 0.04), (x2 + 0.1 * s, y1 + 0.04), (x2 + 0.1 * s, y2 + 0.09 * s), (x1 - 0.02, y2 + 0.09 * s)]]


def propose(found: Dict[str, Tuple[float, Box]], base: Optional[Camera] = None) -> Tuple[Camera, Dict]:
    """Fallback geometry for an unregistered camera from anchors {name: (score, box)}; `base` if anchors are missing."""
    base = base or Camera()
    info: Dict = {"anchors": {k: round(v[0], 2) for k, v in found.items()}}
    handle, desk = found.get("handle"), found.get("desk")
    if not handle or not desk:
        info["fallback"] = "no handle" if not handle else "no desk"
        return base, info
    hx, hy = handle[1][2], handle[1][3]
    s = float(min(2.0, max(0.5, (desk[1][2] - desk[1][0]) / REF_DESK_W)))
    room = [_clip01((hx + dx * s, hy + dy * s)) for dx, dy in FLOOR_LINE] + [(1.0, 1.0), (0.0, 1.0)]
    door_left = hx - DOOR_W * s
    if found.get("doorway") and abs(found["doorway"][1][2] - hx) < 0.1 * s:
        door_left = found["doorway"][1][0]
    y0, y1 = hy + DOOR_Y[0] * s, hy + DOOR_Y[1] * s
    door = [_clip01(p) for p in [(door_left, y0), (hx + 0.02 * s, y0), (hx + 0.02 * s, y1), (door_left, y1 + 0.04 * s)]]
    zones = {}
    if found.get("safe"):
        z = safe_zone(found["safe"][1], s)
        zones = {"A": {"safe": z}, "B": {"safe": z}}
    info.update(scale=round(s, 2))
    return Camera(name="auto", room_floor=room, door=door, zones=zones), info


def _warp(poly, H):
    out = []
    for x, y in poly:
        v = H @ np.array([x, y, 1.0])
        out.append((float(v[0] / v[2]), float(v[1] / v[2])))
    return out


def build_camera(base: Camera, H: Optional[np.ndarray], found: Dict[str, Tuple[float, Box]]) -> Tuple[Camera, Dict]:
    """Registered: hand-drawn polygons + safe zone from the anchors (moved into the reference frame).
    Not registered: geometry proposed from the anchors."""
    if H is None:
        return propose(found, base)
    zones = {k: dict(v) for k, v in base.zones.items()}
    info: Dict = {"anchors": {k: round(v[0], 2) for k, v in found.items()}}
    if found.get("safe"):
        z = _warp(safe_zone(found["safe"][1]), H)
        zones = {"A": {"safe": z}, "B": {"safe": z}}
    cam = Camera(name=base.name + "+reg", room_floor=base.room_floor, door=base.door, zones=zones, to_ref=H)
    return cam, info


class RoiProposer:
    """OWLv2 anchors on early frames; keeps the best-scoring box per anchor."""

    def __init__(self, device: int = 0):
        self.device = device
        self.model = self.proc = None
        self.error: Optional[str] = None

    def load(self) -> Optional[str]:
        if self.model is not None or self.error:
            return self.error
        try:
            import torch
            from transformers import Owlv2ForObjectDetection, Owlv2Processor

            self.dev = f"cuda:{self.device % max(1, torch.cuda.device_count())}" if torch.cuda.is_available() else "cpu"
            self.proc = Owlv2Processor.from_pretrained(ROI_MODEL)
            self.model = Owlv2ForObjectDetection.from_pretrained(ROI_MODEL).to(self.dev).eval()
        except Exception as e:
            self.error = f"ROI proposer unavailable: {type(e).__name__}: {e}"
        return self.error

    def anchors(self, frames: Sequence[np.ndarray]) -> Dict[str, Tuple[float, Box]]:
        import torch
        from PIL import Image

        names = list(QUERIES)
        best: Dict[str, Tuple[float, Box]] = {}
        for img in frames:
            h, w = img.shape[:2]
            side = max(h, w)  # OWLv2 pads to a square: boxes come back in padded coordinates
            inp = self.proc(text=[[QUERIES[n] for n in names]], images=Image.fromarray(np.ascontiguousarray(img[:, :, ::-1])),
                            return_tensors="pt").to(self.dev)
            with torch.no_grad():
                out = self.model(**inp)
            res = self.proc.post_process_grounded_object_detection(out, threshold=0.05, target_sizes=torch.tensor([[side, side]]))[0]
            for sc, lab, b in zip(res["scores"].tolist(), res["labels"].tolist(), res["boxes"].tolist()):
                n = names[lab]
                if sc >= MIN_SCORE[n] and sc > best.get(n, (0.0,))[0]:
                    best[n] = (float(sc), (b[0] / w, b[1] / h, b[2] / w, b[3] / h))
        return best
