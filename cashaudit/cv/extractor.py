"""CV event layer entry point used by analyzers/pipeline.py.

Today: people (person_enter/exit with roles, cashier_zone). Next: ROI states (#8) and auto ROIs (#7).
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..events import EventLog
from .camera import load_camera, normalize
from .detector import PersonDetector
from .people import people_events, torso_hist


class CVEventExtractor:
    def __init__(self, device: int = 0):
        self.detector = PersonDetector(device)
        self.camera = load_camera()
        self.last_info: Dict = {}

    def load(self) -> Optional[str]:
        return self.detector.load()

    def extract(self, frames: List[Tuple[float, np.ndarray]], duration: float, scene: Optional[str] = None) -> EventLog:
        t0 = time.time()
        rows = []
        for t, img in frames:
            h, w = img.shape[:2]
            boxes, confs = self.detector.detect(img)
            nb = normalize(boxes, w, h)
            rows.append((t, nb, [float(c) for c in confs], [torso_hist(img, b) for b in nb]))
        t_det = time.time() - t0
        events, scene_used, info = people_events(rows, self.camera, scene)
        self.last_info = dict(info, camera=self.camera.name, detect_sec=round(t_det, 2), scene_from_cv=scene is None and scene_used is not None)
        return EventLog(events=events, duration=duration, scene=scene_used)
