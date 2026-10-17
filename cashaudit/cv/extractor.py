"""CV event layer entry point used by analyzers/pipeline.py.

Per video: camera geometry from the first frame (registration to the training camera + OWLv2 anchors, #7), then
people (person_enter/exit with roles, cashier_zone, #6), then object states (safe, key, door, screen via the local VLM
on ROI crops; lights from brightness, #8).
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..events import EventLog
from . import roi, states
from .camera import load_camera, normalize
from .detector import PersonDetector
from .people import people_events, torso_hist

CV_ROI = os.getenv("CV_ROI", "1") == "1"  # 0 = always the hand-drawn config of data/cameras.json
CV_ROI_OWL = os.getenv("CV_ROI_OWL", "1") == "1"  # 0 = skip OWLv2 (no safe zone / fallback geometry)
CV_STATES = os.getenv("CV_STATES", "1") == "1"  # 0 = no object states (#8)


class CVEventExtractor:
    def __init__(self, device: int = 0):
        self.detector = PersonDetector(device)
        self.proposer = roi.RoiProposer(device)
        self.base = load_camera()
        self.refs = roi.load_refs()
        self.last_info: Dict = {}
        self.vlm = None  # set by the pipeline when its VLM server answers; object states need it (lights do not)

    def load(self) -> Optional[str]:
        if CV_ROI and CV_ROI_OWL:
            self.proposer.load()  # optional: without it, registration and the hand-drawn config still work
        return self.detector.load()

    def camera_for(self, frames: List[Tuple[float, np.ndarray]]):
        """Geometry + anchors from the first frame. Never raises: falls back to the hand-drawn config, no anchors."""
        if not CV_ROI or not frames:
            return self.base, {}, {"roi": "off"}
        img = frames[0][1]
        try:
            H, reg = roi.register(img, self.refs) if self.refs else (None, {"session": "", "inliers": 0})
            found = self.proposer.anchors([img]) if (CV_ROI_OWL and self.proposer.model is not None) else {}
            cam, info = roi.build_camera(self.base, H, found)
            return cam, found, dict(info, registration=reg, registered=H is not None)
        except Exception as e:
            return self.base, {}, {"roi_error": f"{type(e).__name__}: {e}"}

    def extract(self, frames: List[Tuple[float, np.ndarray]], duration: float, scene: Optional[str] = None) -> EventLog:
        t0 = time.time()
        cam, found, roi_info = self.camera_for(frames)
        t_roi = time.time() - t0
        rows = []
        for t, img in frames:
            h, w = img.shape[:2]
            boxes, confs = self.detector.detect(img)
            nb = normalize(boxes, w, h)
            rows.append((t, nb, [float(c) for c in confs], [torso_hist(img, b) for b in nb]))
        t_det = time.time() - t0 - t_roi
        events, scene_used, info = people_events(rows, cam, scene)
        state_info: Dict = {"off": True}
        if CV_STATES:
            t1 = time.time()
            try:
                door = found["doorway"][1] if "doorway" in found else None
                asker = states.VLMStateAsker(self.vlm).ask if self.vlm is not None else None
                evs, state_info = states.state_events(frames, states.roi_boxes(found, door), asker)
                events = sorted(events + evs, key=lambda e: e.t)
            except Exception as e:  # states must never cost the people events
                state_info = {"error": f"{type(e).__name__}: {e}"}
            state_info["sec"] = round(time.time() - t1, 2)
        self.last_info = dict(info, camera=cam.name, roi=roi_info, roi_sec=round(t_roi, 2), detect_sec=round(t_det, 2),
                              states=state_info, scene_from_cv=scene is None and scene_used is not None)
        return EventLog(events=events, duration=duration, scene=scene_used)
