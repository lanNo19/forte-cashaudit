"""Person detector: RF-DETR (Apache-2.0, Nano to Large only) on the sampled frames.

Loaded lazily so the service starts without torch or weights; `PersonDetector.load()` returns a reason string when
it cannot run, and the pipeline turns that into a warning in `meta`.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np

CV_MODEL = os.getenv("CV_DETECTOR", "small")  # nano | small | medium | base | large
CV_THRESHOLD = float(os.getenv("CV_DET_THRESHOLD", "0.3"))
PERSON = 1  # COCO class id in RF-DETR's label map


class PersonDetector:
    def __init__(self, device: int = 0):
        self.device = device
        self.model = None
        self.error: Optional[str] = None

    def load(self) -> Optional[str]:
        """Returns None when ready, else why not."""
        if self.model is not None or self.error:
            return self.error
        try:
            import torch
            import rfdetr

            cls = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium",
                   "base": "RFDETRBase", "large": "RFDETRLarge"}[CV_MODEL]
            dev = f"cuda:{self.device % max(1, torch.cuda.device_count())}" if torch.cuda.is_available() else "cpu"
            self.model = getattr(rfdetr, cls)(device=dev)
        except Exception as e:  # missing torch/rfdetr/weights must not stop the service
            self.error = f"person detector unavailable: {type(e).__name__}: {e}"
        return self.error

    def detect(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """-> (N, 4) xyxy pixel boxes, (N,) confidences, persons only."""
        d = self.model.predict(np.ascontiguousarray(img_bgr[:, :, ::-1]), threshold=CV_THRESHOLD)
        m = d.class_id == PERSON
        return d.xyxy[m], d.confidence[m]

    def detect_all(self, frames: List[Tuple[float, np.ndarray]]):
        return [(t, *self.detect(img)) for t, img in frames]
