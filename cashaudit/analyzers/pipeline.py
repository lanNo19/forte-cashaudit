"""The real analyzer: frames -> (VLM scan + VLM events) + (CV event layer) -> rules -> fusion -> result.

Ownership:
- EventExtractor.extract  -> CV engineer (replace the placeholder; emit events per events.py)
- VLM prompts / verify    -> ML lead (cashaudit/vlm.py)
- fusion policy           -> ML lead (cashaudit/fusion.py, data/thresholds.json)

Runs end to end today: with no VLM server and the placeholder extractor it returns "no violation" and lists
warnings in meta, so the service never crashes while the pieces are being built.
"""

from __future__ import annotations

import importlib.util
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

import numpy as np

from .. import rules
from ..config import settings
from ..events import EventLog
from ..fusion import fuse
from ..schema import AnalysisResult, Violation
from ..video import probe, sample_frames
from ..vlm import VLMClient, scene_type_ids
from .base import Deadline, Report

WINDOW_SEC = float(os.getenv("VLM_WINDOW_SEC", "16"))
STRIDE_SEC = float(os.getenv("VLM_STRIDE_SEC", "8"))
VLM_MODE = os.getenv("VLM_MODE", "both")  # violations | events | both
EVENT_SAMPLES = int(os.getenv("VLM_EVENT_SAMPLES", "1"))  # 3-5 = majority vote (research: +AUC, more GPU time)


class EventExtractor:
    """CV layer (cashaudit/cv): people -> person_enter/exit with roles + cashier_zone. Returns an EventLog.

    `ready` only says the dependencies are installed (cheap check, no GPU work at start-up); the model loads on the
    first video. Any CV failure returns an empty log and sets `error`, so the job still answers.
    """

    def __init__(self, device: int = 0):
        self.device = device
        self.error: Optional[str] = None
        self.info: dict = {}
        self.vlm = None  # the pipeline's VLM client when its server answers: object states (#8) ask it about ROI crops
        self._cv = None
        self.ready = os.getenv("CV_ENABLED", "1") == "1" and all(
            importlib.util.find_spec(m) is not None for m in ("torch", "rfdetr"))
        if not self.ready:
            self.error = "CV disabled (CV_ENABLED=0)" if os.getenv("CV_ENABLED", "1") != "1" else "torch/rfdetr not installed"

    def extract(self, frames: List[Tuple[float, np.ndarray]], duration: float, scene: Optional[str] = None) -> EventLog:
        try:
            if self._cv is None:
                from ..cv.extractor import CVEventExtractor

                cv = CVEventExtractor(self.device)
                err = cv.load()
                if err:
                    raise RuntimeError(err)
                self._cv = cv
            self._cv.vlm = self.vlm
            log = self._cv.extract(frames, duration, scene)
            self.error, self.info = None, self._cv.last_info
            return log
        except Exception as e:  # the CV layer must never fail the job
            self.error = f"CV layer failed: {type(e).__name__}: {e}"
            return EventLog(events=[], duration=duration, scene=scene)


def _dedupe(events):
    """Overlapping windows report the same event twice: keep one per kind/obj/actor/zone within 3 s."""
    out = []
    for e in sorted(events, key=lambda x: -x.conf):
        if not any(o.kind == e.kind and o.obj == e.obj and o.actor == e.actor and o.zone == e.zone and abs(o.t - e.t) <= 3 for o in out):
            out.append(e)
    return sorted(out, key=lambda x: x.t)


def _windows(frames, win, stride):
    if not frames:
        return []
    out, t_end, start = [], frames[-1][0], 0.0
    while True:
        chunk = [f for f in frames if start <= f[0] < start + win]
        if chunk:
            out.append(chunk)
        if start + win >= t_end:
            break
        start += stride
    return out


def _thin(chunk, max_n):
    if len(chunk) <= max_n:
        return chunk
    step = len(chunk) / max_n
    return [chunk[int(i * step)] for i in range(max_n)]


def _around(frames, t, half=3.0, max_n=8):
    return _thin([f for f in frames if abs(f[0] - t) <= half], max_n)


class PipelineAnalyzer:
    name = "pipeline"

    def __init__(self, device: int = 0):
        self.device = device
        urls = settings.vlm_base_urls
        self.vlm = VLMClient(base_url=urls[device % len(urls)])  # one vLLM per GPU, worker i -> server i
        self.events = EventExtractor(device)
        if os.getenv("VLM_WARMUP", "1") == "1":
            self.vlm.warmup_async()

    def analyze(self, video_path: str, deadline: Deadline, report: Report) -> AnalysisResult:
        t_start = time.time()
        meta = {"analyzer": self.name, "warnings": [], "stages": {}, "vlm_url": self.vlm.base_url}
        info = probe(video_path)
        # Budget-aware sampling: the queue may already have eaten part of the 5 minutes.
        fps = settings.vlm_fps if deadline.remaining() > 150 else max(1.0, settings.vlm_fps / 2)
        frames = sample_frames(video_path, fps=fps, max_side=settings.vlm_max_side)
        meta.update({"duration": info.duration, "fps_used": fps, "n_frames": len(frames)})
        meta["stages"]["decode"] = round(time.time() - t_start, 2)

        candidates: List[Violation] = []
        vlm_ok = self.vlm.available()
        if not vlm_ok:
            meta["warnings"].append(f"VLM server not reachable at {self.vlm.base_url}")

        # Scene decides which rule cards the VLM sees (A = client desk, B = cash room)
        scene = None
        if vlm_ok and settings.vlm_scene_filter:
            try:
                scene = self.vlm.classify_scene(frames)
            except Exception as e:
                meta["warnings"].append(f"scene question failed: {e}")

        # Path B: CV event layer + rules (precise timecodes for states)
        t0 = time.time()
        log = EventLog(events=[], duration=info.duration, scene=scene)
        if self.events.ready:
            self.events.vlm = self.vlm if vlm_ok else None
            log = self.events.extract(frames, info.duration, scene)
            if log.scene and log.scene != scene:
                # CV's behaviour guess (cashier stays + visitors = A, cashier comes and goes = B) was never wrong on the
                # 42 clips; the VLM called the session-5 client desk a cash room, which showed the scan the wrong rule
                # cards (X-38: T02 at 32 s instead of T09 at 33 s). The CV scene drives everything after this point.
                meta["scene_vlm"], scene = scene, log.scene
            log.scene = log.scene or scene
            candidates += rules.run_rules(log)
            meta["n_events"] = len(log.events)
            meta["cv"] = self.events.info
            if self.events.error:
                meta["warnings"].append(self.events.error)
        else:
            meta["warnings"].append(f"event layer off: {self.events.error}")
        meta["scene"] = scene
        meta["stages"]["events"] = round(time.time() - t0, 2)
        report(AnalysisResult(fuse(list(candidates)), dict(meta, partial=True)))

        # Path A: VLM over sliding windows, several windows in flight so vLLM can batch them.
        # VLM_MODE=violations: the VLM names violations; events: it fills the event list and rules.py decides; both.
        t0 = time.time()
        vlm_events = []
        if vlm_ok:
            ids = scene_type_ids(scene)
            windows = [_thin(w, settings.vlm_max_frames) for w in _windows(frames, WINDOW_SEC, STRIDE_SEC)]

            def work(window):
                if deadline.remaining() < 25:
                    return [], [], "skipped (deadline)"
                vs, evs = [], []
                if VLM_MODE in ("violations", "both"):
                    vs = self.vlm.scan_window(window, ids)
                if VLM_MODE in ("events", "both"):
                    evs = self.vlm.extract_events(window, samples=EVENT_SAMPLES, temperature=0.7)
                return vs, evs, None

            with ThreadPoolExecutor(max_workers=max(1, settings.vlm_concurrency)) as ex:
                futs = {ex.submit(work, w): w[0][0] for w in windows}
                for fut in as_completed(futs):
                    try:
                        vs, evs, note = fut.result()
                        candidates += vs
                        vlm_events += evs
                        if note:
                            meta["warnings"].append(f"vlm window {futs[fut]:.0f}s {note}")
                    except Exception as e:  # one bad window must not kill the job
                        meta["warnings"].append(f"vlm window {futs[fut]:.0f}s failed: {e}")
                    report(AnalysisResult(fuse(list(candidates)), dict(meta, partial=True)))
            if vlm_events:
                merged = EventLog(events=_dedupe(log.events + vlm_events), duration=info.duration, scene=scene)
                cv_keys = {(v.type_id, round(v.timecode_sec)) for v in candidates if v.source == "events"}
                for v in rules.run_rules(merged):
                    if (v.type_id, round(v.timecode_sec)) not in cv_keys:
                        v.source = "vlm"  # came from VLM-reported events: held to the VLM floor in fusion
                        candidates.append(v)
                meta["n_vlm_events"] = len(vlm_events)
                if os.getenv("DEBUG_EVENTS"):
                    meta["vlm_events"] = [{k: (lambda x: x if isinstance(x, (int, float, type(None))) else str(x))(getattr(e, k, None)) for k in ("t", "kind", "actor", "obj", "zone", "conf")} for e in merged.events]
            meta["n_windows"] = len(windows)
            if os.getenv("DEBUG_EVENTS"):
                meta["candidates"] = [{k: (lambda x: x if isinstance(x, (int, float, type(None))) else str(x))(getattr(v, k, None)) for k in ("type_id", "timecode_sec", "confidence", "conf", "source")} for v in candidates]
        meta["stages"]["vlm_scan"] = round(time.time() - t0, 2)

        # Fusion + VLM verification of each surviving candidate (may veto visible types, see fusion.py)
        t0 = time.time()

        def verifier(v: Violation):
            if not vlm_ok or deadline.remaining() < 15:
                return None
            return self.vlm.verify(_around(frames, v.timecode_sec), v)

        final = fuse(candidates, verifier=verifier if vlm_ok else None)
        meta["stages"]["fusion"] = round(time.time() - t0, 2)
        meta["runtime_sec"] = round(time.time() - t_start, 2)
        meta["n_candidates"] = len(candidates)
        return AnalysisResult([v.clamp(info.duration) for v in final], meta)
