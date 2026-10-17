"""Object states (#8): safe door, key in the safe lock, room door, laptop screen, lights.

- Safe / key / door / screen: crops of the ROIs from the per-video geometry (#7), sampled at CV_STATE_FPS, sent to the
  team's local VLM (same vLLM server and model as vlm.py, our own small request, vlm.py is not changed). One request per
  sampled second carries every crop and asks one closed question per crop. No training data needed, so it does not
  overfit the 1-2 training clips per state and works on a new camera.
- Lights: global brightness drop, no model.

Per-sample answers are smoothed (a state must hold for CV_STATE_HOLD samples) and turned into the start/end events of
events.py: safe_open/safe_close, key_in_lock/key_removed, door_open/door_close, screen_on/screen_locked, lights_off.
`answers_to_events` and `lights_events` work on plain lists, so they are unit-tested without a model.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..events import Event

STATE_FPS = float(os.getenv("CV_STATE_FPS", "1"))
STATE_HOLD = int(os.getenv("CV_STATE_HOLD", "2"))  # consecutive samples before a state change is believed
STATE_BUDGET_SEC = float(os.getenv("CV_STATE_BUDGET_SEC", "60"))  # stop asking after this; the rest of the job needs time
STATE_CONCURRENCY = int(os.getenv("CV_STATE_CONCURRENCY", os.getenv("VLM_CONCURRENCY", "4")))
CROP_SIDE = 336  # VLM input per crop (Qwen3-VL: 14 px patches, so 24x24 patches)
LIGHTS_DROP = 0.8  # held below this fraction of the opening brightness = lights off. Auto-exposure brightens the
# dark room again (X-11: 137 -> 20 -> ~90, i.e. 0.66); other clips never hold below 0.96 (training clips)
LIGHTS_HOLD_SEC = 2.0

# What each crop is asked. Answer `unknown` when the object is hidden (a person in front of it) or absent.
QUESTIONS = {
    "safe": ("safe_open", "safe_close", "Is the door of the grey metal safe open (swung away from the safe body)?"),
    "key": ("key_in_lock", "key_removed", "Is a key sticking out of the lock on the safe door?"),
    "door": ("door_open", "door_close", "Is the glass door of the room open (not shut in its frame)?"),
    "screen": ("screen_on", "screen_locked", "Is the laptop screen open and showing working content (documents, tables, "
               "applications)? Answer no if the lid is closed, the screen is dark, or it shows a lock/login screen."),
}

Box = Tuple[float, float, float, float]


# ---------- crops ----------

def crop(img: np.ndarray, box: Box, pad: float = 0.15, side: int = CROP_SIDE) -> Optional[np.ndarray]:
    """Padded crop of a normalized box, resized so its longer side is `side` (small objects are enlarged)."""
    import cv2

    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    a, b = int(max(0, x1 - pad * bw) * w), int(max(0, y1 - pad * bh) * h)
    c, d = int(min(1, x2 + pad * bw) * w), int(min(1, y2 + pad * bh) * h)
    if c - a < 4 or d - b < 4:
        return None
    out = img[b:d, a:c]
    s = side / max(out.shape[:2])
    return cv2.resize(out, (max(1, int(out.shape[1] * s)), max(1, int(out.shape[0] * s))), interpolation=cv2.INTER_CUBIC)


def lock_box(safe: Box) -> Box:
    """Where the lock is on the training safe: front face, left part, a bit above the middle (from the crops of X-22)."""
    x1, y1, x2, y2 = safe
    bw, bh = x2 - x1, y2 - y1
    return (x1 - 0.02 * bw, y1 + 0.30 * bh, x1 + 0.30 * bw, y1 + 0.60 * bh)


def roi_boxes(anchors: Dict[str, Tuple[float, Box]], door: Optional[Box]) -> Dict[str, Box]:
    """ROI name -> normalized box in this video. Missing anchors simply mean that state is not checked."""
    out: Dict[str, Box] = {}
    if "safe" in anchors:
        out["safe"] = anchors["safe"][1]
        out["key"] = lock_box(anchors["safe"][1])
    if "laptop" in anchors:
        out["screen"] = anchors["laptop"][1]
    if door:
        out["door"] = door
    return out


# ---------- VLM questions ----------

def build_prompt(names: Sequence[str]) -> str:
    lines = [f"Image {i + 1} ({n}): {QUESTIONS[n][2]}" for i, n in enumerate(names)]
    keys = ", ".join(f'"{n}": "yes" | "no" | "unknown"' for n in names)
    return ("Each image is a close-up crop of one object in a bank cash room, from a fixed CCTV camera.\n" + "\n".join(lines) +
            "\nAnswer unknown if the object is hidden by a person or not in the image.\n"
            "Return JSON only: {" + keys + "}")


def parse_answers(data: dict, names: Sequence[str]) -> Dict[str, Optional[bool]]:
    out: Dict[str, Optional[bool]] = {}
    for n in names:
        v = str(data.get(n, "unknown")).strip().lower()
        out[n] = True if v in ("yes", "true", "1") else False if v in ("no", "false", "0") else None
    return out


class VLMStateAsker:
    """Sends one request per sampled frame with all crops, through the VLM client's connection (vlm.py unchanged)."""

    def __init__(self, vlm):
        self.vlm = vlm

    def ask(self, crops: Dict[str, np.ndarray]) -> Dict[str, Optional[bool]]:
        from ..video import jpeg_b64
        from ..vlm import _extract_json

        names = list(crops)
        content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{jpeg_b64(crops[n], quality=90)}"}} for n in names]
        content.append({"type": "text", "text": build_prompt(names)})
        body = {"model": self.vlm.model, "messages": [{"role": "user", "content": content}], "temperature": 0.0, "max_tokens": 80}
        r = self.vlm.http.post(f"{self.vlm.base_url}/chat/completions", json=body)
        r.raise_for_status()
        return parse_answers(_extract_json(r.json()["choices"][0]["message"]["content"]), names)


def sample(frames: Sequence[Tuple[float, np.ndarray]], fps: float) -> List[Tuple[float, np.ndarray]]:
    out, next_t = [], None
    for t, img in frames:
        if next_t is None or t >= next_t - 1e-6:
            out.append((t, img))
            next_t = t + 1.0 / max(fps, 0.01)
    return out


def ask_all(frames, boxes: Dict[str, Box], ask: Callable[[Dict[str, np.ndarray]], Dict[str, Optional[bool]]],
            budget_sec: float = STATE_BUDGET_SEC, workers: int = STATE_CONCURRENCY) -> Tuple[Dict[str, List[Tuple[float, Optional[bool]]]], Dict]:
    """-> per ROI [(t, yes/no/None)], info. Frames past the budget are skipped (answers so far are kept)."""
    t_start = time.time()
    picked = sample(frames, STATE_FPS)

    def one(item):
        t, img = item
        if time.time() - t_start > budget_sec:
            return t, None, "budget"
        crops = {n: c for n, b in boxes.items() if (c := crop(img, b)) is not None}
        if not crops:
            return t, {}, None
        try:
            return t, ask(crops), None
        except Exception as e:  # one bad request must not lose the others
            return t, None, f"{type(e).__name__}"

    series: Dict[str, List[Tuple[float, Optional[bool]]]] = {n: [] for n in boxes}
    errors: Dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for t, ans, err in sorted(ex.map(one, picked), key=lambda r: r[0]):
            if err:
                errors[err] = errors.get(err, 0) + 1
                continue
            for n in boxes:
                series[n].append((t, (ans or {}).get(n)))
    return series, {"samples": len(picked), "errors": errors, "sec": round(time.time() - t_start, 1)}


# ---------- answers -> events ----------

def answers_to_events(series: Sequence[Tuple[float, Optional[bool]]], on_kind: str, off_kind: str, hold: int = STATE_HOLD,
                      **fields) -> List[Event]:
    """Smoothed state changes, timed at the first sample of the new state. Unknown answers neither confirm nor break a
    run. A state that is already on at the start is reported (T04/T06 need it); an opening "off" is not."""
    out: List[Event] = []
    state: Optional[bool] = None
    run_val: Optional[bool] = None
    run_t, run_n = 0.0, 0
    for t, v in series:
        if v is None:
            continue
        if v == run_val:
            run_n += 1
        else:
            run_val, run_t, run_n = v, t, 1
        if run_n >= hold and run_val != state:
            if run_val or state is not None:
                out.append(Event(run_t, on_kind if run_val else off_kind, conf=0.8, extra={"source": "cv_vlm"}, **fields))
            state = run_val
    return out


def lights_events(frames: Sequence[Tuple[float, np.ndarray]]) -> List[Event]:
    """lights_off when the whole frame stays much darker than at the start for LIGHTS_HOLD_SEC."""
    if len(frames) < 4:
        return []
    lum = [(t, float(img.mean())) for t, img in frames]
    base = float(np.median([v for _, v in lum[: max(2, len(lum) // 10)]]))
    if base < 20:  # already dark: nothing to compare with
        return []
    start = None
    for t, v in lum:
        if v < LIGHTS_DROP * base:
            start = t if start is None else start
            if t - start >= LIGHTS_HOLD_SEC:
                return [Event(start, "lights_off", conf=0.9, extra={"source": "cv", "brightness": round(v / base, 2)})]
        else:
            start = None
    return []


def state_events(frames, boxes: Dict[str, Box], ask: Optional[Callable] = None) -> Tuple[List[Event], Dict]:
    events = lights_events(frames)
    info: Dict = {"rois": sorted(boxes), "lights_off": bool(events)}
    if ask is not None and boxes:
        series, info["vlm"] = ask_all(frames, boxes, ask)
        extra = {"safe": {"zone": "safe"}, "key": {"zone": "safe"}, "door": {"zone": "door"}, "screen": {"obj": "laptop"}}
        for n, s in series.items():
            on, off = QUESTIONS[n][0], QUESTIONS[n][1]
            events += answers_to_events(s, on, off, **extra[n])
            info.setdefault("answers", {})[n] = "".join("y" if v else "n" if v is False else "?" for _, v in s)
    return sorted(events, key=lambda e: e.t), info
