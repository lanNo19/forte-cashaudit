"""Combine candidates from the event layer and the VLM into the final answer.

Policy (the ML lead owns these numbers; tune with tools/eval.py and tools/calibrate.py):
- drop candidates below the per-source floor, raised to the per-type floor in data/thresholds.json;
- types not seen in training (T00, T18-T20) need OPEN_WORLD_CONF: one false alarm costs a whole video;
- for types listed in EVENTS_OWN, a VLM-only candidate needs at least VLM_ALONE_CONF;
- merge candidates of the same type closer than MERGE_SEC; the event layer's timecode wins;
- the VLM verifier re-scores VLM candidates, and may VETO event-layer candidates only for types it can actually
  see (people, objects on the desk, the door). Tiny states (key in lock, screen lock, safe door) stay with CV;
- keep at most MAX_VIOLATIONS, highest confidence first.
"""

from __future__ import annotations

import json
import os
from typing import Callable, Dict, List, Optional

from . import kb
from .config import ROOT
from .schema import Violation

MERGE_SEC = float(os.getenv("FUSION_MERGE_SEC", "6"))
MIN_CONF = {"events": float(os.getenv("EVENTS_MIN_CONF", "0.5")), "vlm": float(os.getenv("VLM_MIN_CONF", "0.6"))}
VLM_ALONE_CONF = float(os.getenv("VLM_ALONE_CONF", "0.8"))
OPEN_WORLD_CONF = float(os.getenv("OPEN_WORLD_CONF", "0.85"))
MAX_VIOLATIONS = int(os.getenv("MAX_VIOLATIONS", "5"))
# Types the event layer is trusted with (decided at check-in 2).
EVENTS_OWN = set(filter(None, os.getenv("EVENTS_OWN", "").split(",")))
# Types the VLM can see well enough to veto an event-layer candidate.
VETO_TYPES = set(filter(None, os.getenv("VETO_TYPES", "T02,T03,T08,T09,T10,T11,T12,T16,T17,T18,T19,T20").split(",")))
VETO_BELOW = float(os.getenv("VETO_BELOW", "0.35"))
THRESHOLDS_PATH = os.getenv("THRESHOLDS_PATH", str(ROOT / "data" / "thresholds.json"))

Verifier = Callable[[Violation], Optional[float]]  # returns new confidence, or None when it could not check


def load_floors(path: str = THRESHOLDS_PATH) -> Dict[str, float]:
    try:
        with open(path, encoding="utf-8") as f:
            return {k: float(v) for k, v in json.load(f).get("floors", {}).items()}
    except (OSError, ValueError):
        return {}


def _floor(c: Violation, floors: Dict[str, float]) -> float:
    floor = max(MIN_CONF.get(c.source, 0.5), floors.get(c.type_id, 0.0))
    if c.type_id == kb.UNSEEN_TYPE_ID or not kb.seen_in_train(c.type_id):
        floor = max(floor, OPEN_WORLD_CONF)
    if c.source == "vlm" and c.type_id in EVENTS_OWN:
        floor = max(floor, VLM_ALONE_CONF)
    return floor


def fuse(candidates: List[Violation], verifier: Optional[Verifier] = None, floors: Optional[Dict[str, float]] = None) -> List[Violation]:
    floors = load_floors() if floors is None else floors
    kept = [c for c in candidates if c.confidence >= _floor(c, floors)]

    kept.sort(key=lambda v: (v.type_id, v.timecode_sec))
    merged: List[Violation] = []
    for c in kept:
        prev = merged[-1] if merged else None
        if prev and prev.type_id == c.type_id and abs(prev.timecode_sec - c.timecode_sec) <= MERGE_SEC:
            events_one = prev if prev.source == "events" else c if c.source == "events" else None
            winner = events_one or max(prev, c, key=lambda v: v.confidence)
            winner.confidence = max(prev.confidence, c.confidence)
            winner.source = "fused" if prev.source != c.source else winner.source
            winner.evidence_ts = sorted(set(prev.evidence_ts) | set(c.evidence_ts))
            merged[-1] = winner
        else:
            merged.append(c)

    if verifier:
        checked = []
        for v in merged:
            new_conf = verifier(v)
            if new_conf is None:
                checked.append(v)
            elif v.source in ("events", "fused"):
                if v.type_id in VETO_TYPES and new_conf < VETO_BELOW:
                    continue  # the VLM looked and clearly did not see it
                checked.append(v)
            else:
                v.confidence = new_conf
                if v.confidence >= _floor(v, floors):
                    checked.append(v)
        merged = checked

    merged.sort(key=lambda v: -v.confidence)
    return sorted(merged[:MAX_VIOLATIONS], key=lambda v: v.timecode_sec)
