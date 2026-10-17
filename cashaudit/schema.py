"""Internal result types and conversion to the jury JSON (brief, section 10).

Everything inside the pipeline speaks these dataclasses. Only `to_api_json` knows the external format,
so the format can never drift between the API, the UI and the eval script.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass
class Violation:
    type_id: str  # "T01".."T17" from rules_kb.json; "T00" = unseen type found by the open-world pass
    type: str  # human-readable type name shown to the jury
    timecode_sec: float  # the moment the rule was broken
    description: str  # what is visible + why it may be a violation + legal basis
    evidence_ts: List[float]  # 1-3 timestamps; frames are cut from the video at these times
    confidence: float = 1.0
    source: str = "unknown"  # vlm | events | fused | oracle | stub
    refs: List[Dict[str, str]] = field(default_factory=list)  # [{act, point, url, strength}]

    def clamp(self, duration: Optional[float]) -> "Violation":
        if duration:
            last = max(0.0, duration - 0.05)
            self.timecode_sec = float(min(max(0.0, self.timecode_sec), last))
            self.evidence_ts = [float(min(max(0.0, t), last)) for t in self.evidence_ts]
        if not self.evidence_ts:
            self.evidence_ts = [self.timecode_sec]
        # 1-3 frames, de-duplicated, in time order
        uniq = sorted({round(t, 2) for t in self.evidence_ts})
        if len(uniq) > 3:
            # keep the frames nearest to the timecode
            uniq = sorted(sorted(uniq, key=lambda t: abs(t - self.timecode_sec))[:3])
        self.evidence_ts = uniq
        return self


@dataclass
class AnalysisResult:
    violations: List[Violation] = field(default_factory=list)
    meta: Dict = field(default_factory=dict)  # runtime, model, warnings, partial flag...

    @property
    def verdict(self) -> str:
        return "violation" if self.violations else "no_violation"

    def to_dict(self) -> Dict:
        return {"violations": [asdict(v) for v in self.violations], "meta": self.meta}

    @staticmethod
    def from_dict(d: Dict) -> "AnalysisResult":
        return AnalysisResult([Violation(**v) for v in d.get("violations", [])], d.get("meta", {}))


def to_api_json(job_id: str, result: AnalysisResult, frames_b64: Dict[float, str], extra: bool = False) -> Dict:
    """Exactly the section-10 shape. `frames_b64` maps evidence timestamp -> base64 JPEG."""
    violations = []
    for v in sorted(result.violations, key=lambda x: x.timecode_sec):
        item = {
            "type": v.type,
            "timecode_sec": round(float(v.timecode_sec), 2),
            "description": v.description,
            "evidence_frames": [
                {"timestamp_sec": round(float(t), 2), "image_base64": frames_b64[round(t, 2)]}
                for t in v.evidence_ts
                if round(t, 2) in frames_b64
            ],
        }
        if extra:
            item.update({"type_id": v.type_id, "confidence": round(v.confidence, 3), "source": v.source, "refs": v.refs})
        violations.append(item)
    out = {"job_id": job_id, "status": "done", "verdict": result.verdict, "violations": violations}
    if extra:
        out["meta"] = result.meta
    return out
