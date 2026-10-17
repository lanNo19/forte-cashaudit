"""TEST ONLY. Answers training clips from the annotations (matched by file hash).

Use it to check the plumbing end to end: the eval script must score 100% with it.
Any file that is not a training clip gets "no violation", so it is useless on test videos.
Never deploy with ANALYZER=oracle.
"""

import csv
import hashlib
import json
from typing import Dict, List

from .. import kb
from ..config import settings
from ..schema import AnalysisResult, Violation
from .base import Deadline, Report


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class OracleAnalyzer:
    name = "oracle"

    def __init__(self, device: int = 0):
        md5 = json.loads(settings.train_md5_path.read_text())
        rows = list(csv.DictReader(open(settings.annotations_path, encoding="utf-8")))
        self.by_hash: Dict[str, List[dict]] = {}
        for r in rows:
            self.by_hash.setdefault(md5[r["clip"]], []).append(r)

    def analyze(self, video_path: str, deadline: Deadline, report: Report) -> AnalysisResult:
        rows = self.by_hash.get(_md5(video_path), [])
        vs = []
        for r in rows:
            if r["label"] != "violation":
                continue
            t = float(r["label_ts_sec"])
            vs.append(Violation(
                type_id=r["type_id"], type=kb.type_name(r["type_id"]), timecode_sec=t,
                description=kb.describe(r["type_id"], f"Эталонная разметка клипа {r['clip']}"),
                evidence_ts=[t - 1, t, t + 1], confidence=1.0, source="oracle", refs=kb.refs(r["type_id"]),
            ))
        return AnalysisResult(vs, {"analyzer": self.name, "matched_clips": [r["clip"] for r in rows]})
