import json
import subprocess
import sys
from pathlib import Path

from cashaudit import kb
from cashaudit.fusion import fuse
from cashaudit.schema import Violation

ROOT = Path(__file__).resolve().parent.parent


def v(tid, t, conf, source):
    return Violation(type_id=tid, type=kb.type_name(tid), timecode_sec=t, description="x", evidence_ts=[t], confidence=conf, source=source)


def test_verifier_vetoes_visible_event_type_but_not_tiny_state():
    cands = [v("T09", 57, 0.8, "events"), v("T05", 14, 0.7, "events")]
    out = fuse(cands, verifier=lambda c: 0.1, floors={})  # VLM says "no" to everything
    assert [c.type_id for c in out] == ["T05"]  # key-in-lock stays with CV


def test_vlm_candidate_rescored_by_verifier():
    out = fuse([v("T08", 10, 0.7, "vlm")], verifier=lambda c: 0.3, floors={})
    assert out == []
    out = fuse([v("T08", 10, 0.7, "vlm")], verifier=lambda c: 0.9, floors={})
    assert out and out[0].confidence == 0.9


def test_per_type_floor_and_unseen_types():
    assert fuse([v("T09", 57, 0.7, "events")], floors={"T09": 0.75}) == []
    assert fuse([v("T18", 5, 0.8, "vlm")], floors={}) == []  # unseen type needs 0.85
    assert len(fuse([v("T18", 5, 0.9, "vlm")], floors={})) == 1


def test_merge_keeps_event_timecode():
    out = fuse([v("T09", 57, 0.7, "events"), v("T09", 60, 0.9, "vlm")], floors={})
    assert len(out) == 1 and out[0].timecode_sec == 57 and out[0].source == "fused" and out[0].confidence == 0.9


def test_calibrate_suggests_floor_above_false_alarms(tmp_path):
    def pred(viols):
        return {"job_id": "x", "status": "done", "verdict": "violation" if viols else "no_violation", "violations": viols}
    (tmp_path / "X-4.json").write_text(json.dumps(pred([{"type_id": "T09", "confidence": 0.9, "timecode_sec": 57, "type": "a"}])))
    (tmp_path / "X-5.json").write_text(json.dumps(pred([{"type_id": "T09", "confidence": 0.7, "timecode_sec": 40, "type": "a"}])))
    out = tmp_path / "thr.json"
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "calibrate.py"), "--preds", str(tmp_path), "--out", str(out), "--write"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert json.loads(out.read_text())["floors"]["T09"] == 0.71
