#!/usr/bin/env python3
"""Per-clip report of one eval run (local mode): scenes, final violations and where each one came from.

    python3 tools/run_report.py var/runs/combo1/both            # all clips
    python3 tools/run_report.py var/runs/combo1/both --wrong    # only false alarms and misses

Columns: clip, label scene, VLM scene (meta.scene_vlm, else meta.scene), CV scene (meta.cv.scene), label, final violations as
type@second/source/confidence. Debug runs (DEBUG_EVENTS=1) also list the candidates before fusion; a candidate's
confidence hints at its path: rules give at most 0.8 x event confidence, the VLM scan gives its own number (usually 0.9+).
Standard library only, so it runs on the VM host.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--labels", default=str(ROOT / "data/train_annotations.csv"))
    ap.add_argument("--wrong", action="store_true", help="only false alarms and misses")
    a = ap.parse_args()
    labels = {r["clip"]: r for r in csv.DictReader(open(a.labels, encoding="utf-8"))}
    run = Path(a.run)
    files = sorted(run.glob("*.json"), key=lambda p: (len(p.stem), p.stem))
    files = [f for f in files if not f.stem.startswith("_") and f.stem not in ("scoreboard",)]
    if not files:
        print(f"no clip results in {run}"); return 1
    scene_bad, rows = 0, 0
    print(f"{'clip':6} {'lab':3} {'vlm':3} {'cv':3}  {'label':12} {'want':9} final violations")
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        lab = labels.get(f.stem, {})
        meta = d.get("meta") or {}
        cv = meta.get("cv") or {}
        got = "violation" if d.get("violations") else "no_violation"
        wrong = got != lab.get("label")
        if a.wrong and not wrong:
            continue
        rows += 1
        # since PR #20 meta.scene is the scene actually used (CV's when it disagrees) and meta.scene_vlm the VLM's guess
        vlm_s, cv_s = meta.get("scene_vlm") or meta.get("scene") or "-", cv.get("scene") or "-"
        if lab.get("scene") and vlm_s not in ("-", lab["scene"]):
            scene_bad += 1
        want = (lab.get("type_id") or "") + (f"@{lab['label_ts_sec']}" if lab.get("label_ts_sec") else "")
        vs = " ".join(f"{v.get('type_id', '?')}@{v['timecode_sec']:.0f}/{v.get('source', '?')}/{v.get('confidence', 0):.2f}"
                      for v in d.get("violations", []))
        mark = ("FA " if got == "violation" else "MISS ") if wrong else ""
        print(f"{f.stem:6} {lab.get('scene', '?'):3} {vlm_s:3} {cv_s:3}  {lab.get('label', '?'):12} {want:9} {mark}{vs}")
        cands = meta.get("candidates")
        if cands and (wrong or not a.wrong):
            hi = [c for c in cands if (c.get("confidence") or 0) > 0.8]
            lo = [c for c in cands if (c.get("confidence") or 0) <= 0.8]
            fmt = lambda cs: " ".join(f"{c['type_id']}@{(c.get('timecode_sec') or 0):.0f}/{c.get('source')}/{(c.get('confidence') or 0):.2f}" for c in cs)
            if hi:
                print(f"{'':7}candidates >0.8 (likely VLM scan):  {fmt(hi)}")
            if lo:
                print(f"{'':7}candidates <=0.8 (rules, or a low scan): {fmt(lo)}")
    print(f"\n{rows} clips shown; VLM scene differs from the label on {scene_bad} of them")
    return 0


if __name__ == "__main__":
    sys.exit(main())
