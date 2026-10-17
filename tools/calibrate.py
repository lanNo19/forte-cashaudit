#!/usr/bin/env python3
"""Suggest per-type confidence floors from an eval run, so the clean clips come back clean.

    python tools/eval.py --local pipeline --videos /path/open/videos --out runs/try3     # saves confidences
    python tools/calibrate.py --preds runs/try3                                         # prints the table
    python tools/calibrate.py --preds runs/try3 --write                                 # writes data/thresholds.json

For each violation type it collects
- true hits: predictions of that type within ±MATCH_SEC of the label on clips labelled with that type;
- false alarms: predictions of that type on clean clips, or far from the label / on clips of another type.
The suggested floor is just above the highest false-alarm confidence, and the table shows what recall that
costs. With 1-3 examples per type this is a sanity check, not a fit: read the table before writing it.
Needs predictions with extra fields (eval.py --local, or the API with API_EXTRA_FIELDS=true).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MATCH_SEC = 5.0


def collect(preds_dir: Path, ann_path: Path):
    rows = list(csv.DictReader(open(ann_path, encoding="utf-8")))
    hits, alarms = {}, {}
    for r in rows:
        f = preds_dir / f"{r['clip']}.json"
        if not f.exists():
            continue
        pred = json.loads(f.read_text(encoding="utf-8"))
        label_t = float(r["label_ts_sec"]) if r["label_ts_sec"] else None
        for v in pred.get("violations") or []:
            tid, conf = v.get("type_id"), v.get("confidence")
            if tid is None or conf is None:
                raise SystemExit(f"{f.name}: no type_id/confidence. Re-run eval.py --local (or API_EXTRA_FIELDS=true).")
            good = r["label"] == "violation" and tid == r["type_id"] and abs(float(v["timecode_sec"]) - label_t) <= MATCH_SEC
            (hits if good else alarms).setdefault(tid, []).append((float(conf), r["clip"]))
    return rows, hits, alarms


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--annotations", default=str(ROOT / "data" / "train_annotations.csv"))
    ap.add_argument("--out", default=str(ROOT / "data" / "thresholds.json"))
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    rows, hits, alarms = collect(Path(args.preds), Path(args.annotations))
    positives = {}
    for r in rows:
        if r["label"] == "violation" and r["type_id"]:
            positives[r["type_id"]] = positives.get(r["type_id"], 0) + 1

    floors = {}
    print(f"{'type':5s} {'pos':>3s} {'hits':>4s} {'alarms':>6s} {'max alarm':>9s} {'floor':>6s} {'recall@floor':>12s}  alarm clips")
    for tid in sorted(set(hits) | set(alarms) | set(positives)):
        h, a = hits.get(tid, []), alarms.get(tid, [])
        max_a = max((c for c, _ in a), default=None)
        floor = round(min(0.99, max_a + 0.01), 2) if max_a is not None else None
        kept = sum(1 for c, _ in h if floor is None or c >= floor)
        rec = f"{kept}/{positives.get(tid, 0)}"
        if floor is not None:
            floors[tid] = floor
        print(f"{tid:5s} {positives.get(tid, 0):>3d} {len(h):>4d} {len(a):>6d} {'' if max_a is None else f'{max_a:.2f}':>9s} "
              f"{'' if floor is None else f'{floor:.2f}':>6s} {rec:>12s}  {', '.join(sorted({c for _, c in a}))[:60]}")
    if args.write:
        Path(args.out).write_text(json.dumps({"floors": floors, "source": str(args.preds)}, indent=1))
        print(f"wrote {len(floors)} floors to {args.out}")
    else:
        print("dry run: add --write to save these floors")


if __name__ == "__main__":
    main()
