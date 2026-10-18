#!/usr/bin/env python3
"""Per-clip side by side of saved runs (local or API), standard library only: runs on the VM host.

    python3 deploy/experiments/compare_runs.py data/train_annotations.csv var/runs/soak5 var/runs/en1/both var/runs/tt1
    NOSCAN_A=1 python3 ...   also drop every "vlm" violation of the ordering types (T08 T09 T11 T12 T20): those
                             types switched off (rule findings on VLM events are labelled "vlm" too). Local runs
                             only: API runs carry no source and are shown as they are

Only clips present in the LAST run are listed. A cell is "ok"/"XX" (verdict right/wrong) then the types@seconds.
"""
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path.cwd() if (Path.cwd() / "data/rules_kb.json").exists() else Path(__file__).resolve().parents[2]
KB = json.loads((ROOT / "data/rules_kb.json").read_text(encoding="utf-8"))
NAME2ID = {t["name"]: t["id"] for t in KB["types"]}
ORDER_A = {"T08", "T09", "T11", "T12", "T20"}
NOSCAN = os.getenv("NOSCAN_A") == "1"


def cell(run, clip, label):
    f = Path(run) / f"{clip}.json"
    if not f.exists():
        return "-", None
    vs = json.loads(f.read_text(encoding="utf-8")).get("violations") or []
    tid = lambda v: v.get("type_id") or NAME2ID.get(v.get("type"), "T00")  # noqa: E731
    if NOSCAN:
        vs = [v for v in vs if not (v.get("source") == "vlm" and tid(v) in ORDER_A)]
    got = "violation" if vs else "no_violation"
    ok = got == label
    return f"{'ok' if ok else 'XX'} " + (",".join(f"{tid(v)}@{float(v['timecode_sec']):.0f}" for v in vs) or "clean"), ok


def main():
    labels, runs = sys.argv[1], sys.argv[2:]
    rows = list(csv.DictReader(open(labels, encoding="utf-8")))
    last = Path(runs[-1])
    rows = [r for r in rows if (last / f"{r['clip']}.json").exists()]
    print(f"{'clip':8} {'label':14} " + " | ".join(Path(r).name if Path(r).name != "both" else Path(r).parent.name for r in runs))
    totals = [0] * len(runs)
    for r in rows:
        want = f"{r['label']}" + (f" {r.get('type_id', '')}@{r.get('label_ts_sec', '')}" if r["label"] == "violation" else "")
        cells = []
        for i, run in enumerate(runs):
            text, ok = cell(run, r["clip"], r["label"])
            totals[i] += bool(ok)
            cells.append(f"{text:28}")
        print(f"{r['clip']:8} {want:14} " + " | ".join(cells))
    print(f"right    {'':14} " + " | ".join(f"{t}/{len(rows):<25}" for t in totals))


if __name__ == "__main__":
    main()
