#!/usr/bin/env python3
"""Two offline checks on files already on the VM (no GPU, standard library only), 18 Oct night:

1. Timecodes (15 points): per violation type, how far the predicted second lands from the label, with its sign, over
   the recent full runs. A steady offset for a type (always ~3 s late, say) is fixable with a per-type shift.
2. CV states under load (Amina, X-11/X-20/X-28/X-31): for every stored job of those clips, did the 60 s state budget
   run out ("budget" errors) or did requests fail, and what did the state crops answer.

    python3 deploy/experiments/dig.py            from the repo folder on the VM
"""
import json
import statistics
from collections import defaultdict
from pathlib import Path

RUNS = ["soak3", "soak1", "combo3/both", "combo2/both", "combo1/both", "logprob1/both", "maxv1/both"]

print("== 1. timecodes: signed offset of the closest same-type prediction (pred - label, s)")
offsets = defaultdict(list)
for run in RUNS:
    p = Path("var/runs") / run / "scoreboard.json"
    if not p.exists():
        continue
    sb = json.loads(p.read_text(encoding="utf-8"))
    b = sb["board"]
    print(f"-- {run}: acc {b['acc']}, median |dt| {b['dt']} s, <=2 s {b.get('dt_le_2s')}")
    for c in sb["per_clip"]:
        if c["label"] != "violation" or c["pred"] != "violation":
            continue
        same = [t for typ, t in zip(c["pred_types"], c["pred_ts"]) if typ == c["type_id"]]
        off = round(min(same, key=lambda t: abs(t - c["label_ts"])) - c["label_ts"], 1) if same else None
        if off is not None:
            offsets[c["type_id"]].append(off)
        preds = " ".join(f"{typ}@{t:g}" for typ, t in zip(c["pred_types"], c["pred_ts"]))
        print(f"   {c['clip']:5} {c['type_id']}@{c['label_ts']:g}  same-type offset {off if off is not None else '-':>5}  preds {preds}")
print("-- per type over all runs above: offsets (s), median")
for tid in sorted(offsets):
    o = offsets[tid]
    print(f"   {tid}: n {len(o):2}  median {statistics.median(o):+5.1f}  {sorted(o)}")

print("\n== 2. CV state answers for X-11, X-20, X-28, X-31 in every stored job (~/jobs-meta.jsonl)")
f = Path.home() / "jobs-meta.jsonl"
if not f.exists():
    print("   ~/jobs-meta.jsonl missing (night2.sh step 0 writes it)")
else:
    for line in f.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if not any(r["file"].endswith(f"-{n}.mov") for n in (11, 20, 28, 31)):
            continue
        m = r.get("meta") or {}
        st = (m.get("cv") or {}).get("states") or {}
        v = st.get("vlm") or {}
        ans = " ".join(f"{k}:{a}" for k, a in (st.get("answers") or {}).items())
        vs = " ".join(f"{x.get('type_id')}@{x['timecode_sec']:.0f}" for x in r.get("violations", []))
        print(f"   {r['utc']} {r['file']:16} wait {r['wait_sec']:5.1f} total {r['total_sec']:5.1f} | samples {v.get('samples')} "
              f"errors {v.get('errors')} states {v.get('sec')} s | {r.get('verdict')} {vs}")
        if ans:
            print(f"      {ans}")
