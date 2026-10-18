#!/usr/bin/env python3
"""Offline what-if for the last decision-making policy: re-score saved runs with stricter rules for keeping a VLM
violation, without touching the GPU. Standard library only (runs on the VM host while an evaluation is running).

Each saved clip result (local runs with DEBUG_EVENTS=1) holds the final violations (confidence = the verifier's P(yes)
for VLM ones) and every candidate before fusion (meta.candidates: scan windows + rules on VLM events + CV events).
A policy only removes final violations; the clip's verdict is "violation" iff anything is left.

    python3 deploy/experiments/fusion_sim.py                       # default runs below
    python3 deploy/experiments/fusion_sim.py var/runs/x/both=data/train_annotations.csv ...

Policies (all leave CV "events" and "fused" violations alone):
    base         as shipped
    p80/p90/p95  VLM violation needs verifier P(yes) >= 0.80 / 0.90 / 0.95
    win2         VLM violation needs >= 2 candidates of its type within +-6 s (two windows or both prompts agree)
    win2A        win2 only for the scene-A order types T08 T09 T11 T12 T20
    scene        VLM violation's type must belong to the scene the pipeline used (meta.scene)
    win2+scene   both
    win2@new     win2, but only when the CV layer could not register the camera to the training room (meta.cv.roi
    win2A@new    .registered is false): a new camera/room. Training clips and their shifted copies all register.
"""
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path.cwd() if (Path.cwd() / "data/rules_kb.json").exists() else Path(__file__).resolve().parents[2]  # run from the repo folder
KB = json.loads((ROOT / "data/rules_kb.json").read_text(encoding="utf-8"))
SCENE = {t["id"]: t["scene"] for t in KB["types"]}
NAME2ID = {t["name"]: t["id"] for t in KB["types"]}
ORDER_A = {"T08", "T09", "T11", "T12", "T20"}
WIN = 6.0
DEFAULT = [("var/runs/combo2/both", "data/train_annotations.csv"), ("var/runs/combo4/both", "data/train_annotations.csv"),
           ("var/runs/own-1-a", "var/own/labels_a.csv"), ("var/runs/own-1-b", "var/own/labels_b.csv")]


def tid(v):
    return v.get("type_id") or NAME2ID.get(v.get("type"), "T00")


def support(v, cands):
    return sum(1 for c in cands if c.get("type_id") == tid(v) and abs((c.get("timecode_sec") or 0) - v["timecode_sec"]) <= WIN)


def unregistered(meta):
    return ((meta.get("cv") or {}).get("roi") or {}).get("registered") is False


def keep(policy, v, cands, meta):
    if v.get("source") in ("events", "fused"):
        return True
    if policy.endswith("@new"):
        if not unregistered(meta):
            return True
        policy = policy[:-4]
    p = v.get("confidence") or 0
    if policy == "base":
        return True
    if policy in ("p80", "p90", "p95"):
        return p >= int(policy[1:]) / 100
    ok_win = support(v, cands) >= 2
    ok_scene = not meta.get("scene") or SCENE.get(tid(v)) in (None, meta["scene"])
    if policy == "win2":
        return ok_win
    if policy == "win2A":
        return ok_win or tid(v) not in ORDER_A
    if policy == "scene":
        return ok_scene
    if policy == "win2+scene":
        return ok_win and ok_scene
    raise ValueError(policy)


POLICIES = ["base", "p80", "p90", "p95", "win2", "win2A", "scene", "win2+scene", "win2@new", "win2A@new"]


def score(run, labels, policy):
    rows = {r["clip"]: r for r in csv.DictReader(open(labels, encoding="utf-8"))}
    acc = miss = fa = npos = nneg = typed = type_ok = 0
    dts, flips = [], []
    for clip, r in rows.items():
        f = Path(run) / f"{clip}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        meta = d.get("meta") or {}
        cands = meta.get("candidates") or []
        vs = [v for v in d.get("violations") or [] if keep(policy, v, cands, meta)]
        base_v = "violation" if d.get("violations") else "no_violation"
        got = "violation" if vs else "no_violation"
        if got != base_v:
            flips.append(f"{clip}:{'+' if got == r['label'] else '-'}")
        acc += got == r["label"]
        if r["label"] == "violation":
            npos += 1
            miss += got != "violation"
            if vs and r.get("label_ts_sec"):
                near = min(vs, key=lambda v: abs(v["timecode_sec"] - float(r["label_ts_sec"])))
                dts.append(abs(near["timecode_sec"] - float(r["label_ts_sec"])))
                typed += 1
                type_ok += tid(near) == r["type_id"]
        else:
            nneg += 1
            fa += got == "violation"
    if not npos and not nneg:
        return None
    n = npos + nneg
    md = f"{statistics.median(dts):.1f}" if dts else "-"
    return (acc, n, miss, npos, fa, nneg, type_ok, typed, md, flips)


def main():
    pairs = [tuple(a.split("=", 1)) for a in sys.argv[1:]] or DEFAULT
    has_cands = False
    for run, labels in pairs:
        if not Path(run).is_dir() or not Path(labels).exists():
            print(f"-- {run}: missing (run dir or {labels})"); continue
        metas = [(json.loads(f.read_text(encoding="utf-8")).get("meta") or {}) for f in Path(run).glob("*.json") if not f.name.startswith("_") and f.name != "scoreboard.json"]
        print(f"-- {run}   camera not registered on {sum(unregistered(m) for m in metas)}/{len(metas)} clips")
        print("   (acc · missed · false alarms · type · median |dt|; flips vs base: + fixed, - broken)")
        for pol in POLICIES:
            s = score(run, labels, pol)
            if s is None:
                print("   no clips matched the labels"); break
            acc, n, miss, npos, fa, nneg, tok, typed, md, flips = s
            print(f"   {pol:11} {acc:2}/{n} · {miss:2}/{npos} · {fa:2}/{nneg} · {tok}/{typed} · {md:>4} s  {' '.join(flips)}")
        any_c = any((json.loads(f.read_text(encoding='utf-8')).get('meta') or {}).get('candidates') for f in Path(run).glob('*.json') if not f.name.startswith('_'))
        has_cands = has_cands or any_c
        if not any_c:
            print("   (no meta.candidates in this run: win2 policies drop every VLM violation here, ignore them)")


if __name__ == "__main__":
    main()
