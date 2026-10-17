#!/usr/bin/env python3
"""Scoreboard for every check-in.

Three ways to get predictions, one way to score them:

  # 1) run an analyzer in-process (no server), save predictions, score
  python tools/eval.py --local pipeline --videos /data/open/videos --out runs/vlm7b

  # 2) drive the live service like the jury does (also a latency / soak test)
  python tools/eval.py --api https://team.example.kz --key $KEY --videos /data/open/videos --out runs/live --parallel 5

  # 3) score predictions that are already on disk
  python tools/eval.py --preds runs/vlm7b

  # smoke-test arbitrary files (no labels): status, latency, schema only
  python tools/eval.py --api http://127.0.0.1:8000 --key $KEY --files a.mov b.mov --out runs/smoke

The board line printed at the end is what goes into the check-in fields on the sprint board.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ANN = ROOT / "data" / "train_annotations.csv"
DUPES = {"X-47", "X-48"}  # byte-identical copies of X-4 / X-5


# ---------------------------------------------------------------- annotations
def load_annotations(path: Path = ANN):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    for r in rows:
        r["label_ts_sec"] = float(r["label_ts_sec"]) if r["label_ts_sec"] else None
    return rows


def duration_of(path: str):
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        return float(out)
    except Exception:
        return None


# ---------------------------------------------------------------- schema check
def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def check_schema(r, duration=None):
    """Errors that would make the jury's automatic scorer misread the answer."""
    errs = []
    if not isinstance(r, dict):
        return ["response is not a JSON object"]
    if not isinstance(r.get("job_id"), str) or not r.get("job_id"):
        errs.append("job_id missing or not a string")
    if r.get("status") != "done":
        errs.append(f"status is {r.get('status')!r}, not 'done'")
        return errs
    if r.get("verdict") not in ("violation", "no_violation"):
        errs.append(f"verdict {r.get('verdict')!r} not in violation/no_violation")
    vs = r.get("violations")
    if not isinstance(vs, list):
        return errs + ["violations is not a list"]
    if (r.get("verdict") == "violation") != (len(vs) > 0):
        errs.append("verdict and violations list disagree")
    for i, v in enumerate(vs):
        p = f"violations[{i}]"
        if not isinstance(v, dict):
            errs.append(f"{p} is not an object")
            continue
        if not isinstance(v.get("type"), str) or not v["type"].strip():
            errs.append(f"{p}.type empty")
        if not _is_num(v.get("timecode_sec")) or v["timecode_sec"] < 0:
            errs.append(f"{p}.timecode_sec not a non-negative number")
        elif duration and v["timecode_sec"] > duration + 0.5:
            errs.append(f"{p}.timecode_sec {v['timecode_sec']} beyond video end {duration:.1f}")
        if not isinstance(v.get("description"), str) or not v["description"].strip():
            errs.append(f"{p}.description empty")
        fr = v.get("evidence_frames")
        if not isinstance(fr, list) or not (1 <= len(fr) <= 3):
            errs.append(f"{p}.evidence_frames must have 1-3 items")
            continue
        for j, f in enumerate(fr):
            q = f"{p}.evidence_frames[{j}]"
            if not isinstance(f, dict) or not _is_num(f.get("timestamp_sec")):
                errs.append(f"{q}.timestamp_sec not a number")
            b = f.get("image_base64") if isinstance(f, dict) else None
            if not isinstance(b, str) or b.startswith("data:"):
                errs.append(f"{q}.image_base64 must be bare base64 (no data: prefix)")
                continue
            try:
                raw = base64.b64decode(b, validate=True)
                if raw[:2] != b"\xff\xd8":
                    errs.append(f"{q} is not a JPEG")
            except Exception:
                errs.append(f"{q}.image_base64 does not decode")
    return errs


# ---------------------------------------------------------------- prediction sources
def run_local(analyzer_name, rows, videos_dir, out):
    from cashaudit.analyzers import Deadline, make_analyzer
    from cashaudit.config import settings
    from cashaudit.schema import to_api_json
    from cashaudit.video import frame_at, jpeg_b64

    an = make_analyzer(analyzer_name, 0)
    timing = {}
    for r in rows:
        path = os.path.join(videos_dir, r["file_in_zip"])
        t0 = time.time()
        try:
            res = an.analyze(path, Deadline(time.time() + settings.job_budget_sec), lambda p: None)
            frames = {}
            for v in res.violations:
                for t in v.evidence_ts:
                    frames.setdefault(round(t, 2), jpeg_b64(frame_at(path, t)))
            resp = to_api_json(f"local-{r['clip']}", res, frames, extra=True)
        except Exception as e:
            resp = {"job_id": f"local-{r['clip']}", "status": "failed", "error": str(e)}
        timing[r["clip"]] = round(time.time() - t0, 2)
        (out / f"{r['clip']}.json").write_text(json.dumps(resp, ensure_ascii=False), encoding="utf-8")
        print(f"  {r['clip']:6s} {resp.get('status'):7s} {resp.get('verdict', ''):13s} {timing[r['clip']]:6.1f}s", flush=True)
    (out / "_timing.json").write_text(json.dumps(timing))


def run_api(base, key, items, out, parallel=1, timeout=330.0):
    import httpx

    base = base.rstrip("/")
    timing = {}

    def one(item):
        name, path = item
        t0 = time.time()
        with httpx.Client(timeout=120) as c:
            try:
                with open(path, "rb") as fh:
                    r = c.post(f"{base}/api/v1/analyze", headers={"X-API-Key": key},
                               files={"file": (os.path.basename(path), fh, "video/quicktime")})
                if r.status_code != 200:
                    return name, {"status": "failed", "error": f"POST {r.status_code}: {r.text[:200]}"}, time.time() - t0
                jid = r.json()["job_id"]
                while time.time() - t0 < timeout:
                    time.sleep(2)
                    g = c.get(f"{base}/api/v1/jobs/{jid}", headers={"X-API-Key": key})
                    body = g.json()
                    if body.get("status") in ("done", "failed"):
                        return name, body, time.time() - t0
                return name, {"job_id": jid, "status": "timeout"}, time.time() - t0
            except Exception as e:
                return name, {"status": "failed", "error": repr(e)}, time.time() - t0

    with ThreadPoolExecutor(max_workers=max(1, parallel)) as ex:
        for name, body, dt in ex.map(one, items):
            timing[name] = round(dt, 2)
            (out / f"{name}.json").write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
            late = "  OVER 300 s" if dt > 300 else ""
            print(f"  {name:12s} {body.get('status', '?'):8s} {body.get('verdict', ''):13s} {dt:6.1f}s{late}", flush=True)
    prev = {}
    tp = out / "_timing.json"
    if tp.exists():
        prev = json.loads(tp.read_text())
    prev.update(timing)
    tp.write_text(json.dumps(prev))


# ---------------------------------------------------------------- scoring
def score(rows, preds_dir: Path, videos_dir=None):
    from cashaudit import kb

    timing = {}
    if (preds_dir / "_timing.json").exists():
        timing = json.loads((preds_dir / "_timing.json").read_text())
    per_clip, schema_errs = [], {}
    for r in rows:
        f = preds_dir / f"{r['clip']}.json"
        pred = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {"status": "missing"}
        dur = duration_of(os.path.join(videos_dir, r["file_in_zip"])) if videos_dir else None
        errs = check_schema(pred, dur)
        if errs:
            schema_errs[r["clip"]] = errs
        pv = pred.get("verdict") if pred.get("status") == "done" else None
        dt, type_ok, closest = None, None, None
        if r["label"] == "violation" and pv == "violation" and pred.get("violations"):
            closest = min(pred["violations"], key=lambda v: abs(float(v.get("timecode_sec", 1e9)) - r["label_ts_sec"]))
            dt = abs(float(closest["timecode_sec"]) - r["label_ts_sec"])
            tid = closest.get("type_id") or kb.type_id_by_name(closest.get("type", ""))
            type_ok = (tid == r["type_id"]) if r["type_id"] else None
        per_clip.append({
            "clip": r["clip"], "label": r["label"], "label_ts": r["label_ts_sec"], "type_id": r["type_id"],
            "status": pred.get("status"), "pred": pv, "correct": pv == r["label"],
            "n_viol": len(pred.get("violations") or []),
            "pred_ts": [round(float(v["timecode_sec"]), 1) for v in pred.get("violations") or [] if _is_num(v.get("timecode_sec"))],
            "pred_types": [v.get("type_id") or kb.type_id_by_name(v.get("type", "")) or "?" for v in pred.get("violations") or []],
            "dt": dt, "type_ok": type_ok, "runtime": timing.get(r["clip"]),
        })

    pos = [c for c in per_clip if c["label"] == "violation"]
    neg = [c for c in per_clip if c["label"] == "no_violation"]
    dts = [c["dt"] for c in pos if c["dt"] is not None]
    typed = [c for c in pos if c["type_ok"] is not None]
    rts = [c["runtime"] for c in per_clip if c["runtime"] is not None]
    board = {
        "acc": f"{sum(c['correct'] for c in per_clip)}/{len(per_clip)}",
        "miss": f"{sum(c['pred'] != 'violation' for c in pos)}/{len(pos)}",
        "fa": f"{sum(c['pred'] == 'violation' for c in neg)}/{len(neg)}",
        "dt": f"{statistics.median(dts):.1f}" if dts else "n/a",
        "dt_le_2s": f"{sum(d <= 2 for d in dts)}/{len(dts)}" if dts else "n/a",
        "type": f"{sum(c['type_ok'] for c in typed)}/{len(typed)}" if typed else "n/a",
        "rt": f"{max(rts):.0f}" if rts else "n/a",
        "unanswered": sum(c["status"] != "done" for c in per_clip),
        "schema_errors": sum(len(v) for v in schema_errs.values()),
    }

    by_type = {}
    for c in pos:
        t = by_type.setdefault(c["type_id"] or "?", {"n": 0, "hit": 0, "dt": [], "type_ok": 0})
        t["n"] += 1
        t["hit"] += c["pred"] == "violation"
        if c["dt"] is not None:
            t["dt"].append(c["dt"])
        t["type_ok"] += bool(c["type_ok"])
    return board, per_clip, by_type, schema_errs


def print_report(board, per_clip, by_type, schema_errs, out: Path):
    from cashaudit import kb

    print("\nPer clip")
    print(f"  {'clip':6s} {'label':12s} {'t':>5s}  {'pred':12s} {'pred t':18s} {'types':14s} {'|dt|':>5s} type  run")
    for c in per_clip:
        mark = " " if c["correct"] else "x"
        print(f"{mark} {c['clip']:6s} {c['label']:12s} {('' if c['label_ts'] is None else int(c['label_ts'])):>5}  "
              f"{str(c['pred'] or c['status']):12s} {str(c['pred_ts'])[:18]:18s} {','.join(c['pred_types'])[:14]:14s} "
              f"{'' if c['dt'] is None else round(c['dt'], 1):>5} {'' if c['type_ok'] is None else ('ok' if c['type_ok'] else 'NO'):4s} "
              f"{'' if c['runtime'] is None else c['runtime']}")
    print("\nPer violation type (training clips)")
    for tid in sorted(by_type):
        t = by_type[tid]
        med = f"{statistics.median(t['dt']):.1f}" if t["dt"] else "-"
        print(f"  {tid:4s} hit {t['hit']}/{t['n']}  median|dt| {med:>4s}  type ok {t['type_ok']}/{t['n']}  {kb.type_name(tid)[:60]}")
    if schema_errs:
        print("\nSchema problems (fix these first: the jury scores automatically)")
        for clip, errs in schema_errs.items():
            for e in errs[:4]:
                print(f"  {clip}: {e}")
    line = (f"Video acc {board['acc']} · Missed {board['miss']} · False alarms {board['fa']} · Median |dt| {board['dt']} s "
            f"(≤2 s: {board['dt_le_2s']}) · Type acc {board['type']} · Longest clip {board['rt']} s · "
            f"Unanswered {board['unanswered']} · Schema errors {board['schema_errors']}")
    print("\nBOARD  " + line)
    (out / "scoreboard.json").write_text(json.dumps({"board": board, "per_clip": per_clip}, ensure_ascii=False, indent=1))
    (out / "scoreboard.md").write_text("**" + line + "**\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", help="directory with <clip>.json predictions to score")
    ap.add_argument("--local", help="run this analyzer in-process (stub | oracle | pipeline)")
    ap.add_argument("--api", help="base URL of the running service")
    ap.add_argument("--key", default=os.getenv("API_KEY", ""), help="X-API-Key for --api")
    ap.add_argument("--videos", help="directory with the training .mov files (open/videos)")
    ap.add_argument("--files", nargs="*", help="arbitrary videos for a smoke test (no labels)")
    ap.add_argument("--out", help="where to write predictions (default: --preds)")
    ap.add_argument("--clips", help="comma-separated subset, e.g. X-1,X-2,X-13")
    ap.add_argument("--no-dupes", action="store_true", help="drop X-47/X-48 (copies of X-4/X-5)")
    ap.add_argument("--labels", help="annotations CSV (default data/train_annotations.csv), e.g. our own clips")
    ap.add_argument("--parallel", type=int, default=1, help="concurrent uploads in --api mode")
    args = ap.parse_args()

    out = Path(args.out or args.preds or "runs/latest")
    out.mkdir(parents=True, exist_ok=True)

    if args.files:
        if not args.api:
            ap.error("--files needs --api")
        run_api(args.api, args.key, [(Path(f).stem, f) for f in args.files], out, args.parallel)
        for f in args.files:
            body = json.loads((out / f"{Path(f).stem}.json").read_text(encoding="utf-8"))
            errs = check_schema(body, duration_of(f))
            print(f"  {Path(f).stem}: {'schema OK' if not errs else '; '.join(errs[:3])}")
        return

    rows = load_annotations(Path(args.labels)) if args.labels else load_annotations()
    if args.clips:
        keep = set(args.clips.split(","))
        rows = [r for r in rows if r["clip"] in keep]
    if args.no_dupes:
        rows = [r for r in rows if r["clip"] not in DUPES]

    if args.local:
        if not args.videos:
            ap.error("--local needs --videos")
        run_local(args.local, rows, args.videos, out)
    elif args.api:
        if not args.videos:
            ap.error("--api needs --videos (or use --files)")
        run_api(args.api, args.key, [(r["clip"], os.path.join(args.videos, r["file_in_zip"])) for r in rows], out, args.parallel)
    elif not args.preds:
        ap.error("give --preds, --local or --api")

    print_report(*score(rows, out, args.videos), out)


if __name__ == "__main__":
    main()
