"""Order-type false alarms (T08 cash before ID, T12 bag without documents, ...) survive the verifier at P(yes) 0.98:
it sees only +-3 s around the candidate, where the ID or document shown earlier is not in view, and a small VLM tends
to say "yes". This asks the opposite question on a wider window (20 s before to 2 s after the candidate):
"before this moment, was the ID / document shown?" A violation is only real if that answer is "no".

Measures it on the final VLM violations of these types in saved runs (no pipeline rerun: one VLM request each).
Run inside the api container (it needs the VLM server and the videos), from the repo folder on the VM:
    docker compose exec -T api python3 - < ~/negate_test.py
"""
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, "/app")
from cashaudit.video import sample_frames  # noqa: E402
from cashaudit.vlm import SYSTEM, VLMClient  # noqa: E402

NEG = {
    "T08": "Before the client handed cash to the cashier (around t={t:.0f}s), did the client give the cashier or show an "
           "ID card, passport or other identity document?",
    "T12": "Before the cashier took the cash bag (around t={t:.0f}s), did the cash collector hand over or show a document, "
           "paper list or ID card to the cashier?",
    "T20": "Before handing the cash to the client (around t={t:.0f}s), did the cashier count the banknotes in front of the "
           "client (by hand or on the counting machine)?",
    "T11": "Before the client left (around t={t:.0f}s), had the counting of the client's cash already finished?",
}
RUNS = [("var/runs/combo2/both", "data/train_annotations.csv", "var/videos"),
        ("var/runs/own-1-a", "var/own/labels_a.csv", "var/own"), ("var/runs/own-1-b", "var/own/labels_b.csv", "var/own")]
vlm = VLMClient()


def p_yes(frames, question):
    prompt = f"{vlm._time_hint(frames)}\n{question}\nAnswer with exactly one word: yes or no."
    body = {"model": vlm.model, "temperature": 0.0, "max_tokens": 1, "logprobs": True, "top_logprobs": 10,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": vlm._media(frames) + [{"type": "text", "text": prompt}]}]}
    r = vlm.http.post(f"{vlm.base_url}/chat/completions", json=body)
    r.raise_for_status()
    top = r.json()["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    yes = sum(math.exp(x["logprob"]) for x in top if x["token"].strip().lower() in ("yes", "да"))
    no = sum(math.exp(x["logprob"]) for x in top if x["token"].strip().lower() in ("no", "нет"))
    return yes / (yes + no) if yes + no > 0 else None


def window(frames, t, before=20.0, after=2.0, n=12):
    w = [f for f in frames if t - before <= f[0] <= t + after]
    if len(w) > n:
        w = [w[round(i * (len(w) - 1) / (n - 1))] for i in range(n)]
    return w


totals = {}
for run, labels, vdir in RUNS:
    if not Path(run).is_dir():
        print(f"-- {run}: missing"); continue
    print(f"-- {run}")
    for r in csv.DictReader(open(labels, encoding="utf-8")):
        f = Path(run) / f"{r['clip']}.json"
        if not f.exists():
            continue
        vs = json.loads(f.read_text(encoding="utf-8")).get("violations") or []
        todo = [v for v in vs if v.get("source") == "vlm" and v.get("type_id") in NEG]
        if not todo:
            continue
        frames = sample_frames(str(Path(vdir) / r["file_in_zip"]), fps=2, max_side=848)
        drops = []
        for v in todo:
            p = p_yes(window(frames, v["timecode_sec"]), NEG[v["type_id"]].format(t=v["timecode_sec"]))
            v["p_neg"] = p
            print(f"   {r['clip']:22} {r['label']:12} {v['type_id']}@{v['timecode_sec']:.0f} p_yes={v.get('confidence', 0):.2f} "
                  f"p_neg={'-' if p is None else f'{p:.2f}'}")
        for thr in (0.5, 0.7):
            left = [v for v in vs if not (v in todo and v.get("p_neg") is not None and v["p_neg"] >= thr)]
            before, after = bool(vs), bool(left)
            if before != after:
                ok = (r["label"] == "violation") == after
                totals.setdefault(thr, []).append(f"{r['clip']}:{'+' if ok else '-'}")
print("\nverdict flips if a violation is dropped when p_neg >= threshold (+ fixed, - broken):")
for thr, flips in sorted(totals.items()):
    print(f"   {thr}: {' '.join(flips) or 'none'}")
