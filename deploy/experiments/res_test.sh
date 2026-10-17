#!/usr/bin/env bash
# Does a high-resolution video still get the VLM? The training clips are 464x848; a VLM window holds up to 32 frames,
# and Qwen3-VL spends one token per 32x32 pixels, so a 1080p frame is ~2,000 tokens and 32 of them overflow the
# server's 16,384-token context (VLM_MAX_LEN). Every window would then fail and only the CV layer would answer.
# The jury's last video comes from another camera, so this runs our raw phone clips (not the 848 px copies that
# deploy/vm.sh own makes) twice: at native resolution (VLM_MAX_SIDE=0, today's default) and with VLM_MAX_SIDE=848.
# Needs var/own_raw (raw clips) and var/own/labels.csv (written by deploy/vm.sh own). Run from the repo folder on the VM,
# when no other evaluation is running (two evaluations at once can run the GPU out of memory):
#   deploy/experiments/res_test.sh 2>&1 | tee ~/res.out
set -uo pipefail
cd "$(dirname "$0")/../.."
CLIPS=${CLIPS:-OWN-T02-violation-a,OWN-T02-clean-a,OWN-T09-violation-b,OWN-T09-clean-b,OWN-T17-violation-a,OWN-T19-violation-b}

echo "== raw clips: count, codec, width, height, frame rate"
docker compose exec -T api sh -c 'for f in var/own_raw/*.mov var/own_raw/*.mp4; do ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,width,height,r_frame_rate -of csv=p=0 "$f"; done' | sort | uniq -c
echo "(tokens per frame ~ width/32 x height/32; 32 frames must fit in 16,384 with the prompt: 464x848 ~ 384, 1080p ~ 2,000)"

echo "== labels for: $CLIPS"
docker compose exec -T -e CLIPS="$CLIPS" api python3 - <<'EOF'
import csv, os
from pathlib import Path
want = os.environ["CLIPS"].split(",")
raw = {p.stem: p.name for p in Path("var/own_raw").iterdir()}
rows = [r for r in csv.DictReader(open("var/own/labels.csv", encoding="utf-8")) if r["clip"] in want]
for r in rows:
    r["file_in_zip"] = raw[r["clip"]]
Path("var/runs").mkdir(exist_ok=True)
with open("var/runs/res-labels.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(len(rows), "clips:", ", ".join(r["file_in_zip"] for r in rows))
EOF

for side in 0 848; do
  echo; echo "== $(date +%H:%M:%S) UTC  VLM_MAX_SIDE=$side"
  docker compose exec -T -e VLM_MAX_SIDE=$side -e DEBUG_EVENTS=1 api python3 tools/eval.py --local pipeline \
    --videos var/own_raw --labels var/runs/res-labels.csv --out var/runs/res-$side 2>&1 | grep -E "^BOARD|^[x ] OWN|Traceback"
done

echo; echo "== per clip: native vs VLM_MAX_SIDE=848 vs the 848 px copy (own-1); 'vlm fails' = VLM requests that errored"
python3 - "$CLIPS" <<'EOF'
import json, sys
from pathlib import Path
def show(path):
    p = Path(path)
    if not p.exists():
        return "-"
    d = json.loads(p.read_text(encoding="utf-8"))
    m = d.get("meta") or {}
    w = m.get("warnings") or []
    fails = [x for x in w if "failed" in x or "error" in x.lower()]
    vs = " ".join(f"{v.get('type_id')}@{v['timecode_sec']:.0f}" for v in d.get("violations", []))
    return (f"{d.get('verdict', d.get('status'))} [{vs}] fps {m.get('fps_used')} {m.get('runtime_sec')} s, "
            f"vlm fails {len(fails)}/{m.get('n_windows', '?')} win" + (f" ({fails[0][:90]})" if fails else ""))
for c in sys.argv[1].split(","):
    print(c)
    print("  native :", show(f"var/runs/res-0/{c}.json"))
    print("  848    :", show(f"var/runs/res-848/{c}.json"))
    print("  copy   :", show(f"var/runs/own-1-{c[-1]}/{c}.json"))
EOF
