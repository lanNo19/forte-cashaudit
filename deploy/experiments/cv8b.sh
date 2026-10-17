#!/usr/bin/env bash
# Amina's rerun of cv/states with the CV false-alarm fixes (cacaf11), 17 Oct evening. On the VM, from the repo folder,
# after merging origin/cv/states:
#   nohup deploy/experiments/cv8b.sh > ~/cv8b.out 2>&1 &      then: tail -f ~/cv8b.out
# It first waits for a running `deploy/vm.sh modes` run (rebuilding the api container would kill it), rebuilds the api
# container only (vLLM keeps running), then: CV only and full pipeline on the originals and the shifted copies, load test.
# Output: ~/cv8b-runs.tgz (var/runs/cv2-* + logs) and the BOARD lines at the end of ~/cv8b.out.
set -uo pipefail
cd "$(dirname "$0")/../.."
LOGS=~/cv8b-logs; mkdir -p "$LOGS"
PORT=$(grep -E '^API_PORT=' .env 2>/dev/null | cut -d= -f2); PORT=${PORT:-80}
step() { echo; echo "== $(date +%H:%M:%S) $1"; }
dc() { docker compose exec -T "$@"; }
run() { local name=$1; shift; step "$name"; dc "$@" 2>&1 | tee "$LOGS/$name.log" | grep -E "^BOARD|Error|Traceback" ; }

if pgrep -f "deploy/vm.sh modes" >/dev/null; then
  step "waiting for the running deploy/vm.sh modes run to finish"
  while pgrep -f "deploy/vm.sh modes" >/dev/null; do sleep 30; done
fi

step "code: $(git log --oneline -1)"
git merge-base --is-ancestor origin/cv/states HEAD \
  || { echo "origin/cv/states ($(git log --oneline -1 origin/cv/states)) is NOT merged here: git fetch origin, merge it, run again"; exit 1; }
step "rebuild the api container (vLLM keeps running)"
docker compose up -d --build --no-deps api 2>&1 | tail -2
for _ in $(seq 1 90); do curl -s -m 3 "localhost:$PORT/healthz" 2>/dev/null | grep -q '"ok":true' && break; sleep 2; done
dc api python3 -c "import torch, rfdetr, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__)"

if [ "$(ls var/shifted/*.mov 2>/dev/null | wc -l)" -lt 42 ]; then
  step "camera-shifted copies of the 42 clips"
  dc api python3 tools/make_shift_set.py --videos var/videos --out var/shifted | tail -3
fi

# CV only: the VLM is pointed at a dead address on purpose
for set in videos shifted; do
  run "cv2-only-$set" -e VLM_BASE_URL=http://127.0.0.1:9/v1 -e VLM_WARMUP=0 api \
    python3 tools/eval.py --local pipeline --videos "var/$set" --out "var/runs/cv2-only-$set"
done
# full pipeline with the VLM; DEBUG_EVENTS=1 saves the raw CV and VLM events per clip
for set in videos shifted; do
  run "cv2-both-$set" -e VLM_MODE=both -e DEBUG_EVENTS=1 api \
    python3 tools/eval.py --local pipeline --videos "var/$set" --out "var/runs/cv2-both-$set"
done

step "load test: largest clip x5 through the API, slowest must be < 240 s"
LOAD_OUT=cv2-load deploy/vm.sh load 2>&1 | tee "$LOGS/load.log"

tar czf ~/cv8b-runs.tgz var/runs/cv2-* -C ~ cv8b-logs
step "BOARD lines"
grep -H "^BOARD" "$LOGS"/*.log
echo; echo "packed: ~/cv8b-runs.tgz  (laptop: gcloud compute scp test-vm-a100:~/cv8b-runs.tgz . --zone us-central1-f)"
