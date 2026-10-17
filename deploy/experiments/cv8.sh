#!/usr/bin/env bash
# Amina's CV runs for #7 (camera geometry) + #8 (object states), 17 Oct. Needs the GPU image (API_DOCKERFILE=Dockerfile.gpu)
# and the cv/states code in the api container. Run from the repo folder on the VM, in the background:
#   nohup deploy/experiments/cv8.sh > ~/cv8.out 2>&1 &      then: tail -f ~/cv8.out
# Output: ~/cv8-runs.tgz (predictions + logs) and the BOARD lines at the end of ~/cv8.out.
set -uo pipefail
cd "$(dirname "$0")/../.."
LOGS=~/cv8-logs; mkdir -p "$LOGS"
step() { echo; echo "== $(date +%H:%M:%S) $1"; }
dc() { docker compose exec -T "$@"; }
run() { local name=$1; shift; step "$name"; dc "$@" 2>&1 | tee "$LOGS/$name.log" | grep -E "^BOARD|Error|Traceback" ; }

step "versions in the api container"
dc api python3 -c "import torch, rfdetr, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__)"

if [ "$(ls var/shifted/*.mov 2>/dev/null | wc -l)" -lt 42 ]; then
  step "camera-shifted copies of the 42 clips"
  dc api python3 tools/make_shift_set.py --videos var/videos --out var/shifted | tail -3
fi

# CV only: the VLM is pointed at a dead address on purpose (people + geometry)
for set in videos shifted; do
  run "cv-only-$set" -e VLM_BASE_URL=http://127.0.0.1:9/v1 -e VLM_WARMUP=0 api \
    python3 tools/eval.py --local pipeline --videos "var/$set" --out "var/runs/cv-only-$set"
done
# full pipeline with the VLM: this is where #8's object states run
for set in videos shifted; do
  run "cv-both-$set" -e VLM_MODE=both api python3 tools/eval.py --local pipeline --videos "var/$set" --out "var/runs/cv-both-$set"
done
# same, object states off, to see what #8 adds
run "cv-both-nostates" -e VLM_MODE=both -e CV_STATES=0 api \
  python3 tools/eval.py --local pipeline --videos var/videos --out var/runs/cv-both-nostates

step "load test: largest clip x5 through the API, slowest must be < 240 s"
deploy/vm.sh load 2>&1 | tee "$LOGS/load.log"

tar czf ~/cv8-runs.tgz var/runs/cv-only-* var/runs/cv-both-* var/runs/load -C ~ cv8-logs
step "BOARD lines"
grep -H "^BOARD" "$LOGS"/*.log
echo; echo "packed: ~/cv8-runs.tgz  (laptop: gcloud compute scp test-vm-a100:~/cv8-runs.tgz ~/Downloads/ --zone us-central1-f)"
