#!/usr/bin/env bash
# Amina's rerun of cv/states after the CV false-alarm fixes (cacaf11), 17 Oct. Needs the GPU image
# (API_DOCKERFILE=Dockerfile.gpu) with origin/cv/states merged and the api rebuilt (--no-deps keeps vLLM up).
# Run from the repo folder on the VM, in the background:
#   nohup deploy/experiments/cv8b.sh > ~/cv8b.out 2>&1 &      then: tail -f ~/cv8b.out
# Output: ~/cv8b-runs.tgz (var/runs/cv2-* + logs) and the 4 BOARD lines + load result at the end of ~/cv8b.out.
set -uo pipefail
cd "$(dirname "$0")/../.."
LOGS=~/cv8b-logs; mkdir -p "$LOGS"
step() { echo; echo "== $(date +%H:%M:%S) $1"; }
dc() { docker compose exec -T "$@"; }
run() { local name=$1; shift; step "$name"; dc "$@" 2>&1 | tee "$LOGS/$name.log" | grep -E "^BOARD|Error|Traceback" ; }

step "versions and code in the api container"
dc api python3 -c "import torch, rfdetr; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
git log --oneline -1 origin/cv/states; git merge-base --is-ancestor origin/cv/states HEAD && echo "cv/states fixes merged" \
  || { echo "origin/cv/states is NOT merged into this checkout: merge it and rebuild the api first"; exit 1; }

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
deploy/vm.sh load 2>&1 | tee "$LOGS/load.log"
rm -rf var/runs/cv2-load && cp -r var/runs/load var/runs/cv2-load

tar czf ~/cv8b-runs.tgz var/runs/cv2-* -C ~ cv8b-logs
step "BOARD lines"
grep -H "^BOARD" "$LOGS"/*.log
echo; echo "packed: ~/cv8b-runs.tgz  (laptop: gcloud compute scp test-vm-a100:~/cv8b-runs.tgz . --zone us-central1-f)"
