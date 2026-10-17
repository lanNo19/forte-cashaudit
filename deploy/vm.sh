#!/usr/bin/env bash
# Helpers for the A100 VM. Run from the repo folder on the VM: deploy/vm.sh <command>
#
#   status            containers, /healthz, GPU memory
#   prep <open.zip>   unpack the training clips into var/videos (names as eval.py expects, checksums checked)
#   eval <run>        all 42 clips through the running API, 5 in parallel; results in var/runs/<run>
#   modes [tag]       ML lead's comparison: eval-local with VLM_MODE violations, events, both; packs ~/<tag>.tgz
#                     (DEBUG_EVENTS=1 deploy/vm.sh modes <tag> also saves raw VLM events and candidates per clip)
#   own [tag]         our own phone clips: var/own_raw/{OWN-xxa/b clips, shotlog.csv} -> jury format, one BOARD line per
#                     phone (a, b), packed as ~/own-<tag>.tgz (tools/prepare_own.py explains the shot log)
#   load              jury worst case: the largest clip uploaded 5 times at once; slowest must be < 240 s
#   logs [15m]        errors, timeouts and warnings from the API log
#   pin               freeze the vLLM image to the digest that is running, switch to offline weights, restart
#   idle              is it safe to stop the VM now? (no job queued or running)
set -euo pipefail
cd "$(dirname "$0")/.."
# Variables set on the command line that should reach the evaluation inside the container, e.g.
#   DEBUG_EVENTS=1 deploy/vm.sh modes debug1      VLM_MODE / VLM_FPS / ... override .env for one run
# (read before .env is loaded, so only what you typed is passed)
PASS_ENV=$(env | grep -E '^(DEBUG_[A-Z_]*|VLM_[A-Z_]*|CV_[A-Z_]*|EVENTS_[A-Z_]*|FUSION_[A-Z_]*|OPEN_WORLD_CONF|VETO_[A-Z_]*)=' | cut -d= -f1 || true)
[ -f .env ] && { set -a; . ./.env; set +a; }
PORT=${API_PORT:-80}

in_api() {  # NO_DOCKER=1 runs the same command on the host (for testing this script without Docker)
  if [ -n "${NO_DOCKER:-}" ]; then sh -c "$1"; return; fi
  set -- "$1"; local args=()
  for v in $PASS_ENV; do args+=(-e "$v=${!v}"); done
  docker compose exec -T "${args[@]}" api sh -c "$1"
}

case "${1:-status}" in
  status)
    docker compose ps
    echo; curl -s -m 5 "localhost:$PORT/healthz" || echo "API not answering on port $PORT"
    echo; nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
    ;;
  prep)
    python3 tools/prepare_videos.py --zip "${2:?path to open.zip}" --out var/videos
    ;;
  eval)
    run=${2:?run name, e.g. base-4b}
    in_api "python tools/eval.py --api http://localhost:8000 --key \"\${API_KEYS%%,*}\" --videos var/videos --out var/runs/$run --parallel ${PARALLEL:-5}"
    ;;
  modes)
    tag=${2:-first_run}
    for m in violations events both; do
      echo "== VLM_MODE=$m"
      in_api "VLM_MODE=$m python tools/eval.py --local ${EVAL_ANALYZER:-pipeline} --videos var/videos --out var/runs/$tag/$m" 2>&1 | tee ~/"$tag-$m.log"
    done
    tar czf ~/"$tag.tgz" -C var/runs "$tag" -C ~ "$tag-violations.log" "$tag-events.log" "$tag-both.log"
    echo; grep -H "^BOARD" ~/"$tag"-*.log || true
    echo "packed: ~/$tag.tgz  (laptop: gcloud compute scp test-vm-a100:~/$tag.tgz . --zone us-central1-f)"
    ;;
  own)
    tag=${2:-1}
    [ -f var/own_raw/shotlog.csv ] || { echo "put the clips and shotlog.csv into var/own_raw first"; exit 1; }
    in_api "python3 tools/prepare_own.py --src var/own_raw --log var/own_raw/shotlog.csv --out var/own" || echo "(fix the PROBLEM lines above, or go on without those clips)"
    for phone in a b; do
      echo "== phone $phone"
      in_api "python3 tools/eval.py --local ${EVAL_ANALYZER:-pipeline} --videos var/own --labels var/own/labels_$phone.csv --out var/runs/own-$tag-$phone" 2>&1 | tee ~/"own-$tag-$phone.log" | grep -E "^BOARD|^[x ] OWN|Traceback"
    done
    tar czf ~/"own-$tag.tgz" -C var/runs "own-$tag-a" "own-$tag-b" -C ~ "own-$tag-a.log" "own-$tag-b.log"
    echo; grep -H "^BOARD" ~/"own-$tag"-[ab].log; echo "packed: ~/own-$tag.tgz"
    ;;
  load)
    in_api 'big=$(ls -S var/videos/*.mov | head -1); d=$(mktemp -d); for i in 1 2 3 4 5; do ln -s "$PWD/$big" "$d/worst-$i.mov"; done; echo "5 x $big"; python tools/eval.py --api http://localhost:8000 --key "${API_KEYS%%,*}" --files "$d"/*.mov --out var/runs/load --parallel 5; rm -rf "$d"'
    ;;
  logs)
    docker compose logs --since "${2:-15m}" api 2>&1 | grep -i -E "error|timeout|warn|traceback" | tail -60 || echo "no errors in the last ${2:-15m}"
    ;;
  pin)
    img=${VLLM_IMAGE:-vllm/vllm-openai:latest}
    digest=$(docker inspect --format '{{index .RepoDigests 0}}' "$img")
    sed -i "s#^VLLM_IMAGE=.*#VLLM_IMAGE=$digest#; s#^HF_HUB_OFFLINE=.*#HF_HUB_OFFLINE=1#" .env
    grep -E "^(VLLM_IMAGE|HF_HUB_OFFLINE)=" .env
    docker compose up -d
    ;;
  idle)
    h=$(curl -s -m 5 "localhost:$PORT/healthz") || { echo "API not answering: nothing to lose, safe to stop"; exit 0; }
    echo "$h" | python3 -c 'import json, sys
h = json.load(sys.stdin); j = h.get("jobs", {})
busy = h.get("queue", 0) + j.get("queued", 0) + j.get("processing", 0)
print("SAFE to stop" if not busy else f"BUSY: {busy} job(s) queued or running, wait")
sys.exit(1 if busy else 0)'
    ;;
  *)
    sed -n '2,12p' "$0"; exit 1
    ;;
esac
