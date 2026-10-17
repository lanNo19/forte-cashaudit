#!/usr/bin/env bash
# Overnight chain, 18 Oct ~03:45 (Platform asleep until ~07:30). Run on the VM from the repo folder, on main:
#   nohup deploy/experiments/night2.sh > ~/night2.out 2>&1 &        progress: tail -f ~/night2.out
# Nothing is trained or merged: it only measures the two candidates for the 09:30 release decision.
# 0. every finished job's stored result (var/jobs: meta with fps used, warnings, CV states) without the JPEGs
#    -> ~/jobs-meta.jsonl, for Amina's state-budget check on soak1/soak3 and the X-14 flips in the load tests
# 1. RC1 (main) on our own clips (var/own_raw: 12 types x violation/clean x 2 phones), DEBUG_EVENTS=1   -> ~/own-1.tgz
# 2. load test with VLM_CONCURRENCY=8 instead of 4. With WORKERS=1 one job keeps only 4 requests in flight; the 5th of
#    5 long uploads started with < 150 s left, ran at half fps and flipped clean X-14 to violation at 219 s. .env restored.
# 3. combo4 = main + origin/ml/veto-only (fusion fixes only), DEBUG_EVENTS=1: the 42 clips (VLM_MODE=both) -> ~/combo4.tgz,
#    then our own clips -> ~/own-c4.tgz. Back to main and rebuilt.
# 4. RC1 on the camera-shifted copies through the API, 5 at once (lowest priority, last)  -> var/runs/soak3-shifted
# It always ends on main with .env restored and the api rebuilt, even when a step fails, and prints the BOARD lines.
set -uo pipefail
cd "$(dirname "$0")/../.."

main() {
  PORT=$(grep -E '^API_PORT=' .env 2>/dev/null | cut -d= -f2); PORT=${PORT:-80}
  [ "$(git rev-parse --abbrev-ref HEAD)" = main ] || { echo "NOT STARTED: the VM is not on main (git checkout main)"; exit 1; }
  git diff --quiet || { echo "NOT STARTED: local changes in tracked files (git diff --stat)"; exit 1; }
  pgrep -f "deploy/vm.sh (modes|own|load|eval)" >/dev/null && { echo "NOT STARTED: another vm.sh run is going (pgrep -af vm.sh)"; exit 1; }
  [ -f var/own_raw/shotlog.csv ] || { echo "NOT STARTED: var/own_raw/shotlog.csv missing"; exit 1; }
  git fetch -q origin
  trap finish EXIT
  step "start on $(git log --oneline -1)"
  step "0/4 save every finished job's full result (soak1, soak3, load tests; CV meta included, JPEGs dropped)"
  jobs_meta

  step "1/4 RC1 on our own clips"
  DEBUG_EVENTS=1 deploy/vm.sh own 1

  step "2/4 load test with VLM_CONCURRENCY=8"
  cp .env ~/env.night2.bak
  if grep -q '^VLM_CONCURRENCY=' .env; then sed -i 's/^VLM_CONCURRENCY=.*/VLM_CONCURRENCY=8/' .env; else echo "VLM_CONCURRENCY=8" >> .env; fi
  docker compose up -d --no-deps --force-recreate api 2>&1 | tail -1
  LOAD_OUT=load-c8 deploy/vm.sh load 2>&1 | tee ~/load-c8.log
  cp ~/env.night2.bak .env && rm -f ~/env.night2.bak
  docker compose up -d --no-deps --force-recreate api 2>&1 | tail -1

  step "3/4 combo4 = main + origin/ml/veto-only ($(git log --oneline -1 origin/ml/veto-only))"
  combo4
  back_to_main

  step "4/4 RC1 on the camera-shifted copies through the API, 5 at once"
  if [ "$(ls var/shifted/*.mov 2>/dev/null | wc -l)" -lt 42 ]; then
    docker compose exec -T api python3 tools/make_shift_set.py --videos var/videos --out var/shifted | tail -2
  fi
  ready && docker compose exec -T api sh -c 'python3 tools/eval.py --api http://localhost:8000 --key "${API_KEYS%%,*}" --videos var/shifted --out var/runs/soak3-shifted --parallel 5' \
    2>&1 | tee ~/soak3-shifted.log | grep -E "^BOARD|Traceback|failed"
}

combo4() {
  git checkout -q -B combo4 main && git -c user.name=vm -c user.email=vm@localhost merge -q --no-edit origin/ml/veto-only \
    || { echo "combo4: merge failed, skipped"; git merge --abort 2>/dev/null; return; }
  docker compose up -d --build --no-deps api 2>&1 | tail -1
  DEBUG_EVENTS=1 MODES=both deploy/vm.sh modes combo4
  python3 tools/run_report.py var/runs/combo4/both --wrong
  step "3/4 combo4 on our own clips"
  DEBUG_EVENTS=1 deploy/vm.sh own c4
}

back_to_main() {
  [ "$(git rev-parse --abbrev-ref HEAD)" = main ] && ! git rev-parse -q --verify combo4 >/dev/null && return
  git checkout -q main && git branch -q -D combo4 2>/dev/null
  docker compose up -d --build --no-deps api 2>&1 | tail -1
  ready
}

step() { echo; echo "== $(date +%H:%M:%S) UTC  $1"; }
jobs_meta() {
  python3 - <<'EOF'
import json, pathlib, time
rows, store = [], pathlib.Path("var/jobs")
for d in sorted(store.iterdir()) if store.is_dir() else []:
    try:
        j = json.loads((d / "job.json").read_text()); r = json.loads((d / "result.json").read_text())
    except Exception:
        continue
    for v in r.get("violations", []):
        v.pop("evidence_frames", None)
    rows.append({"utc": time.strftime("%m-%d %H:%M:%S", time.gmtime(j["created"])), "file": j["filename"],
                 "wait_sec": round((j.get("started") or j["created"]) - j["created"], 1),
                 "total_sec": round((j.get("finished") or j["created"]) - j["created"], 1), **r})
rows.sort(key=lambda r: r["utc"])
p = pathlib.Path.home() / "jobs-meta.jsonl"
p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
print(f"{len(rows)} jobs -> {p}")
for r in rows:
    if r["file"].startswith("worst-"):
        m = r.get("meta") or {}
        print(f"  {r['utc']} {r['file']:12} wait {r['wait_sec']:6.1f}s total {r['total_sec']:6.1f}s fps {m.get('fps_used')} "
              f"{r.get('verdict')} {[v.get('type_id') for v in r.get('violations', [])]} warnings {len(m.get('warnings') or [])}")
EOF
}
ready() {
  for _ in $(seq 1 90); do curl -s -m 3 "localhost:$PORT/healthz" 2>/dev/null | grep -q '"ok":true' && return 0; sleep 2; done
  echo "API NOT READY after 3 min"; return 1
}
finish() {
  step "finish: main, .env, api"
  [ -f ~/env.night2.bak ] && cp ~/env.night2.bak .env && rm -f ~/env.night2.bak && docker compose up -d --no-deps --force-recreate api 2>&1 | tail -1
  back_to_main
  ready && curl -s -m 5 "localhost:$PORT/healthz"; echo
  echo "code: $(git log --oneline -1)   $(grep -E '^(WORKERS|VLM_CONCURRENCY)=' .env | tr '\n' ' ')"
  step "summary (own clips: phone a = .mov, phone b = .mp4; 12 violation + 12 clean each)"
  grep -H "^BOARD" ~/own-1-a.log ~/own-1-b.log ~/own-c4-a.log ~/own-c4-b.log ~/combo4-both.log ~/soak3-shifted.log 2>/dev/null
  echo "load test, VLM_CONCURRENCY=8 (VLM_CONCURRENCY=4 was: slowest 219 s, 4/5 no_violation):"
  grep -E "worst-[0-9] +(done|failed)" ~/load-c8.log 2>/dev/null
  jobs_meta | tail -12
}

main "$@"; exit
