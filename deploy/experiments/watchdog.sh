#!/usr/bin/env bash
# Watchdog test (sprint 3, rule 5): the jury must get valid JSON within 300 s of upload even when the VLM server dies
# in the middle of a job. Uploads one clip through the jury API, stops vlm0 20 s later, waits for the answer, checks
# the section-10 fields, then starts vlm0 again and waits until it serves. Run from the repo folder on the VM:
#   deploy/experiments/watchdog.sh                 (default clip: the longest training clip, X-14)
# The public service answers without the VLM for ~3 min while this runs.
set -uo pipefail
cd "$(dirname "$0")/../.."
[ -f .env ] && { set -a; . ./.env; set +a; }
PORT=${API_PORT:-80}; KEY=${API_KEYS%%,*}
clip=${1:-var/videos/#U0425-14.mov}
J='import json, sys'

echo "== $(date +%H:%M:%S) upload $clip"
id=$(curl -s -m 60 -H "X-API-Key: $KEY" -F "file=@$clip" "localhost:$PORT/api/v1/analyze" \
     | python3 -c "$J; print(json.load(sys.stdin)['job_id'])") || { echo "upload failed"; exit 1; }
t_up=$(date +%s)
echo "job $id; stopping vlm0 in 20 s"
sleep 20
docker compose stop vlm0 >/dev/null 2>&1 && echo "== $(date +%H:%M:%S) vlm0 stopped"

st=processing
while [ "$st" = processing ]; do
  sleep 5
  body=$(curl -s -m 10 -H "X-API-Key: $KEY" "localhost:$PORT/api/v1/jobs/$id")
  st=$(echo "$body" | python3 -c "$J; print(json.load(sys.stdin).get('status'))" 2>/dev/null || echo "?")
  [ $(( $(date +%s) - t_up )) -gt 330 ] && { echo "NO ANSWER 330 s after upload"; break; }
done
echo "== answer $(( $(date +%s) - t_up )) s after upload (limit 300): status $st"
echo "$body" | python3 -c "$J
b = json.load(sys.stdin)
need = {'job_id', 'status', 'verdict', 'violations'}
bad = sorted(need - set(b)) if b.get('status') == 'done' else []
for v in b.get('violations', []):
    bad += [f'violation missing {k}' for k in ('type', 'timecode_sec', 'description', 'evidence_frames') if k not in v]
print('verdict', b.get('verdict'), '| violations', len(b.get('violations', [])), '| schema', 'OK' if not bad else bad)
print('error:', b['error']) if b.get('error') else None"

echo "== $(date +%H:%M:%S) starting vlm0 again (the model takes 2-3 min to load)"
docker compose start vlm0 >/dev/null 2>&1
for _ in $(seq 1 72); do
  docker compose logs --since 5m vlm0 2>&1 | grep -q "Application startup complete" && { echo "== $(date +%H:%M:%S) vlm0 serving"; break; }
  sleep 5
done
curl -s -m 5 "localhost:$PORT/healthz"; echo
