#!/usr/bin/env bash
# Read-only check of a machine before we serve from it. Changes nothing. ~1-2 min (bandwidth test).
# Run: bash deploy/probe.sh 2>&1 | tee probe.txt   and paste the output back.
set -u
say() { printf '\n== %s\n' "$*"; }
mbit() { awk -v b="${1:-0}" 'BEGIN { printf "%.1f Mbit/s", b * 8 / 1e6 }'; }

say host
hostname; uname -srm; (. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME"); echo "cpus: $(nproc)"
free -g | awk '/Mem/ { print "RAM GB:", $2, " available:", $7 }'

say disk
df -h "$HOME" /tmp 2>/dev/null

say gpus
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>&1
nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | sed 's/^/driver: /'

say "gpu processes (gpu index, pid, MiB, user, running for, command)"
nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null | tr -d ' ' > /tmp/.probe_gpus
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' |
  while IFS=, read -r uuid pid mem; do
    idx=$(awk -F, -v u="$uuid" '$2 == u { print $1 }' /tmp/.probe_gpus)
    info=$(ps -o user=,etime=,args= -p "$pid" 2>/dev/null | cut -c1-140)
    echo "gpu $idx  pid $pid  ${mem} MiB  $info"
  done
rm -f /tmp/.probe_gpus

say "docker / root"
if command -v docker >/dev/null; then
  docker info --format 'docker {{.ServerVersion}}, runtimes: {{range $k, $v := .Runtimes}}{{$k}} {{end}}' 2>&1 | head -2
else echo "docker: not installed"; fi
sudo -n true 2>/dev/null && echo "sudo: yes, no password" || echo "sudo: no (or needs a password)"
id

say "python / torch / vllm"
python3 --version 2>&1
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)" 2>&1 | tail -1
python3 -c "import vllm; print('vllm', vllm.__version__)" 2>&1 | tail -1
for t in conda uv; do command -v $t >/dev/null && echo "$t: $(command -v $t)"; done

say "model cache"
HFC=${HF_HOME:-$HOME/.cache/huggingface}
ls "$HFC/hub" 2>/dev/null | grep -i qwen || echo "no Qwen models in $HFC/hub"
du -sh "$HFC" 2>/dev/null

say "reachability (http code, seconds)"
for u in https://huggingface.co https://registry-1.docker.io/v2/ https://pypi.org https://github.com \
         https://controlplane.tailscale.com https://api.trycloudflare.com; do
  printf '%-40s %s\n' "$u" "$(curl -s -o /dev/null -m 10 -w '%{http_code} %{time_total}' "$u")"
done

say "bandwidth (jury uploads come in the download direction)"
d=$(curl -s -o /dev/null -m 60 -w '%{speed_download}' 'https://speed.cloudflare.com/__down?bytes=50000000')
echo "download: $(mbit "$d")"
u=$(head -c 10000000 /dev/urandom | curl -s -o /dev/null -m 60 -w '%{speed_upload}' --data-binary @- https://speed.cloudflare.com/__up)
echo "upload:   $(mbit "$u")"
