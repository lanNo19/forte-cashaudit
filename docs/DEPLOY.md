# Deploy

Owner: Platform (Ablay + Claude). Two targets, same `docker-compose.yml`:

- **A. Lab box (2× RTX 5090) + outbound tunnel: $0.** Preferred if the box has outbound internet. The lab firewall
  blocks incoming connections; a tunnel connects outward, so it is not blocked.
- **B. GCP `a2-highgpu-1g` (1× A100 40 GB): backup, paid per hour.** Set up once, stopped, started only if A fails.

Locality (brief, section 7): our own box or a VM we rent counts as "the team's resources" (FAQ 22) and all models run
inside our service. Never enable or call Vertex AI, Gemini, Video Intelligence or any other hosted AI API.
The brief does not say when the jury runs or how long the service must stay up: ask the organisers.

## Will the hardware hold the jury load?

Test videos are like the training ones: 10–145 s, about 29 MB at most (FAQ 1, 25). Worst case: 5 × 143 s uploaded at
once, each answered within 5 minutes of its upload.

| VLM_MODE | prompt tokens for 5 × 143 s | 1× A100 (estimate) |
|---|---|---|
| events | ~1.1 M (17 windows × ~13k tokens × 5) | ~3 min: passes, thin margin |
| both | ~1.2–2.2 M (the second prompt per window reuses the cached image prefix if it is still in vLLM's cache) | 3–5 min: measure |

Estimates only (A100 prefill ~7–9k tokens/s for an 8B model); 2× 5090 has more total throughput than 1× A100.
`make load-test` measures it. If the slowest of 5 misses 240 s: `VLM_MODE=events`, then `VLM_INPUT=video` (half the
image tokens), then `VLM_FPS=1.5`. Training clips are much shorter (median 23 s), so the normal case is easy.

## A. Lab box + tunnel

Check first, on the box: `curl -sI https://pypi.org | head -1` must print an HTTP status.

1. Docker + NVIDIA container toolkit as in B2 (skip the GCP driver installer; Blackwell needs driver ≥ 570).
2. Code and `.env` as in B3, with `COMPOSE_PROFILES=2gpu`, `VLM_BASE_URL=http://vlm0:8001/v1,http://vlm1:8002/v1`,
   `VLM_GPU_MEM=0.80`, `API_PORT=8000`. Start and pin as in B4.
3. Tailscale Funnel: a stable `https://<box>.<tailnet>.ts.net`, no domain, free.
   ```bash
   curl -fsSL https://tailscale.com/install.sh | sh
   sudo tailscale up                 # log in once; enable HTTPS + Funnel in the admin console if it asks (it prints the link)
   sudo tailscale funnel --bg 8000
   tailscale funnel status           # the public URL to submit
   ```
4. Backup tunnel if Funnel is blocked: `cloudflared tunnel --url http://localhost:8000` prints a `trycloudflare.com`
   URL. It changes on every restart, so it is a stand-by, not the URL we submit.
5. Smoke and load tests (B5, B6) against the public URL, from a machine outside the lab.
6. From Sunday 09:30 nothing but the service runs on this box (an experiment on the GPUs slows the jury's answers).

## B. GCP A100 (backup)

### Cost: pay only while it runs

| Machine | $/h (us-central1 / europe-west4) |
|---|---|
| a2-highgpu-1g, 1× A100 40 GB (ours) | 3.67 / 3.75 |
| same on Spot | ~2.0–2.2, but Google can reclaim it during the jury run |
| a2-highgpu-2g, 2× A100 | 7.35 |
| g2-standard-8, 1× L4 | ~0.84, but ~2.5× slower: long videos in parallel miss 5 min |

Stopped VM: only the disk (200 GB balanced, ~$0.67/day) and the reserved IP (~$0.24/day). Plan: today 1–2 h to set up
and load-test (~$4–8), then stop. If A fails on Sunday, start at 08:00 and stop as soon as the jury is done
(Sun 08:00–18:00 ≈ $37). Prices observed 11–17 Oct 2026 (sparecores.com, gcloud-compute.com); check the console.
Set a budget alert in Billing.

> Starting a stopped A100 VM can fail when the zone has no free A100s. Start it early on Sunday so there is time to
> fall back. Once its URL is submitted, never stop it until the jury has finished.

Set these once (Cloud Shell or a laptop with `gcloud`):

```bash
VM=<instance-name>; ZONE=<zone, e.g. us-central1-a>; REGION=${ZONE%-*}
gcloud compute instances describe $VM --zone $ZONE \
  --format="value(machineType.basename(),scheduling.provisioningModel,scheduling.automaticRestart,networkInterfaces[0].accessConfigs[0].natIP)"
```

### B1. Disk, static IP, firewall

```bash
# disk: the default boot disk is too small for the vLLM image (~20 GB) + Qwen3-VL-8B (~17 GB) + CV weights
gcloud compute disks resize $VM --zone $ZONE --size 200GB      # boot disk usually has the VM's name
# then on the VM: lsblk; sudo growpart /dev/sda 1; sudo resize2fs /dev/sda1   (device names may differ: check lsblk)

# keep the IP across stop/start (the submitted URL must not change)
IP=$(gcloud compute instances describe $VM --zone $ZONE --format="value(networkInterfaces[0].accessConfigs[0].natIP)")
gcloud compute addresses create cashaudit-ip --addresses $IP --region $REGION

# open 80/443 only
gcloud compute firewall-rules create cashaudit-web --allow tcp:80,tcp:443 --target-tags cashaudit --source-ranges 0.0.0.0/0
gcloud compute instances add-tags $VM --zone $ZONE --tags cashaudit
```

### B2. Driver, Docker, NVIDIA container toolkit (on the machine)

```bash
nvidia-smi || {   # GCP only: Deep Learning VM images already have the driver; plain Debian/Ubuntu need this
  curl -L https://storage.googleapis.com/compute-gpu-installation-us/installer/latest/cuda_installer.pyz -o cuda_installer.pyz
  sudo python3 cuda_installer.pyz install_driver   # may reboot; run it again after the reboot if it asks
}
curl -fsSL https://get.docker.com | sudo sh && sudo usermod -aG docker $USER && sudo systemctl enable docker
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
newgrp docker; docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi   # must list the GPUs
```

### B3. Code and secrets

```bash
# private repo: a read-only deploy key (GitHub > repo > Settings > Deploy keys > add the .pub)
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N "" && cat ~/.ssh/id_ed25519.pub
git clone git@github.com:lanNo19/forte-cashaudit.git && cd forte-cashaudit
cp .env.example .env
python3 -c "import secrets; print(secrets.token_urlsafe(32))"   # put it in API_KEYS; this is the jury key
```

GCP `.env`: `API_KEYS`, and either `API_PORT=80` (no domain: the service is `http://<ip>/`) or
`COMPOSE_PROFILES=https`, `DOMAIN=<name whose A record is the static IP>`, `API_PORT=8000`.

### B4. Start, then pin

```bash
docker compose up -d --build
docker compose logs -f vlm0          # first start downloads ~17 GB; wait for "Application startup complete"
curl -s localhost:${API_PORT:-80}/healthz
# pin the exact vLLM build you just ran, so a restart cannot pull another version:
docker inspect --format '{{index .RepoDigests 0}}' vllm/vllm-openai:latest   # -> VLLM_IMAGE=<that> in .env
# processing must not need the internet once weights are cached:
sed -i 's/^HF_HUB_OFFLINE=.*/HF_HUB_OFFLINE=1/' .env && docker compose up -d
```

### B5. Smoke test from outside

```bash
make eval-api URL=<public-url> KEY=<key> VIDEOS=<path>/open/videos OUT=runs/deploy1   # all 42, 5 in parallel
```
Expect 0 schema errors, every job under 300 s. Open the URL in a browser and upload one clip.

### B6. Load test: the jury's worst case

```bash
make load-test URL=<public-url> KEY=<key> VIDEOS=<path>/open/videos
```
Uploads the largest training clip 5 times at once. Pass: the slowest answer under 240 s (30 s margin to the 270 s
budget). Post the numbers on the board.

### B7. Night and jury window

- `docker compose ps` all healthy; `docker compose logs --since 10m api | grep -i -E "error|timeout"`.
- After a code change: `git pull && docker compose up -d --build api` (the vLLM containers keep running).
- Restarts are safe: finished jobs reload from `var/jobs`; jobs that were running re-queue within their deadline.

### B8. After the jury has finished

GCP: delete the VM, its disk and the static IP (a reserved IP that is not attached is still billed). Test videos must
not be kept anywhere (brief, section 10).
