# Deploy

Owner: Platform (Ablay + Claude). Target: the GCP `a2-highgpu-1g` VM (1× A100 40 GB, 12 vCPU, 85 GB RAM), running
only during work sessions. Teammates do not log in to it: they push to GitHub, Platform pulls, runs
`deploy/vm.sh eval`, and posts the BOARD line. The L4 VM stays stopped; it is an emergency spare only.

Locality (brief, section 7): a VM we rent and run ourselves is "the team's resources" (FAQ 22) and all models run
inside our service. Never enable or call Vertex AI, Gemini, Video Intelligence or any other hosted AI API.

## Model and load

Default VLM: **Qwen3-VL-4B-Instruct** (Apache-2.0, ~9 GB). On the A100 it leaves ~14 GB for the CV models.
8B (~17.5 GB) also fits: set `VLM_MODEL=Qwen/Qwen3-VL-8B-Instruct` and `VLM_GPU_MEM=0.75`, then compare both with
`deploy/vm.sh eval`.

Jury worst case: 5 videos of ~143 s / 29 MB uploaded at once, each answered within 5 min of upload. Rough estimate
for 4B on one A100: 1.5–2.5 min. Measure it with `deploy/vm.sh load` (pass: slowest < 240 s). If it fails:
`VLM_MODE=events`, then `VLM_INPUT=video`, then `VLM_FPS=1.5`.

## Sessions: the VM runs only when we use it (Almaty time)

| Session | On | Off | Work |
|---|---|---|---|
| 1 | Sat ~16:30 | ~19:00 | setup, first real run on 42 clips, load test, numbers on the board |
| 2 | 20:30 | 22:30 | pull teammates' PRs (merged by 20:30), eval, calibrate; check-in 21:00 |
| 3 | 00:30 | 03:00 | RC1 at 01:30, soak test, angle-shift set, machine image backup |
| 4 | Sun 07:30 | jury done | RC2, 10:30 freeze, 11:00 load test from outside, 11:30 submit. **Never stop after submitting** |

About 17.5 h if the jury is done by Sun 18:00: ~$64 at $3.67/h (always on: ~$94). Stopped, it costs only the disk
(~$0.67/day for 200 GB) and the reserved IP (~$0.24/day). Ask the organisers when the jury runs.

Risk: a stopped A100 VM may not start again when the zone has no free A100s (`ZONE_RESOURCE_POOL_EXHAUSTED`). That is
why session 4 starts at 07:30 and why session 3 ends with a machine image (see "Emergency").

## First setup (session 1)

On your laptop (Cloud Shell works too):

```bash
VM=test-vm-a100; ZONE=us-central1-f; REGION=us-central1   # static IP 34.171.142.215 (cashaudit-ip)
gcloud compute instances describe $VM --zone $ZONE --format="table(machineType.basename(),scheduling.provisioningModel,disks[0].diskSizeGb,networkInterfaces[0].accessConfigs[0].natIP)"
```

Expect `a2-highgpu-1g  STANDARD  <size>  <ip>`. `SPOT` means Google can take it back during the jury run: stop
and ask before going on.

**1. Make the IP permanent first.** An ephemeral IP changes on every stop/start; reserving it keeps the URL.

```bash
IP=$(gcloud compute instances describe $VM --zone $ZONE --format="value(networkInterfaces[0].accessConfigs[0].natIP)")
gcloud compute addresses create cashaudit-ip --addresses $IP --region $REGION
gcloud compute firewall-rules create cashaudit-web --allow tcp:80,tcp:443 --target-tags cashaudit --source-ranges 0.0.0.0/0
gcloud compute instances add-tags $VM --zone $ZONE --tags cashaudit
echo "service URL: http://$IP/"
```

**2. Disk: grow it, don't clean or zip.** We need ~60 GB free (vLLM image ~20 GB, API image ~2 GB now and ~10 GB
once CV adds torch, model 9 GB, 8B for comparison 17.5 GB, clips 0.3 GB). Zipping checkpoints frees almost nothing
(weights do not compress). Growing works while the VM runs; it cannot be shrunk later.

```bash
DISK=$(gcloud compute instances describe $VM --zone $ZONE --format="value(disks[0].source.basename())")
gcloud compute disks resize $DISK --zone $ZONE --size 200     # only if df -h / on the VM shows < 60 GB free
gcloud compute scp ~/Downloads/open.zip $VM:~ --zone $ZONE    # training clips, 293 MB
gcloud compute ssh $VM --zone $ZONE
```

On the VM:

```bash
# your old experiments: let them checkpoint, then Ctrl-C (not kill -9); list what still holds the GPU:
nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv
# grow the filesystem if you resized the disk (lsblk shows the device; usually sda/sda1 or nvme0n1/nvme0n1p1)
lsblk; sudo growpart /dev/sda 1 && sudo resize2fs /dev/sda1; df -h /
```

**3. Docker from static binaries (not apt).** Our VM (Deep Learning VM, Ubuntu 24.04, driver 580, toolkit
preinstalled) has a half-finished kernel 7.0 upgrade in apt. **Never run `apt --fix-broken install` or `apt upgrade`
before the jury is done**: it would make the next boot use kernel 7.0, which the NVIDIA driver may not support (no
GPU). Docker therefore comes from Docker's release files. Run each line separately:

```bash
curl -fsSL https://download.docker.com/linux/static/stable/x86_64/docker-27.3.1.tgz | sudo tar xz -C /usr/local/bin --strip-components=1
sudo mkdir -p /usr/local/lib/docker/cli-plugins && sudo curl -fsSL https://github.com/docker/compose/releases/download/v2.29.7/docker-compose-linux-x86_64 -o /usr/local/lib/docker/cli-plugins/docker-compose && sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
printf '[Unit]\nDescription=Docker (static binaries)\nAfter=network-online.target\nWants=network-online.target\n[Service]\nExecStart=/usr/local/bin/dockerd\nRestart=always\nLimitNOFILE=1048576\nDelegate=yes\nKillMode=process\n[Install]\nWantedBy=multi-user.target\n' | sudo tee /etc/systemd/system/docker.service >/dev/null
sudo nvidia-ctk runtime configure --runtime=docker && sudo groupadd -f docker && sudo usermod -aG docker $USER && sudo systemctl daemon-reload && sudo systemctl enable --now docker
sudo docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi --query-gpu=name --format=csv   # A100
```
Log out and back in so `docker` works without sudo. Done on `test-vm-a100` 17 Oct 16:36.

**4. Code, key, clips.** The repo is private: add the VM's public key as a read-only deploy key (GitHub > repo >
Settings > Deploy keys), or paste it to Claude to add.

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N "" -q && cat ~/.ssh/id_ed25519.pub
git clone git@github.com:lanNo19/forte-cashaudit.git && cd forte-cashaudit
cp .env.example .env
sed -i "s#^API_KEYS=.*#API_KEYS=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')#" .env
grep ^API_KEYS .env            # the jury key: keep it, never commit it
deploy/vm.sh prep ~/open.zip   # -> 42/42 clips ready in var/videos
```

**5. Start, test, pin.**

```bash
docker compose up -d --build
docker compose logs -f vlm0           # first start downloads the image and ~9 GB of weights; wait for "startup complete"
deploy/vm.sh status
deploy/vm.sh eval base-4b             # 42 clips through the API: BOARD line for the board
deploy/vm.sh load                     # slowest of 5 must be < 240 s
deploy/vm.sh pin                      # freeze the vLLM image digest, offline weights, restart
```

From your laptop, open `http://<IP>/` and upload one clip.

## Every session

```bash
gcloud compute instances start $VM --zone $ZONE     # laptop. Containers start by themselves (restart: unless-stopped)
gcloud compute ssh $VM --zone $ZONE
cd forte-cashaudit && git pull && docker compose up -d --build api   # new code; vLLM keeps running
deploy/vm.sh status                                  # vLLM ready after ~2-3 min (weights are on disk)
deploy/vm.sh eval <run-name>                         # post the BOARD line
deploy/vm.sh logs
deploy/vm.sh idle && exit                            # SAFE to stop?
gcloud compute instances stop $VM --zone $ZONE       # laptop
```

Restarts are safe: finished jobs reload from `var/jobs`; running jobs re-queue within their deadline.
If `start` fails with "resource pool exhausted": retry every few minutes
(`until gcloud compute instances start $VM --zone $ZONE; do sleep 120; done`), and see "Emergency" by session 4.

## Comparing 4B and 8B

```bash
sed -i 's#^VLM_MODEL=.*#VLM_MODEL=Qwen/Qwen3-VL-8B-Instruct#; s#^VLM_GPU_MEM=.*#VLM_GPU_MEM=0.75#; s#^HF_HUB_OFFLINE=.*#HF_HUB_OFFLINE=0#' .env
docker compose up -d && docker compose logs -f vlm0   # downloads 17.5 GB the first time
deploy/vm.sh eval base-8b && deploy/vm.sh load
```
Switch back the same way, then `deploy/vm.sh pin`.

## Emergency: the A100 will not start

End of session 3, make a backup image (a few cents of storage):

```bash
gcloud compute machine-images create cashaudit-rc1 --source-instance $VM --source-instance-zone $ZONE
```

If the VM cannot start by 09:00 Sunday, recreate it in another zone of the same region, keeping the static IP:

```bash
NAT=$(gcloud compute instances describe $VM --zone $ZONE --format="value(networkInterfaces[0].accessConfigs[0].name)")
gcloud compute instances delete-access-config $VM --zone $ZONE --access-config-name "$NAT"   # frees the static IP
gcloud compute instances create cashaudit-b --zone <other-zone-in-$REGION> --source-machine-image cashaudit-rc1 --address $IP
```

Last resort: the L4 VM with 4B, `VLM_MODE=events`, `VLM_FPS=1.5`. It handles normal videos; 5 long ones at once may
miss 5 minutes. A static IP only moves within its region.

## After the jury has finished

Delete the VM's disk snapshots/machine images and release the static IP (an unattached reserved IP is still billed).
Test videos must not be kept anywhere (brief, section 10).
