# Status

Update this table in the same PR as the work. Newest first in the log at the bottom.
States: **done** (tested), **working** (runs, numbers pending), **placeholder** (stub that returns nothing),
**experimental** (code exists, never run against the real thing), **todo**.

| Component | Owner | State | Next step | How to check |
|---|---|---|---|---|
| Jury API, auth, job queue, deadlines, watchdog | Platform | done | deploy behind a stable public URL | `make check`; `make eval-api … --parallel 5` |
| Upload UI (RU) | Platform | done | check on the GPU box in Chrome | open `/`, upload a clip |
| Deploy: GCP A100, compose, Qwen3-VL-4B default, on/off sessions | Platform (Ablay + Claude) | working | session 1 setup: `docs/DEPLOY.md` | `deploy/vm.sh status`; `deploy/vm.sh load` slowest < 240 s |
| Public URL (stable across restarts) | Platform (Ablay + Claude) | done | – | https://34-171-142-215.sslip.io/healthz (Caddy + Let's Encrypt on static IP 34.171.142.215) |
| GPU service image for the CV layer (`Dockerfile.gpu`, on the vLLM image) | Platform (Ablay + Claude) | experimental | build on the VM once CV needs torch: `API_DOCKERFILE=Dockerfile.gpu` | `docker compose build api` prints the torch version |
| VLM client: scan, events, verify, scene | ML lead | experimental | run on the A100 (4B default, 8B to compare); compare `VLM_MODE` and `VLM_INPUT` | `deploy/vm.sh eval <run>` |
| Fusion + per-type floors + veto | ML lead | done (logic), numbers todo | `make calibrate` after the first real run | `tests/test_fusion.py` |
| CV event layer: people (`cashaudit/cv/`) | CV (Amina) | working: T02 1/1, T10 1/1, 0 false alarms CV-only | ROI states + desk change (#8) | `tests/test_cv_people.py`; `VLM_MODE=violations` eval |
| CV per-video geometry (#7): registration + OWLv2 anchors | CV (Amina) | working: 42/42 shifted copies registered, same score as unshifted | check the unregistered fallback on own oblique clips | `tests/test_cv_roi.py`; eval on `make shift-set` |
| Rules T01–T12, T14, T16, T17 | CV + ML lead | done on synthetic logs | feed real events; write T13, T15, T18–T20 | `tests/test_rules.py` |
| Knowledge base: 20 types, 46 legal points | everyone | done | check the bank-secrecy article in the 2026 Law on Banks | `tests/test_kb.py` |
| Eval: scoreboard, schema check, API/latency mode | everyone | done | – | `make eval-local` |
| Camera-angle test set | everyone | done | film 8–12 own clips from an oblique angle | `make shift-set` |
| Jury docs (Claude Doc, RU) | Platform + Claude | draft | fill URL, key, final numbers, known errors | section 8 of the brief |

## Integration checklist for the CV layer

1. Implement `EventExtractor.extract(frames, duration, scene)` in `analyzers/pipeline.py` (or `cashaudit/cv/` and
   import it there). Frames are `[(t_sec, BGR)]` at `VLM_FPS`, native resolution.
2. Emit events from the table in `docs/INTERFACES.md` §4. Set `ready = True`.
3. Run `make eval-local` with `VLM_MODE=violations` to see CV-only hits, then `both`.
4. Calibrate: `make calibrate PREDS=runs/<run>`; commit `data/thresholds.json` if the table looks sane.

## Log

- 2026-10-17 19:30 · Amina + Claude · per-video geometry (#7): the training camera moved up to 66 px / 5 deg between recording sessions; ORB registration to 6 session references (`data/camera_ref.npz`, points only, no images) maps the hand-drawn polygons, incl. mirror; OWLv2-large anchors give the safe zone per video and place door/floor when a camera does not register. Shift-set CV-only: hand-drawn `BOARD Video acc 24/42 · Missed 18/22 · False alarms 0/20 · Median |dt| 0.7 s (≤2 s: 3/4) · Type acc 2/4` (2 wrong-type alarms) → registration `BOARD Video acc 22/42 · Missed 20/22 · False alarms 0/20 · Median |dt| 0.2 s (≤2 s: 2/2) · Type acc 2/2`; 42/42 shifted clips registered.
- 2026-10-17 17:30 · Amina + Claude · CV people layer (#6): RF-DETR Small persons at VLM_FPS, room-floor filter (corridor behind the glass), greedy tracker + door-aware stitching, clothing colour only to re-identify after gaps > 1 s, roles by time/behaviour, CV scene guess when the VLM has none, `cashier_zone`. `data/cameras.json` (hand-drawn for the training camera, normalized). CV-only eval (42 clips): `BOARD Video acc 22/42 · Missed 20/22 · False alarms 0/20 · Median |dt| 0.7 s (≤2 s: 2/2) · Type acc 2/2 · Unanswered 0 · Schema errors 0`. RTX 3050: ~0.12 s/frame detect.
- 2026-10-17 17:30 · Claude · A100 live: 42/42 answered, 0 schema errors, base-4b 22/42 (prompt bug, PR #12), load test 5×143 s slowest 81 s. `Dockerfile.gpu` (opt-in, on the vLLM image) for the CV layer.
- 2026-10-17 16:30 · Claude · A100-only deploy with on/off sessions, Qwen3-VL-4B default (8B one switch away), `deploy/vm.sh` (prep, eval, load, pin, idle), `tools/prepare_videos.py` (zip has mixed Latin/Cyrillic names).
- 2026-10-17 14:40 · Claude · deploy for 1 or 2 GPUs (compose profiles), Caddy option, `make load-test`, `docs/DEPLOY.md` (lab box + tunnel, GCP A100 backup with costs). Ablay moves to Platform.
- 2026-10-17 14:10 · Claude · repo pushed to GitHub (private), CI green; labels and issues #1–#10 for sprints 1–2.
- 2026-10-17 13:55 · Claude · VLM as event sensor (`VLM_MODE`), Qwen3-VL-8B default, one vLLM per GPU, scene filter,
  verifier veto, per-type floors + calibrate tool, video input mode (experimental). 31 tests pass.
- 2026-10-17 13:25 · Claude · legal index (46 points), candidate types T18–T20, camera-shift set generator.
- 2026-10-17 13:12 · Claude · service skeleton, eval script; oracle scores 42/42 through the live API.
