# Status

Update this table in the same PR as the work. Newest first in the log at the bottom.
States: **done** (tested), **working** (runs, numbers pending), **placeholder** (stub that returns nothing),
**experimental** (code exists, never run against the real thing), **todo**.

| Component | Owner | State | Next step | How to check |
|---|---|---|---|---|
| Jury API, auth, job queue, deadlines, watchdog | Platform | done | deploy behind a stable public URL | `make check`; `make eval-api … --parallel 5` |
| Upload UI (RU) | Platform | done | check on the GPU box in Chrome | open `/`, upload a clip |
| Deploy: compose with 1 or 2 vLLM servers (`2gpu` profile), Caddy (`https` profile) | Platform (Ablay + Claude) | experimental | lab box + tunnel, or GCP A100 backup: `docs/DEPLOY.md` | `/healthz`; `make load-test` slowest < 240 s |
| Public URL (stable across restarts) | Platform (Ablay + Claude) | todo | Tailscale Funnel from the lab box (needs outbound internet), else GCP static IP | URL survives `docker compose restart` |
| VLM client: scan, events, verify, scene | ML lead | experimental | run on the real Qwen3-VL-8B; compare `VLM_MODE` and `VLM_INPUT` | `make eval-local` |
| Fusion + per-type floors + veto | ML lead | done (logic), numbers todo | `make calibrate` after the first real run | `tests/test_fusion.py` |
| CV event layer (`EventExtractor`) | CV | placeholder | tracking + roles, ROI states (door, safe, key, screen), desk change | `tests/test_rules.py` shows the events each rule needs |
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

- 2026-10-17 14:40 · Claude · deploy for 1 or 2 GPUs (compose profiles), Caddy option, `make load-test`, `docs/DEPLOY.md` (lab box + tunnel, GCP A100 backup with costs). Ablay moves to Platform.
- 2026-10-17 14:10 · Claude · repo pushed to GitHub (private), CI green; labels and issues #1–#10 for sprints 1–2.
- 2026-10-17 13:55 · Claude · VLM as event sensor (`VLM_MODE`), Qwen3-VL-8B default, one vLLM per GPU, scene filter,
  verifier veto, per-type floors + calibrate tool, video input mode (experimental). 31 tests pass.
- 2026-10-17 13:25 · Claude · legal index (46 points), candidate types T18–T20, camera-shift set generator.
- 2026-10-17 13:12 · Claude · service skeleton, eval script; oracle scores 42/42 through the live API.
