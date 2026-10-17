# AGENTS.md: read this before changing anything

Guide for the coding agents (and humans) of the 3-person team. It tells you what the system is, who owns which part,
how the parts connect, and the rules every change must follow. Keep it current: if you change a contract or finish a
component, update this file, `docs/INTERFACES.md` and `docs/STATUS.md` in the same PR.

## What we are building

A fully local service for **ForteBank Datathon 26**. It takes an overhead CCTV video (.mov) of a bank cashier's desk or
cash room and returns, per the brief's section 10 JSON: verdict (`violation` / `no_violation`) and, per violation,
type, timecode, description with the Kazakh legal basis, and 1–3 evidence JPEGs.
Deadline: **Sunday 18 Oct 2026, 12:00 Almaty**. The jury scores automatically via the API on 5+ unseen videos (the last
one from a different camera angle, possibly with unseen violation types). Scoring: detection 25, timecode 15, type 15,
legal grounding 15, robustness 10, product 10, technical soundness 10.

## Hard rules (breaking one can disqualify the team)

1. **No cloud AI.** Never call OpenAI, Anthropic, Google, or any hosted model API from code, not even with frames or text.
   The VLM runs in our own vLLM. `tools/check_licences.py` fails CI on cloud AI endpoints.
2. **Commercial-use licences only.** Out: `ultralytics` (AGPL, includes its YOLO, RT-DETR, trackers and any weights
   trained with it), `boxmot` (AGPL), YOLO-World (GPL), `rfdetr[plus]` sizes (PML), FLUX.1-dev/Kontext, HunyuanVideo,
   API-only models (Grounding DINO 1.5/1.6, T-Rex2, DINO-X), NVIDIA NIM. Grey zone, ask the organisers first:
   DINOv3, SAM 3, Cosmos-Reason2, Molmo2. Fine: see the stack in `docs/DECISIONS.md`. CI enforces the obvious ones.
3. **No face recognition or identification.** Roles (cashier, client, collector) come from position and behaviour.
4. **The jury JSON is exact.** Only `schema.to_api_json` builds it; `tools/eval.py` checks it. Never add or rename
   fields in the jury API (extras go behind `API_EXTRA_FIELDS`).
5. **Every video gets an answer within 300 s of upload, including queue time.** Analyzers must check
   `deadline.remaining()` and call `report(partial)` after each stage.
6. **Never commit videos, frames from test videos, `.env` or API keys.** Test videos may not be kept or used for training.
7. **Citations only from `data/legal_index.json`** (via `kb.py`). Never invent a point number. Interpretive references
   are labelled as such in the description.

## Map: who owns what

| Area | Files | Owner | Contract it must honour |
|---|---|---|---|
| Jury API, job queue, deadlines, UI | `cashaudit/api.py`, `jobs.py`, `static/index.html` | Platform (Ablay + Claude) | Section 10 JSON; `Analyzer` protocol |
| Deploy, GPU servers, public URL | `Dockerfile`, `docker-compose.yml`, `.env.example`, `deploy/`, `docs/DEPLOY.md` | Platform (Ablay + Claude) | One vLLM per GPU; `VLM_BASE_URL` list |
| CV event layer | `cashaudit/analyzers/pipeline.py` → `EventExtractor` (move it to `cashaudit/cv/` when it grows) | CV | Emits `EventLog` in the vocabulary of `events.py` |
| Rule checks | `cashaudit/rules.py` | CV (states) + ML lead (procedure types) | One function per type id; reads only the event log |
| VLM client and prompts | `cashaudit/vlm.py` | ML lead | Returns `Violation`s and `Event`s; times in video seconds |
| Fusion and thresholds | `cashaudit/fusion.py`, `data/thresholds.json`, `tools/calibrate.py` | ML lead | Input/output: lists of `Violation` |
| Knowledge base | `data/rules_kb.json`, `data/legal_index.json`, `cashaudit/kb.py` | everyone, by PR | `tests/test_kb.py` must pass |
| Evaluation | `tools/eval.py`, `tools/make_shift_set.py`, `data/train_annotations.csv` | everyone | The `BOARD` line is what goes on the sprint board |
| Docs for the jury | Claude Doc "Сервис проверки кассовой дисциплины" (shared separately) | Platform + Claude | Section 8 of the brief |

## How the pieces connect

```
upload ─▶ api.py ─▶ jobs.JobManager (deadline = upload + 270 s, watchdog at 290 s)
                         │ one worker thread per GPU
                         ▼
          analyzers/pipeline.PipelineAnalyzer.analyze(video, deadline, report)
             ├─ video.sample_frames (VLM_FPS, native resolution)
             ├─ vlm.classify_scene → scene A (client desk) | B (cash room) → rule cards shown to the VLM
             ├─ CV: EventExtractor.extract(frames) → EventLog ──┐
             ├─ VLM: scan_window → Violation candidates          │
             ├─ VLM: extract_events → Event list ────────────────┤ merged EventLog
             │                                                   ▼
             │                                   rules.run_rules(log) → Violation candidates
             └─ fusion.fuse(candidates, verifier=vlm.verify) → final Violations
                         ▼
          jobs._finish: evidence JPEGs at evidence_ts → schema.to_api_json → GET /api/v1/jobs/{id}
```

The **event log** is the seam between people: CV and the VLM both write events; `rules.py` turns events into
violations; nobody else needs to know how an event was detected. Which events each rule needs, and who is expected to
emit them, is in `docs/INTERFACES.md` ("Rules → required events").

## How to work (agents and humans)

1. Start by reading `docs/STATUS.md` (what is done, what is a placeholder, what is next) and the section of
   `docs/INTERFACES.md` for the part you touch. Work items are GitHub issues labelled `owner:ml`, `owner:cv`,
   `owner:platform`, `owner:claude`, `owner:team` and `sprint:N` (`gh issue list -l owner:cv`); each says when it
   is done. Close it from the PR with `Closes #N`. The sprint board (check-in numbers, decisions) is shared by Ablay.
2. Work on a branch: `cv/<topic>`, `ml/<topic>`, `platform/<topic>`, `docs/<topic>`. Small PRs, merged often
   (at least before each check-in: 17:00, 21:00, 01:30, 09:30).
3. Stay inside your owner's files. Changing a **contract** (`schema.py`, `events.py` vocabulary, `analyzers/base.py`,
   the jury JSON, `rules_kb.json` structure) needs: the change in `docs/INTERFACES.md`, the `contract-change` label on
   the PR, and a note in the team chat. Additive changes (a new event kind, a new env var) are fine; renames are not.
4. Before pushing: `make check` (licence/cloud scan + all tests, ~10 s, no GPU needed).
5. In the PR: what changed, how you tested it, and the eval `BOARD` line if detection behaviour changed.
   Update `docs/STATUS.md` and, for decisions, add a line to `docs/DECISIONS.md`.
6. Never break `main`: CI must be green. The service must keep starting with `ANALYZER=stub` and `ANALYZER=pipeline`
   even when the VLM server or the CV layer is missing (they degrade to warnings in `meta`).

## Commands

```bash
make install            # pip install -r requirements.txt
make check              # licence/cloud scan + pytest (CI runs the same)
make serve-stub         # API + UI on :8000, analyzer answers "no violation" (key: dev)
make serve-pipeline     # real pipeline; needs vLLM servers (VLM_BASE_URL)
make eval-local VIDEOS=/path/open/videos OUT=runs/x     # score an analyzer in-process
make eval-api URL=https://… KEY=… VIDEOS=…              # what the jury does, 5 uploads in parallel
make shift-set VIDEOS=… SHIFTED=/path/shifted           # camera-angle robustness copies
make calibrate PREDS=runs/x                             # per-type floors so clean clips stay clean
```

Testing without a GPU: `ANALYZER=oracle` answers training clips from the labels (plumbing must score 42/42);
`tests/test_vlm.py` shows how to fake the vLLM server with `httpx.MockTransport`.

## Gotchas

- Run **one** uvicorn process: the job queue lives in memory. Scale with `WORKERS` (= number of GPUs).
- `X-47`/`X-48` are byte-identical copies of `X-4`/`X-5`: 18 distinct clean clips, not 20.
- Label timecode convention: "left unattended" types = when the cashier exits; ordering types = the out-of-order
  action; exposure types = when the extra person enters; T06 = when the cashier gets up. See `rules.py` header.
- The key in the safe lock is ~5 px: invisible to the VLM at full frame. Small states belong to CV ROI crops.
- The final jury video uses a different camera angle: never hard-code pixel ROIs without a per-camera config.
- Russian is the language of all jury-facing text (descriptions, UI).
