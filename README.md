# forte-cashaudit

Local service that checks overhead CCTV video of a bank cashier's workplace for cash-discipline violations and returns
the verdict, type, timecode, evidence frames and the Kazakh legal basis (ForteBank Datathon 26). No cloud AI: the VLM
runs in our own vLLM. **Private repo:** the organisers' data may be used only within the hackathon.

**Agents and new teammates: start with [AGENTS.md](AGENTS.md)**, then [docs/STATUS.md](docs/STATUS.md) and
[docs/INTERFACES.md](docs/INTERFACES.md). Decisions: [docs/DECISIONS.md](docs/DECISIONS.md). Research behind the
architecture: [docs/RESEARCH.md](docs/RESEARCH.md).

## Quick start (no GPU)

```bash
make install
make check                 # licence/cloud scan + 34 tests, ~10 s
make serve-stub            # UI at http://127.0.0.1:8000, API key "dev"
curl -H "X-API-Key: dev" -F "file=@video.mov" http://127.0.0.1:8000/api/v1/analyze
curl -H "X-API-Key: dev" http://127.0.0.1:8000/api/v1/jobs/<job_id>
```

The training clips are not in git. Put the organisers' `open/videos` folder next to the repo (or pass `VIDEOS=…`).

## Evaluate

```bash
make eval-local VIDEOS=/path/open/videos OUT=runs/try1            # in-process, prints the BOARD line
make eval-api URL=https://<public-url> KEY=$KEY VIDEOS=…          # like the jury: 5 uploads in parallel
make shift-set VIDEOS=… SHIFTED=/path/shifted                     # tilted/zoomed/mirrored copies, same labels
make eval-local VIDEOS=/path/shifted OUT=runs/shift
make calibrate PREDS=runs/try1                                    # per-type floors so clean clips stay clean
```

`tools/eval.py` also checks every answer against the section-10 format (bare base64 JPEGs, 1–3 frames, verdict
consistent with the list). `ANALYZER=oracle` answers training clips from the labels and must score 42/42: use it to
test plumbing only, never deploy it.

## Deploy

Step by step, with the on/off schedule and costs: [docs/DEPLOY.md](docs/DEPLOY.md). One GCP A100, run only during
work sessions; `deploy/vm.sh` has the routine commands (prep, eval, load, pin, idle).

```bash
cp .env.example .env       # API_KEYS (deploy/vm.sh prep unpacks the clips)
docker compose up -d --build
make load-test URL=<public-url> KEY=<key> VIDEOS=…    # jury worst case: largest clip × 5 at once
```

- Compose runs **one vLLM per GPU** (`vlm0`; `vlm1` with the `2gpu` profile); on the A100 both API workers share one.
- Run **one** uvicorn process: the job queue is in memory. Scale with `WORKERS`.
- The jury's 5-minute clock starts at upload, so queue time counts.
- After the weights are cached set `HF_HUB_OFFLINE=1`: the brief requires no internet for processing.
- Uploaded videos are deleted after analysis (`KEEP_VIDEOS=false`).

## Layout

```
cashaudit/         api.py jobs.py schema.py kb.py video.py events.py rules.py vlm.py fusion.py
  analyzers/       base.py (contract) stub.py oracle.py pipeline.py (CV EventExtractor placeholder lives here)
  static/          index.html (Russian UI)
data/              rules_kb.json legal_index.json train_annotations.csv train_md5.json [thresholds.json]
tools/             eval.py calibrate.py make_shift_set.py build_legal_index.py check_licences.py
tests/             one file per module; test_rules.py doubles as the CV layer's spec
docs/              STATUS.md INTERFACES.md DECISIONS.md RESEARCH.md
```

Licence rules, ownership and how the parts connect are in [AGENTS.md](AGENTS.md).
