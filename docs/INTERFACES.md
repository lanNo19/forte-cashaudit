# Interfaces

The contracts between the parts. Additive changes are fine; renames and removals need a PR labelled
`contract-change` that updates this file. Code is the source of truth; this file explains it.

## 1. Analyzer (`cashaudit/analyzers/base.py`)

```python
analyze(video_path: str, deadline: Deadline, report: Callable[[AnalysisResult], None]) -> AnalysisResult
```
- `deadline.remaining()` = seconds left until upload + `JOB_BUDGET_SEC` (270). Degrade (lower fps, skip verifier)
  when it is low. The watchdog answers at upload + `HARD_TIMEOUT_SEC` (290) with the latest `report(...)`.
- Built once per worker: `make_analyzer(name, device=i)`; worker `i` should use GPU `i % <GPU count>` (one A100 = every worker on GPU 0) and VLM server `i % n`.
- Implementations: `stub` (always no violation), `oracle` (training labels by file hash, TEST ONLY), `pipeline`.

## 2. Result types (`cashaudit/schema.py`)

```python
Violation(type_id="T09", type="<name from rules_kb>", timecode_sec=57.0,
          description="<what is visible>. <why>. Основание: <act, point>. Требуется проверка человеком.",
          evidence_ts=[56.0, 57.0, 59.0],   # 1-3 seconds; JPEGs are cut by jobs.py, analyzers never encode them
          confidence=0.8, source="events" | "vlm" | "fused" | "oracle" | "stub", refs=[{act, title, point, url, strength}])
AnalysisResult(violations=[...], meta={...})   # verdict = "violation" iff violations
```
Build descriptions with `kb.describe(type_id, observed)` and refs with `kb.refs(type_id)` (or `kb.suggest_refs` for
T00). Never write a legal reference by hand.

## 3. Jury API (brief, section 10), `cashaudit/api.py`

| Call | Request | Response |
|---|---|---|
| `POST /api/v1/analyze` | multipart field `file` (.mov), header `X-API-Key` | `{"job_id": "3f2a9c1e", "status": "processing"}` |
| `GET /api/v1/jobs/{job_id}` | header `X-API-Key` | `processing` → `{"job_id","status"}`; `done` → `{"job_id","status","verdict","violations":[{"type","timecode_sec","description","evidence_frames":[{"timestamp_sec","image_base64"}]}]}`; `failed` → `{"job_id","status","error"}` |
| `POST /ui/analyze`, `GET /ui/jobs/{id}` | no key (UI) | same + `type_id`, `confidence`, `source`, `refs`, `meta`, `elapsed_sec` |
| `GET /healthz` | – | `{"ok", "queue", "workers", "jobs", "analyzer"}` |

`meta.scene` is the scene the pipeline used; when the CV layer's behaviour guess disagrees with the VLM's scene, the CV scene wins and the VLM's answer is in `meta.scene_vlm` (UI/debug only, not in the jury JSON).

401 without a valid key, 400 for a non-video, 413 above `MAX_UPLOAD_MB`, 404 for an unknown job.

## 4. Event log (`cashaudit/events.py`): the seam between CV, VLM and rules

`Event(t, kind, actor=None, obj=None, zone=None, conf=1.0, extra={})`, `EventLog(events, duration, scene)`.
`t` is seconds from the start of the video. State pairs become intervals with `log.intervals("safe_open")`.

| kind | fields | expected emitter |
|---|---|---|
| `person_enter`, `person_exit` | `actor`: cashier, client, collector, other | CV (tracking + zones), VLM |
| `cashier_zone` | `zone`: desk, safe, door, out (emit on change) | CV |
| `handover` | `obj`: cash, id, cash_bag, doc, card; `extra.from`, `extra.to` | VLM (CV optional via hands) |
| `object_placed`, `object_removed` | `obj`: cash, cash_bag, doc, tablet, handbag, coat, food, cup; `zone`: desk, safe | CV (desk change) + VLM |
| `desk_clear` | – | CV |
| `safe_open` / `safe_close` | – | CV (safe crop → local VLM, `cv/states.py`) |
| `key_in_lock` / `key_removed` | – | CV (lock crop → local VLM; ~5 px key, weakest) |
| `door_open` / `door_close` | – | CV (doorway crop → local VLM) |
| `counting_start` / `counting_end` | `obj`: cash; `extra.device`: counter, hand, detector | CV for counter (ROI motion), VLM for hand/detector |
| `screen_on` / `screen_locked` | `obj`: laptop | CV (laptop crop → local VLM) |
| `consuming` | `actor`: cashier; `obj`: food, drink | VLM |
| `lights_off` | – | CV (global brightness held < 0.8 of the start) |

VLM events carry `extra.source = "vlm"`. Overlapping windows are de-duplicated (same kind/obj/actor/zone within 3 s).

## 5. Rules → required events (`cashaudit/rules.py`)

| Type | Fires when | Events it needs | Timecode |
|---|---|---|---|
| T01 door not closed | cashier enters; door confirmed open 6–8 s later (states are ~1 fps), no `door_close` | `person_enter(cashier)`, `door_open/close` | entry |
| T02 outsider in cash room | `person_enter(other)` in scene B while the cashier works | `person_enter`, `cashier_zone` | entry |
| T03 valuables left | cashier exits with cash/cash_bag/doc/tablet on the desk | `person_exit(cashier)`, `object_placed/removed`, `desk_clear` | exit |
| T04 safe left open | safe open, cashier seen at the safe, then back at the desk > 2 s | `safe_open/close`, `cashier_zone` | back at desk |
| T05 key in lock | cashier exits while key is in the lock | `key_in_lock/removed`, `person_exit(cashier)` | exit |
| T06 screen not locked | cashier exits with screen on (no lights_off) | `screen_on/locked`, `person_exit`, `cashier_zone` | gets up |
| T07 end of day | lights off while screen on | `lights_off`, `screen_on`, `person_exit` | exit |
| T08 cash before ID | first cash handover to cashier with no earlier ID/doc handover in the visit | `person_enter/exit(client)`, `handover` | handover |
| T09 cash exposed to next client | a later client enters while cash from before is on the desk | `person_enter/exit(client)`, `object_placed(cash)` | entry |
| T10 third person during service | another client/other enters during a client visit | `person_enter/exit` | entry |
| T11 client leaves during count | client exits inside a counting interval | `person_exit(client)`, `counting_start/end` | exit |
| T12 bag without documents | cash_bag handed to cashier with no doc/ID handover in the 60 s before | `handover` | handover |
| T14 no detector check | hand counting starts within 10 s after counter, no detector | `counting_*` with `device` | hand count |
| T16 personal items | handbag/coat placed on the desk | `object_placed` | placed |
| T17 food | `consuming`, else food/cup placed | `consuming`, `object_placed` | first |
| T13, T15, T18–T20 | no rule yet: VLM path only (`scan_window`) | – | – |

`tests/test_rules.py` has one synthetic event log per training scenario: the fastest way to see what to emit.

## 6. VLM client (`cashaudit/vlm.py`)

`VLMClient(base_url, model, input_mode)` → `classify_scene(frames)`, `scan_window(frames, ids)`,
`extract_events(frames, samples, temperature)`, `verify(frames, violation)`, `warmup_async()`.
Frames are `[(t_sec, BGR ndarray)]`. All returned times are seconds of the full video (video mode shifts the
window-relative times). Only rule ids for the current scene are shown when `VLM_SCENE_FILTER=true`.

## 7. Fusion (`cashaudit/fusion.py`)

`fuse(candidates, verifier=None, floors=None) -> List[Violation]`. Floors: per source (`EVENTS_MIN_CONF`,
`VLM_MIN_CONF`), per type (`data/thresholds.json`, written by `tools/calibrate.py`), unseen types
(`OPEN_WORLD_CONF`). The verifier re-scores VLM candidates and may veto event candidates of `VETO_TYPES` below
`VETO_BELOW`. Same-type candidates within `FUSION_MERGE_SEC` merge; the event timecode wins.

## 8. Data files

| File | What | Who changes it |
|---|---|---|
| `data/rules_kb.json` | 20 types: `id, name, scene, cue, refs[{src, point, strength}], examples, clean, seen_in_train` | by PR; `tests/test_kb.py` guards it |
| `data/legal_index.json` | 46 legal points with opening quote and summary; the only allowed citations | by PR; `tools/build_legal_index.py` can fill quotes |
| `data/thresholds.json` | `{"floors": {"T09": 0.71}}` | ML lead via `tools/calibrate.py --write` |
| `data/cameras.json` | per camera, normalized `[x, y]` polygons: `room_floor`, `door`, `zones.{A,B}.safe` | CV (hand-drawn for the training camera) |
| `data/camera_ref.npz` | ORB points + descriptors (no images) of one frame per training-camera session, for registration | CV via `tools/build_camera_ref.py` |
| `data/train_annotations.csv` | clip, file, scene, label, timecode, type | fixed (add own recordings as new rows) |
| `data/train_md5.json` | hashes for the oracle | fixed |

## 9. Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `API_KEYS` | (empty = refuse all) | comma-separated jury keys |
| `ANALYZER` | stub | stub, oracle, pipeline |
| `WORKERS` | 1 | worker threads = GPUs |
| `JOB_BUDGET_SEC` / `HARD_TIMEOUT_SEC` | 270 / 290 | analyzer deadline / watchdog |
| `API_EXTRA_FIELDS` | false | add type_id, confidence, refs to the jury JSON (keep false for the jury) |
| `VLM_BASE_URL` | http://127.0.0.1:8001/v1 | comma-separated, one per GPU |
| `VLM_MODEL` | Qwen/Qwen3-VL-4B-Instruct | served model name (8B: same family, compare with `deploy/vm.sh eval`) |
| `VLM_FPS`, `VLM_MAX_SIDE`, `VLM_MAX_FRAMES` | 2, 848, 32 | what the VLM sees per 16 s window; frames with a longer side are scaled down (0 = native: 1080p overflows the 16k context) |
| `VLM_WINDOW_SEC`, `VLM_STRIDE_SEC` | 16, 8 | sliding windows |
| `VLM_CONCURRENCY` | 8 | windows in flight per video |
| `VLM_INPUT` | images | images or video (experimental, native timestamps) |
| `VLM_MODE` | both | violations, events, both |
| `VLM_EVENT_SAMPLES` | 1 | >1 = majority vote at temperature 0.7 |
| `VLM_SCENE_FILTER` | true | show only the current scene's rules |
| `EVENTS_MIN_CONF`, `VLM_MIN_CONF`, `OPEN_WORLD_CONF`, `VLM_ALONE_CONF` | 0.5, 0.6, 0.85, 0.8 | fusion floors |
| `VETO_TYPES`, `VETO_BELOW` | visible types, 0.35 | VLM veto of event candidates |
| `EVENTS_OWN` | (empty) | types where VLM-only candidates need `VLM_ALONE_CONF` |
| `VLM_FINAL_CONF`, `VLM_SUPPORT_MIN`, `VLM_SUPPORT_TYPES`, `VLM_SUPPORT_SCOPE`, `VLM_SCENE_MATCH` | 0, 1, (all), all, 0 | stricter keeping of VLM-only violations (off by default): verifier P(yes) floor; candidates of the type within `FUSION_MERGE_SEC` (windows/prompts agreeing); types the support rule applies to; `new_camera` = support rule only when the CV layer could not register the camera (`meta.cv.roi.registered` false, `meta.support_rule`); type must belong to the scene used. Try them offline with `deploy/experiments/fusion_sim.py` |
| `MAX_VIOLATIONS` | 5 | per video |
| `KEEP_VIDEOS` | false | uploaded videos are deleted after analysis |
| `CV_ENABLED` | 1 | 0 = skip the CV layer (it is also skipped when torch/rfdetr are missing) |
| `CV_DETECTOR`, `CV_DET_THRESHOLD` | small, 0.3 | RF-DETR size (nano…large) and raw person threshold |
| `CAMERA` | default | entry of `data/cameras.json` to use |
| `CV_STATES`, `CV_STATE_FPS`, `CV_STATE_HOLD`, `CV_STATE_BUDGET_SEC`, `CV_STATE_CONCURRENCY` | 1, 1, 2, 60, `VLM_CONCURRENCY` | object states (#8): crops asked to the VLM per sampled second; a state must hold 2 samples; stop asking after 60 s |
| `CV_ROI`, `CV_ROI_OWL`, `CV_ROI_MODEL` | 1, 1, google/owlv2-large-patch14-ensemble | per-video geometry: registration to the training camera, OWLv2 anchors (safe zone; door/floor when not registered) |
