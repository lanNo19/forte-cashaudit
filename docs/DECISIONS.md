# Decisions

One line per decision: date, decision, why, who. Newest first. Reverse a decision with a new line, never by editing.

| When | Decision | Why | Who |
|---|---|---|---|
| 2026-10-17 13:50 | VLM is a sensor: it reports events; `rules.py` decides violations (`VLM_MODE=events` favoured, `both` until measured) | Segment-then-judge beats direct error detection (ZeProM); code rules beat LLM judgement on rule-defined events (NSVS-TL) | research, team |
| 2026-10-17 13:50 | Qwen3-VL-8B-Instruct, one vLLM server per RTX 5090 | Apache-2.0; better temporal grounding than Qwen2.5-VL-7B (48.3 vs 39.3 TimeLens); one server on "all" GPUs used one card | research |
| 2026-10-17 13:50 | Small states (key, screen, safe, door) belong to CV ROI crops, not the VLM | a 5 px key is ~1/40 of a VLM token | research |
| 2026-10-17 13:50 | VLM verifier may veto event candidates only for types it can see | proposer–verifier pattern; avoid VLM vetoing states it cannot resolve | research |
| 2026-10-17 13:50 | No VLM fine-tuning, no paired-clip VLM comparison, no anomaly scorers | fine-tuning needed 9k samples; 7B VLMs at chance on video diff; anomaly scorers target obvious events | research |
| 2026-10-17 13:50 | Stack: Qwen3-VL-8B, vLLM, RF-DETR Nano–Large, Grounding DINO 1.0 / OWLv2 / Florence-2 (offline labelling), SAM 2, supervision/trackers ByteTrack, rtmlib, DINOv2, OpenCV | all Apache-2.0 or MIT | research |
| 2026-10-17 13:50 | Banned: ultralytics, boxmot, YOLO-World, rfdetr[plus], FLUX.1-dev/Kontext, HunyuanVideo, API-only models, NIM; grey zone needs organiser OK: DINOv3, SAM 3, Cosmos-Reason2, Molmo2 | commercial-licence rule (FAQ 24) | research |
| 2026-10-17 13:50 | Own clips from an oblique angle are the only extra data worth collecting | no licence-clean public teller data; unseen views drop to chance in the literature | research |
| 2026-10-17 13:00 | Jury API exactly per section 10; extras only behind `API_EXTRA_FIELDS` | automatic scoring | team |
| 2026-10-17 13:00 | Citations only from `data/legal_index.json`; interpretive references labelled | criterion 4 penalises invented references | team |
| 2026-10-17 12:45 | Hybrid architecture: CV event layer + local VLM + rule engine + fusion | few examples per type, unseen types, new angle | team |
