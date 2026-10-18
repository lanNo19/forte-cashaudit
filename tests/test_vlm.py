"""VLM client and the events mode of the pipeline, against a fake vLLM server (no GPU)."""

import json
import subprocess
import time

import httpx
import numpy as np
import pytest

from cashaudit.analyzers import pipeline as pl
from cashaudit.analyzers.base import Deadline
from cashaudit.vlm import VLMClient


def fake_client(handler):
    c = VLMClient(base_url="http://fake/v1", model="fake")
    c.http = httpx.Client(transport=httpx.MockTransport(handler))
    return c


def reply(content):
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def frames(t0=0.0, n=8):
    return [(t0 + i, np.zeros((64, 48, 3), np.uint8)) for i in range(n)]


def test_extract_events_majority_vote():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        evs = [{"kind": "handover", "t": 3.0, "actor": "client", "obj": "cash", "from": "client", "to": "cashier", "confidence": 0.9}]
        if calls["n"] == 1:
            evs.append({"kind": "lights_off", "t": 5.0, "confidence": 0.9})  # appears in 1 of 3 runs only
        if calls["n"] == 3:
            evs = []
        evs.append({"kind": "not_a_kind", "t": 1.0})  # outside the closed list: dropped
        return reply(json.dumps({"events": evs}))

    evs = fake_client(handler).extract_events(frames(), samples=3, temperature=0.7)
    assert [e.kind for e in evs] == ["handover"]
    assert evs[0].extra["to"] == "cashier" and evs[0].extra["source"] == "vlm"


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    p = tmp_path_factory.mktemp("v") / "clip.mov"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=duration=20:size=232x424:rate=30",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(p)], check=True)
    return str(p)


def test_events_mode_finds_cash_before_id(video, monkeypatch):
    def handler(req):
        if req.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "fake"}]})
        text = [c for c in json.loads(req.content)["messages"][1]["content"] if c["type"] == "text"][0]["text"]
        if "Rule:" in text:
            return reply('{"holds": true, "confidence": 0.85, "observed": "ok"}')
        if "List what HAPPENS" in text and "t=0.0s" in text:
            return reply(json.dumps({"events": [
                {"kind": "person_enter", "t": 1.0, "actor": "client", "confidence": 0.9},
                {"kind": "handover", "t": 6.0, "actor": "client", "obj": "cash", "from": "client", "to": "cashier", "confidence": 0.9},
            ]}))
        return reply('{"events": [], "violations": []}')

    monkeypatch.setattr(pl, "VLM_MODE", "events")
    an = pl.PipelineAnalyzer()
    an.vlm = fake_client(handler)
    res = an.analyze(video, Deadline(time.time() + 270), lambda p: None)
    assert [v.type_id for v in res.violations] == ["T08"]
    v = res.violations[0]
    assert abs(v.timecode_sec - 6.0) < 0.01 and v.source == "vlm" and "п. 19" in v.description


def test_video_mode_sends_mp4_and_shifts_relative_times():
    seen = {}

    def handler(req):
        content = json.loads(req.content)["messages"][1]["content"]
        seen["types"] = [c["type"] for c in content]
        seen["url"] = content[0].get("video_url", {}).get("url", "")
        return reply('{"events": [{"kind": "door_open", "t": 2.0, "confidence": 0.8}]}')

    c = fake_client(handler)
    c.input_mode = "video"
    evs = c.extract_events(frames(t0=40.0, n=8))
    assert seen["types"] == ["video_url", "text"] and seen["url"].startswith("data:video/mp4;base64,")
    assert evs[0].t == 42.0  # window-relative 2.0 s + window start 40 s


def test_scene_question_and_filter():
    from cashaudit.vlm import scene_type_ids
    c = fake_client(lambda req: reply('{"scene": "B", "confidence": 0.9}'))
    assert c.classify_scene(frames()) == "B"
    ids_b = scene_type_ids("B")
    assert "T01" in ids_b and "T09" not in ids_b  # door rule is cash-room, cash-on-desk-for-next-client is client desk
    assert scene_type_ids(None) is None
    unsure = fake_client(lambda req: reply('{"scene": "A", "confidence": 0.4}'))
    assert unsure.classify_scene(frames()) is None


def test_scan_prompt_has_no_concrete_example_values():
    # Qwen3-VL-4B copied the old example ("T09", t 7.0) into its answers on 17 Oct: keep the schema abstract.
    import inspect
    from cashaudit import vlm
    src = inspect.getsource(vlm.VLMClient.scan_window)
    assert '"T09"' not in src and '"t": 7' not in src


def test_english_rule_cards_switch(monkeypatch):
    from cashaudit import vlm
    assert "Visual cue: Кассир" in vlm._rule_cards(["T05"])
    monkeypatch.setenv("VLM_PROMPT_LANG", "en")
    cards = vlm._rule_cards(["T05"])
    assert "T05: Key left in the safe lock. Visual cue: After the cashier closes the safe" in cards


def test_time_text_parts_before_each_frame(monkeypatch):
    c = VLMClient(base_url="http://fake/v1", model="fake", input_mode="images")
    fr = frames(10.0, 3)
    assert [p["type"] for p in c._media(fr)] == ["image_url"] * 3
    monkeypatch.setenv("VLM_TIME_TEXT", "1")
    parts = c._media(fr)
    assert [p["type"] for p in parts] == ["text", "image_url"] * 3
    assert parts[0]["text"] == "<10.0 seconds>" and parts[4]["text"] == "<12.0 seconds>"
    assert "<t seconds>" in c._time_hint(fr)


def test_scan_skip_types(monkeypatch):
    import importlib
    monkeypatch.setenv("VLM_SCAN_SKIP", "T08,T12")
    mod = importlib.reload(pl)
    try:
        assert mod.SCAN_SKIP == {"T08", "T12"}
    finally:
        monkeypatch.delenv("VLM_SCAN_SKIP")
        importlib.reload(pl)


def test_check_style_keeps_only_items_marked_violation(monkeypatch):
    monkeypatch.setenv("VLM_SCAN_STYLE", "check")
    seen = {}

    def handler(req):
        seen["prompt"] = json.loads(req.content)["messages"][1]["content"][-1]["text"]
        return reply(json.dumps({
            "happens": "The cashier counts cash.",
            "checks": {"T05": "violation", "T08": "ok", "T12": "unclear"},
            "violations": [{"type_id": "T05", "t": 3.0, "observed": "Ключ в замке."},
                           {"type_id": "T08", "t": 2.0, "observed": "listed but marked ok"}]}))

    vs = fake_client(handler).scan_window(frames(), ["T05", "T08", "T12"])
    assert [(v.type_id, v.timecode_sec, v.source) for v in vs] == [("T05", 3.0, "vlm")]
    assert '"ok" = the procedure is followed' in seen["prompt"] and "for EVERY checklist id" in seen["prompt"]


def test_check_style_without_summary(monkeypatch):
    monkeypatch.setenv("VLM_SCAN_STYLE", "check")
    monkeypatch.setenv("VLM_CHECK_SUMMARY", "0")
    seen = {}

    def handler(req):
        seen["prompt"] = json.loads(req.content)["messages"][1]["content"][-1]["text"]
        return reply(json.dumps({"checks": {"T05": "violation"}, "violations": [{"type_id": "T05", "t": 3.0, "observed": "Ключ."}]}))

    vs = fake_client(handler).scan_window(frames(), ["T05"])
    assert [v.type_id for v in vs] == ["T05"]
    assert "\"happens\"" not in seen["prompt"] and "Step 1" not in seen["prompt"] and "For EVERY checklist id" in seen["prompt"]


def test_verify_context_window(monkeypatch):
    import importlib
    from cashaudit.schema import Violation
    fr = [(float(t), None) for t in range(0, 40)]
    v = Violation(type_id="T08", type="x", timecode_sec=30.0, description="", evidence_ts=[30.0])
    assert [f[0] for f in pl._verify_frames(fr, v)][0] >= 27.0  # default: +-3 s
    monkeypatch.setenv("VERIFY_BEFORE_SEC", "20")
    mod = importlib.reload(pl)
    try:
        got = [f[0] for f in mod._verify_frames(fr, v)]
        assert got[0] <= 11.0 and got[-1] <= 33.0 and len(got) <= 16
        v.type_id = "T02"  # not a context type: unchanged
        assert [f[0] for f in mod._verify_frames(fr, v)][0] >= 27.0
    finally:
        monkeypatch.delenv("VERIFY_BEFORE_SEC")
        importlib.reload(pl)
