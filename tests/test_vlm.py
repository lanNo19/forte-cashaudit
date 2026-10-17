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
