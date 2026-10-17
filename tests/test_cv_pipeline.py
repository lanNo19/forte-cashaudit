"""The CV layer's scene overrules the VLM's for the rest of the pipeline (A100: X-35..39 called a cash room)."""

import json
import subprocess
import time

import httpx
import pytest

from cashaudit.analyzers import pipeline as pl
from cashaudit.analyzers.base import Deadline
from cashaudit.events import EventLog
from cashaudit.vlm import VLMClient, scene_type_ids


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    p = tmp_path_factory.mktemp("v") / "clip.mov"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=duration=12:size=232x424:rate=30",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(p)], check=True)
    return str(p)


class FakeCV:
    ready, error, info, vlm = True, None, {}, None

    def extract(self, frames, duration, scene):
        return EventLog(events=[], duration=duration, scene="A")


def test_cv_scene_drives_the_scan_cards(video, monkeypatch):
    seen = {"scan_texts": []}

    def handler(req):
        if req.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "fake"}]})
        text = [c for c in json.loads(req.content)["messages"][1]["content"] if c["type"] == "text"][0]["text"]
        if "Which setting is this?" in text:
            return httpx.Response(200, json={"choices": [{"message": {"content": '{"scene": "B", "confidence": 0.9}'}}]})
        seen["scan_texts"].append(text)
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"violations": [], "events": []}'}}]})

    monkeypatch.setattr(pl, "VLM_MODE", "violations")
    an = pl.PipelineAnalyzer()
    an.vlm = VLMClient(base_url="http://fake/v1", model="fake")
    an.vlm.http = httpx.Client(transport=httpx.MockTransport(handler))
    an.events = FakeCV()
    res = an.analyze(video, Deadline(time.time() + 270), lambda p: None)
    assert res.meta["scene"] == "A" and res.meta["scene_vlm"] == "B"
    a_only = set(scene_type_ids("A")) - set(scene_type_ids("B"))
    b_only = set(scene_type_ids("B")) - set(scene_type_ids("A"))
    scans = "\n".join(seen["scan_texts"])
    assert scans and any(t in scans for t in a_only) and not any(f"{t}:" in scans for t in b_only)
