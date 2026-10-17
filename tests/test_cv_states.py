"""Object states (#8) without a GPU: fake VLM answers (httpx.MockTransport, as in test_vlm.py), synthetic frames."""

import base64
import json

import cv2
import httpx
import numpy as np

from cashaudit.cv import states
from cashaudit.cv.roi import plausible
from cashaudit.events import Event, EventLog
from cashaudit.rules import run_rules
from cashaudit.vlm import VLMClient


def series(s, t0=0.0):
    """'nnyyy?y' -> [(t, False/True/None)] at 1 s steps."""
    return [(t0 + i, {"y": True, "n": False, "?": None}[c]) for i, c in enumerate(s)]


def kinds(evs):
    return [(e.t, e.kind) for e in evs]


def test_state_change_is_timed_at_its_first_sample_and_needs_two():
    evs = states.answers_to_events(series("nnyyyynn"), "safe_open", "safe_close", zone="safe")
    assert kinds(evs) == [(2.0, "safe_open"), (6.0, "safe_close")]
    assert evs[0].zone == "safe"


def test_single_wrong_answer_and_unknowns_do_not_flip_the_state():
    evs = states.answers_to_events(series("yyynyy??yy"), "screen_on", "screen_locked", obj="laptop")
    assert kinds(evs) == [(0.0, "screen_on")]


def test_opening_off_state_is_not_reported():
    assert states.answers_to_events(series("nnnn"), "door_open", "door_close") == []


def test_lights_off_from_brightness():
    bright, dark = np.full((40, 30, 3), 180, np.uint8), np.full((40, 30, 3), 40, np.uint8)
    frames = [(t / 2, bright if t < 20 else dark) for t in range(30)]
    evs = states.lights_events(frames)
    assert kinds(evs) == [(10.0, "lights_off")]
    assert states.lights_events([(t / 2, bright) for t in range(30)]) == []


def test_crop_enlarges_small_objects():
    img = np.zeros((848, 464, 3), np.uint8)
    c = states.crop(img, (0.4, 0.3, 0.45, 0.33))
    assert max(c.shape[:2]) == states.CROP_SIDE


def fake_vlm(answer_for_brightness):
    """VLM that answers from the first crop's brightness (requests run in parallel, so not from their order)."""
    seen = {"n": 0}

    def handler(req):
        seen["n"] += 1
        content = json.loads(req.content)["messages"][0]["content"]
        text = [c for c in content if c["type"] == "text"][0]["text"]
        assert "Image 1 (safe)" in text and "JSON only" in text
        jpg = base64.b64decode(content[0]["image_url"]["url"].split(",", 1)[1])
        mean = float(cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR).mean())
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(answer_for_brightness(mean))}}]})

    c = VLMClient(base_url="http://fake/v1", model="fake")
    c.http = httpx.Client(transport=httpx.MockTransport(handler))
    return c, seen


def test_vlm_asker_end_to_end_gives_t04_with_the_people_events():  # X-20: safe opened at 2, cashier back at desk at 5
    frames = [(t / 2, np.full((848, 464, 3), 50 if t < 4 else 150, np.uint8)) for t in range(24)]  # 12 s, opens at 2 s
    vlm, seen = fake_vlm(lambda mean: {"safe": "yes" if mean > 100 else "no", "key": "no"})
    boxes = states.roi_boxes({"safe": (0.5, (0.41, 0.29, 0.56, 0.42))}, None)
    evs, info = states.state_events(frames, boxes, states.VLMStateAsker(vlm).ask)
    assert seen["n"] == 12 and info["vlm"]["errors"] == {}  # 1 fps sampling, one request each
    assert kinds(evs) == [(2.0, "safe_open")]
    people = [Event(0, "cashier_zone", zone="safe"), Event(5, "cashier_zone", zone="desk")]
    vs = run_rules(EventLog(people + evs, 12.0, "B"))
    assert [(v.type_id, v.timecode_sec) for v in vs] == [("T04", 5.0)]


def test_vlm_failures_are_counted_not_raised():
    frames = [(float(t), np.zeros((100, 100, 3), np.uint8)) for t in range(4)]

    def boom(crops):
        raise httpx.ConnectError("down")

    evs, info = states.state_events(frames, {"safe": (0.1, 0.1, 0.5, 0.5)}, boom)
    assert evs == [] and info["vlm"]["errors"] == {"ConnectError": 4}


def test_unparseable_answers_count_as_unknown():
    assert states.parse_answers({"safe": "maybe"}, ["safe", "door"]) == {"safe": None, "door": None}


def test_safe_anchor_must_look_like_the_grey_safe():
    img = np.zeros((848, 464, 3), np.uint8)
    img[250:350, 190:260] = (138, 140, 142)  # light grey box
    img[400:470, 60:120] = (30, 30, 30)  # black counting machine
    assert plausible("safe", (190 / 464, 250 / 848, 260 / 464, 350 / 848), img)
    assert not plausible("safe", (60 / 464, 400 / 848, 120 / 464, 470 / 848), img)
