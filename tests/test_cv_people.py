"""People layer on synthetic boxes (no model, no GPU): the geometry of the training camera in data/cameras.json.

Positions mirror the training clips: the door is on the left (x < 0.3), the seated client's feet are hidden by the desk
at about (0.40, 0.43), the scene-A cashier sits in the foreground at about (0.47, 0.67), corridor people seen through
the glass have their feet above y = 0.44.
"""

from cashaudit.cv.camera import Camera, load_camera
from cashaudit.cv.people import people_events
from cashaudit.events import EventLog
from cashaudit.rules import run_rules

CAM = load_camera("default")
DOOR = (0.15, 0.49)
CLIENT_SEAT = (0.40, 0.43)
CASHIER_A = (0.47, 0.67)
CASHIER_B_DESK = (0.30, 0.47)
CORRIDOR = (0.07, 0.33)


def box(foot, h=0.25):
    x, y = foot
    return (x - 0.05, y - h, x + 0.05, y)


def walk(a, b, n):
    return [(a[0] + (b[0] - a[0]) * i / max(1, n - 1), a[1] + (b[1] - a[1]) * i / max(1, n - 1)) for i in range(n)]


def clip(duration, people):
    """people: list of (t_start, [foot per 0.5 s]). -> frames at 2 fps."""
    frames = []
    for k in range(int(duration * 2) + 1):
        t = k / 2
        boxes = []
        for t_start, path in people:
            i = round((t - t_start) * 2)
            if 0 <= i < len(path) and path[i] is not None:
                boxes.append(box(path[i]))
        frames.append((t, boxes, [0.9] * len(boxes)))
    return frames


def kinds(evs):
    return [(round(e.t, 1), e.kind, e.actor) for e in evs if e.kind.startswith("person_")]


def test_corridor_people_are_ignored():
    frames = clip(10, [(0, [CASHIER_A] * 21), (0, [CORRIDOR] * 21)])
    evs, _, info = people_events(frames, CAM, "A")
    assert info["tracks"] == 1 and kinds(evs) == []


def test_scene_b_outsider_enters_while_cashier_works():  # X-30
    cashier = walk(DOOR, CASHIER_B_DESK, 6) + [CASHIER_B_DESK] * 30
    outsider = walk(DOOR, (0.2, 0.47), 4) + [(0.2, 0.47)] * 10
    evs, scene, _ = people_events(clip(18, [(1, cashier), (12, outsider)]), CAM, None)
    assert scene == "B"
    assert kinds(evs) == [(1.0, "person_enter", "cashier"), (12.0, "person_enter", "other")]
    vs = run_rules(EventLog(evs, 18.0, scene))
    assert [(v.type_id, v.timecode_sec) for v in vs] == [("T02", 12.0)]


def test_scene_a_client_visits_and_third_person():  # X-18, X-4
    cashier = [CASHIER_A] * 81
    client1 = walk(DOOR, CLIENT_SEAT, 4) + [CLIENT_SEAT] * 20 + walk(CLIENT_SEAT, DOOR, 4)  # 3 s .. 17 s
    third = walk(DOOR, (0.12, 0.5), 4)  # enters at 8 s during the visit, stays 2 s
    client2 = walk(DOOR, CLIENT_SEAT, 4) + [CLIENT_SEAT] * 30  # 25 s .. end
    evs, scene, _ = people_events(clip(40, [(0, cashier), (3, client1), (8, third + [(0.12, 0.5)] * 4), (25, client2)]), CAM, None)
    assert scene == "A"
    assert kinds(evs) == [(3.0, "person_enter", "client"), (8.0, "person_enter", "other"), (11.5, "person_exit", "other"),
                          (16.5, "person_exit", "client"), (25.0, "person_enter", "client")]
    assert [v.type_id for v in run_rules(EventLog(evs, 40.0, scene))] == ["T10"]


def test_occlusion_in_the_room_is_not_an_exit():
    seat = [CLIENT_SEAT] * 6 + [None] * 4 + [CLIENT_SEAT] * 15  # hidden for 2 s while seated
    evs, _, _ = people_events(clip(12, [(0, [CASHIER_A] * 25), (0, seat)]), CAM, "A")
    assert kinds(evs) == [(0.0, "person_enter", "client")]


def test_cashier_leaves_through_the_door():  # X-26: zone desk -> door -> out
    path = [CASHIER_B_DESK] * 16 + walk(CASHIER_B_DESK, DOOR, 4)
    evs, _, _ = people_events(clip(15, [(0, path)]), CAM, "B")
    zones = [(round(e.t, 1), e.zone) for e in evs if e.kind == "cashier_zone"]
    assert zones[0] == (0.0, "desk") and zones[-1][1] == "out" and "door" in [z for _, z in zones]
    assert kinds(evs) == [(9.5, "person_exit", "cashier")]


def test_unknown_camera_falls_back_to_whole_frame(tmp_path):
    cam = load_camera("nope", path=tmp_path / "missing.json")
    assert isinstance(cam, Camera) and cam.in_room((0.5, 0.1)) and cam.near_door((0.01, 0.5))


def test_duplicate_box_on_a_seated_person_is_not_a_newcomer():  # X-14 on the A100
    seat = (0.45, 0.47)  # X-14: seated at the desk, away from the door
    cashier = [seat] * 41
    duplicate = [(seat[0] + 0.02, seat[1] + 0.01)] * 10  # second box on her, from 10 s
    evs, _, info = people_events(clip(20, [(0, cashier), (10, duplicate)]), CAM, "B")
    assert info["tracks"] == 1 and kinds(evs) == []


def test_newcomer_first_seen_at_the_seat_is_kept():  # X-4: the client's door frames were missed
    evs, _, _ = people_events(clip(20, [(0, [CASHIER_A] * 41), (5, [CLIENT_SEAT] * 31)]), CAM, "A")
    assert kinds(evs) == [(5.0, "person_enter", "client")]


def test_behaviour_overrules_a_wrong_scene():  # A100: the VLM called the session-5 client desk a cash room
    cashier = [CASHIER_A] * 61
    client = walk(DOOR, CLIENT_SEAT, 4) + [CLIENT_SEAT] * 60  # stays to the end
    evs, scene, info = people_events(clip(30, [(0, cashier), (3, client)]), CAM, "B")
    assert scene == "A" and info["scene_overruled"] == "B"
    assert kinds(evs) == [(3.0, "person_enter", "client")]
    assert run_rules(EventLog(evs, 30.0, scene)) == []  # no T02
