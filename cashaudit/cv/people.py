"""People layer: person boxes per frame -> tracks -> roles -> person_enter/exit + cashier_zone events.

Pure numpy, no model: `detector.py` produces the boxes, this file decides what they mean, so the logic is unit-tested
without a GPU (tests/test_cv_people.py).

Ideas that make it robust at 2 fps:
- Only people standing on the room floor count (camera config `room_floor`): people seen through the glass wall are
  in the corridor and their feet never cross the glass line.
- People enter and leave only through the door. A track that starts or ends away from the door is the same person
  losing their id (occlusion, sitting down), so it is stitched to the closest track instead of making a fake
  enter/exit.
- Roles come from time and behaviour only (no faces): the cashier is the person present from the start who stays
  longest, or the first person to enter an empty room. In scene B everybody else is `other`; in scene A the first
  visitor is a `client` and anyone entering during a client visit is `other`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..events import Event
from .camera import Camera

MIN_CONF = 0.45  # person detections below this are ignored (corridor ghosts, reflections)
MAX_GAP_SEC = 3.0  # a track survives this long without a detection
MATCH_DIST = 0.9  # max centre distance between frames, in units of box height
STITCH_SEC = 6.0  # a track that ends away from the door can be continued this long after
MIN_TRACK_SEC = 1.5  # shorter chains that never reach the door are noise
STAY_A_FRAC = 0.85  # scene guess: cashier in the room for this fraction of the video -> client desk (A)
AT_START_SEC = 0.5  # seen in the first two sampled frames = was there before the video started
ZONE_HOLD = 2  # observations a cashier zone must hold before it is reported
REID_GAP_SEC = 1.0  # after a longer gap, motion is not enough: clothing colour must match too
DOOR_MARGIN = 0.06  # first sighting this close to the door = a real entry (normalized units)
GHOST_DIST = 0.6  # a late track starting closer than this (box heights) to a tracked person is a duplicate box
APP_MAX = 0.8  # clothing-colour distance (Bhattacharyya) above which two boxes are different people

Box = Tuple[float, float, float, float]  # x1, y1, x2, y2, normalized to [0, 1]


@dataclass
class Obs:
    t: float
    box: Box
    conf: float
    hist: Optional[np.ndarray] = None  # clothing colour, used only to tell people apart after a gap; not identity

    @property
    def foot(self) -> Tuple[float, float]:
        return ((self.box[0] + self.box[2]) / 2, self.box[3])

    @property
    def centre(self) -> Tuple[float, float]:
        return ((self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2)

    @property
    def height(self) -> float:
        return max(1e-3, self.box[3] - self.box[1])


@dataclass
class Track:
    obs: List[Obs] = field(default_factory=list)
    role: Optional[str] = None

    @property
    def start(self) -> float:
        return self.obs[0].t

    @property
    def end(self) -> float:
        return self.obs[-1].t

    @property
    def conf(self) -> float:
        return float(np.mean([o.conf for o in self.obs]))


def torso_hist(img_bgr: np.ndarray, box: Box) -> Optional[np.ndarray]:
    """Colour histogram (hue, saturation, value) of the upper body: clothing colour only."""
    import cv2

    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = box
    bh, bw = y2 - y1, x2 - x1
    crop = img_bgr[int(h * (y1 + 0.15 * bh)):int(h * (y1 + 0.55 * bh)), int(w * (x1 + 0.2 * bw)):int(w * (x2 - 0.2 * bw))]
    if crop.size < 30:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
    return hist / max(1e-6, float(hist.sum()))


def _app(a: Obs, b: Obs) -> float:
    """Bhattacharyya distance between clothing colours (0 = same); 0 when unknown."""
    if a.hist is None or b.hist is None:
        return 0.0
    return float(np.sqrt(max(0.0, 1.0 - float(np.sum(np.sqrt(a.hist * b.hist))))))


def _same(a: Obs, b: Obs) -> bool:
    """Pose changes the colours a lot, so they only veto a match across a gap of more than REID_GAP_SEC."""
    return abs(b.t - a.t) <= REID_GAP_SEC + 0.01 or _app(a, b) <= APP_MAX


def _dist(a: Obs, b: Obs) -> float:
    (ax, ay), (bx, by) = a.centre, b.centre
    return float(np.hypot(ax - bx, ay - by)) / max(a.height, b.height)


def filter_people(frames: Sequence[tuple], cam: Camera) -> List[Tuple[float, List[Obs]]]:
    """Keep confident person boxes whose feet are on the room floor. Frame = (t, boxes, confs[, hists])."""
    out = []
    for fr in frames:
        t, boxes, confs = fr[:3]
        hists = fr[3] if len(fr) > 3 else [None] * len(boxes)
        keep = [Obs(t, tuple(map(float, b)), float(c), hs) for b, c, hs in zip(boxes, confs, hists) if c >= MIN_CONF]
        out.append((t, [o for o in keep if cam.in_room(o.foot)]))
    return out


def track(frames: List[Tuple[float, List[Obs]]]) -> List[Track]:
    """Greedy nearest-neighbour tracker; good enough for <= 4 people at 2 fps."""
    live: List[Track] = []
    done: List[Track] = []
    for t, obs in frames:
        for tr in [tr for tr in live if t - tr.end > MAX_GAP_SEC]:
            live.remove(tr)
            done.append(tr)
        pairs = sorted((_dist(tr.obs[-1], o), i, j) for i, tr in enumerate(live) for j, o in enumerate(obs))
        used_t, used_o = set(), set()
        for d, i, j in pairs:
            if d > MATCH_DIST or i in used_t or j in used_o or not _same(live[i].obs[-1], obs[j]):
                continue
            live[i].obs.append(obs[j])
            used_t.add(i)
            used_o.add(j)
        live += [Track([o]) for j, o in enumerate(obs) if j not in used_o]
    return sorted(done + live, key=lambda tr: tr.start)


def stitch(tracks: List[Track], cam: Camera, t0: float, t1: float) -> List[Track]:
    """Join a track that ends away from the door to a later track that starts away from the door."""
    tracks = sorted(tracks, key=lambda tr: tr.start)
    changed = True
    while changed:
        changed = False
        for a in tracks:
            if a.end >= t1 - 0.01 or cam.near_door(a.obs[-1].foot):
                continue
            cands = [b for b in tracks if b is not a and 0 < b.start - a.end <= STITCH_SEC
                     and b.start > t0 + 0.01 and not cam.near_door(b.obs[0].foot) and _same(a.obs[-1], b.obs[0])]
            if not cands:
                continue
            b = min(cands, key=lambda b: (b.start - a.end, _dist(a.obs[-1], b.obs[0])))
            a.obs += b.obs
            tracks.remove(b)
            changed = True
            break
    # A track that appears later, away from the door, right on top of a person already tracked is a duplicate box
    # (X-14 on the A100: a second box on the seated cashier became an "other"). A real newcomer appears elsewhere, even
    # when the door frames were missed (X-4: the client is first seen at the seat; X-30: the cashier at the desk).
    long = [tr for tr in tracks if tr.end - tr.start >= MIN_TRACK_SEC]

    def ghost(tr: Track) -> bool:
        if tr.start <= t0 + AT_START_SEC or cam.near_door(tr.obs[0].foot, DOOR_MARGIN):
            return False
        first = tr.obs[0]
        for o in long:
            if o is tr or not (o.start < first.t <= o.end):
                continue
            near = min(o.obs, key=lambda x: abs(x.t - first.t))
            if abs(near.t - first.t) <= MAX_GAP_SEC and _dist(near, first) < GHOST_DIST:
                return True
        return False

    return [tr for tr in long if not ghost(tr)]


def guess_scene(tracks: List[Track], t0: float, t1: float) -> Optional[str]:
    """A = cashier sits there the whole time and visitors come; B = the cashier comes and goes.
    Cashier alone the whole time: unknown (roles do not depend on it then)."""
    cashier = next((tr for tr in tracks if tr.role == "cashier"), None)
    if cashier is None:
        return None
    span = max(1e-3, t1 - t0)
    stays = cashier.start <= t0 + AT_START_SEC and (cashier.end - cashier.start) / span >= STAY_A_FRAC
    if not stays:
        return "B"
    return "A" if any(tr is not cashier for tr in tracks) else None


def assign_roles(tracks: List[Track], scene: Optional[str], t0: float, t1: float) -> Tuple[Optional[str], Optional[str]]:
    """Sets track.role in place. Returns (scene used, the given scene if our guess overruled it).

    The behaviour guess wins over a given scene when they disagree: it was never wrong on the 42 training clips, while
    the VLM called the session-5 client-desk clips (X-35..39) a cash room, which turned every client into an outsider."""
    if not tracks:
        return scene, None
    at_start = [tr for tr in tracks if tr.start <= t0 + AT_START_SEC]
    cashier = max(at_start, key=lambda tr: (tr.end - tr.start, len(tr.obs))) if at_start else tracks[0]
    cashier.role = "cashier"
    guess = guess_scene(tracks, t0, t1)
    overruled = scene if (scene and guess and guess != scene) else None
    scene = guess or scene
    client_until = -1.0
    for tr in sorted(tracks, key=lambda tr: tr.start):
        if tr is cashier:
            continue
        if scene == "A" and tr.start >= client_until:
            tr.role = "client"
            client_until = tr.end
        else:
            tr.role = "other"
    return scene, overruled


def cashier_zones(tr: Track, cam: Camera, t1: float, scene: Optional[str] = None) -> List[Event]:
    """cashier_zone events (desk | safe | door | out), emitted on change after ZONE_HOLD observations."""
    out: List[Event] = []
    current, pending, count, first_t = None, None, 0, tr.start
    for o in tr.obs:
        z = cam.zone_of(o.foot, scene)
        if z == pending:
            count += 1
        else:
            pending, count, first_t = z, 1, o.t
        if pending != current and (count >= ZONE_HOLD or current is None):
            out.append(Event(first_t, "cashier_zone", actor="cashier", zone=pending, conf=round(tr.conf, 2), extra={"source": "cv"}))
            current = pending
    if tr.end < t1 - 0.01:
        out.append(Event(tr.end, "cashier_zone", actor="cashier", zone="out", conf=round(tr.conf, 2), extra={"source": "cv"}))
    return out


def people_events(frames: Sequence[Tuple[float, Sequence[Box], Sequence[float]]], cam: Camera,
                  scene: Optional[str] = None) -> Tuple[List[Event], Optional[str], Dict]:
    """Normalized person boxes per sampled frame -> (events, scene used, debug info)."""
    if not frames:
        return [], scene, {"tracks": 0}
    t0, t1 = frames[0][0], frames[-1][0]
    tracks = stitch(track(filter_people(frames, cam)), cam, t0, t1)
    scene, overruled = assign_roles(tracks, scene, t0, t1)
    events: List[Event] = []
    for tr in tracks:
        c = round(min(0.95, tr.conf), 2)
        # A client already seated at the start still opens a visit (T09/T10 need it); the cashier and outsiders
        # present from the start did not "enter" on camera.
        if tr.start > t0 + 0.01 or tr.role == "client":
            events.append(Event(tr.start, "person_enter", actor=tr.role, zone="room", conf=c, extra={"source": "cv"}))
        if tr.end < t1 - 0.01:
            events.append(Event(tr.end, "person_exit", actor=tr.role, zone="room", conf=c, extra={"source": "cv"}))
        if tr.role == "cashier":
            events += cashier_zones(tr, cam, t1, scene)
    info = {"tracks": len(tracks), "roles": [(tr.role, round(tr.start, 1), round(tr.end, 1)) for tr in tracks], "scene": scene,
            "scene_overruled": overruled}
    return sorted(events, key=lambda e: e.t), scene, info
