"""Per-camera geometry in normalized coordinates (fractions of width/height), never hard-coded pixels.

`data/cameras.json` holds one entry per camera: `room_floor` (where feet of people inside the room can be),
`door` (where people enter and leave), optional `safe`. Issue #7 will propose these automatically from the first
frames; until then the training camera is described by hand and `CAMERA` (env) picks the entry.
An unknown camera falls back to `default`: the whole frame is room and the door is the frame border.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

CAMERAS_FILE = Path(__file__).resolve().parents[2] / "data" / "cameras.json"
EDGE = 0.06  # fallback door: this close to the frame border

Point = Tuple[float, float]


def _inside(p: Point, poly: Sequence[Point]) -> bool:
    """Ray casting point-in-polygon."""
    x, y = p
    inside = False
    n = len(poly)
    for i in range(n):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


@dataclass
class Camera:
    name: str = "default"
    room_floor: Optional[List[Point]] = None
    door: Optional[List[Point]] = None
    zones: Dict[str, Dict[str, List[Point]]] = field(default_factory=dict)  # scene -> {"safe": [...]}
    # 3x3 homography (or 2x3 affine), normalized video coords -> the frame the polygons were drawn on (roi.py)
    to_ref: Optional[np.ndarray] = None

    def ref(self, p: Point) -> Point:
        if self.to_ref is None:
            return p
        v = self.to_ref @ np.array([p[0], p[1], 1.0])
        return (float(v[0] / v[2]), float(v[1] / v[2])) if len(v) == 3 else (float(v[0]), float(v[1]))

    def in_room(self, p: Point) -> bool:
        return _inside(self.ref(p), self.room_floor) if self.room_floor else True

    def near_door(self, p: Point, margin: float = 0.0) -> bool:
        """In the door polygon, or within `margin` of it (normalized units, checked at 4 offsets)."""
        if self.door:
            q = self.ref(p)
            if _inside(q, self.door):
                return True
            return margin > 0 and any(_inside((q[0] + dx, q[1] + dy), self.door)
                                      for dx, dy in ((margin, 0), (-margin, 0), (0, margin), (0, -margin)))
        x, y = p
        return min(x, y, 1 - x, 1 - y) < EDGE

    def zone_of(self, p: Point, scene: Optional[str] = None) -> str:
        """desk | safe | door. Furniture moves between layouts, so zones other than the door are per scene."""
        if self.near_door(p):
            return "door"
        q = self.ref(p)
        for name, poly in self.zones.get(scene or "", {}).items():
            if _inside(q, poly):
                return name
        return "desk"


def load_camera(name: Optional[str] = None, path: Path = CAMERAS_FILE) -> Camera:
    name = name or os.getenv("CAMERA", "default")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return Camera()
    cfg = data.get(name) or data.get("default") or {}
    pts = lambda v: [tuple(map(float, q)) for q in v] if v else None  # noqa: E731
    return Camera(name=name if name in data else "default", room_floor=pts(cfg.get("room_floor")), door=pts(cfg.get("door")),
                  zones={s: {k: pts(v) for k, v in z.items()} for s, z in (cfg.get("zones") or {}).items()})


def normalize(boxes: np.ndarray, w: int, h: int) -> List[Tuple[float, float, float, float]]:
    if len(boxes) == 0:
        return []
    b = np.asarray(boxes, dtype=float) / np.array([w, h, w, h])
    return [tuple(r) for r in np.clip(b, 0, 1).tolist()]
