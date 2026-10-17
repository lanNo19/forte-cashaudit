"""Event log: the contract between the CV layer (produces events) and the rule checks (consume them).

The CV layer never decides what is a violation; the rules never touch pixels. Each row is one thing
that happened, or a state that started/ended.

Vocabulary (kind -> fields that matter):

  person_enter / person_exit   actor: cashier | client | collector | other     zone: room | service
  handover                     obj: cash | id | bag | doc | card  extra: {"from": role, "to": role}
  object_placed                obj: cash | bag | coat | food | cup | doc | tablet  zone: desk   actor: who placed it
  object_removed               obj, zone (e.g. cash moved from desk into the safe)
  desk_clear                   zone: desk  (nothing valuable left on the desk)
  safe_open / safe_close       zone: safe
  key_in_lock / key_removed    zone: safe
  door_open / door_close       zone: door
  cashier_leaves_desk / cashier_returns
  counting_start / counting_end   obj: cash   extra: {"device": "counter" | "hand" | "detector"}
  screen_on / screen_locked    obj: laptop
  lights_off

States are derived from start/end pairs (see `intervals`). Times are seconds from video start.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

PAIRS = {
    "safe_open": "safe_close",
    "key_in_lock": "key_removed",
    "door_open": "door_close",
    "screen_on": "screen_locked",
    "counting_start": "counting_end",
    "cashier_leaves_desk": "cashier_returns",
}


@dataclass
class Event:
    t: float
    kind: str
    actor: Optional[str] = None
    obj: Optional[str] = None
    zone: Optional[str] = None
    conf: float = 1.0
    extra: Dict = field(default_factory=dict)


@dataclass
class EventLog:
    events: List[Event]
    duration: float
    scene: Optional[str] = None  # "A" client desk | "B" cash room | None unknown

    def sorted(self) -> List[Event]:
        return sorted(self.events, key=lambda e: e.t)

    def of(self, kind: str, **match) -> List[Event]:
        out = []
        for e in self.sorted():
            if e.kind != kind:
                continue
            if any(getattr(e, k) != v for k, v in match.items()):
                continue
            out.append(e)
        return out

    def intervals(self, start_kind: str, end_kind: Optional[str] = None, **match) -> List[Tuple[float, float]]:
        """[start, end) intervals of a state; an open state runs to the end of the video."""
        end_kind = end_kind or PAIRS[start_kind]
        out, open_t = [], None
        for e in self.sorted():
            if any(getattr(e, k) != v for k, v in match.items()):
                continue
            if e.kind == start_kind and open_t is None:
                open_t = e.t
            elif e.kind == end_kind and open_t is not None:
                out.append((open_t, e.t))
                open_t = None
        if open_t is not None:
            out.append((open_t, self.duration))
        return out

    def objects_on_desk(self, t: float, objs: Iterable[str]) -> List[str]:
        """Which of `objs` are on the desk at time t (placed and not removed/cleared since)."""
        objs = set(objs)
        present: Dict[str, bool] = {}
        for e in self.sorted():
            if e.t > t:
                break
            if e.kind == "object_placed" and e.obj in objs and (e.zone in (None, "desk")):
                present[e.obj] = True
            elif e.kind == "object_removed" and e.obj in objs:
                present[e.obj] = False
            elif e.kind == "desk_clear":
                present = {k: False for k in present}
        return [o for o, p in present.items() if p]


def in_any(t: float, ivs: List[Tuple[float, float]]) -> bool:
    return any(a <= t < b for a, b in ivs)
