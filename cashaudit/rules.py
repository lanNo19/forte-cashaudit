"""Rule checks over the event log. One function per violation type; each returns Violations.

Timecode convention learned from the training labels:
- "left something unattended" types (T03, T05, T07): the moment the cashier exits the room;
- T06 (screen not locked): the moment the cashier gets up from the desk;
- ordering types (T08, T11, T12): the moment of the out-of-order action;
- exposure types (T09, T10, T02): the moment the extra person enters.

The CV layer only has to emit events (see events.py). Types not covered here (T13, T15) are left to
the VLM path.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from . import kb
from .events import Event, EventLog, in_any
from .schema import Violation

VALUABLES = ("cash", "cash_bag", "doc", "tablet")
PERSONAL = ("handbag", "coat")
FOOD = ("food", "cup")
ID_DOCS = ("id", "card", "doc")

DOOR_GRACE_SEC = 6.0  # cashier must close the door within this many seconds of entering
DOOR_SAMPLE_SEC = 2.0  # door states come from ~1 fps answers: "still open" may be confirmed up to this much later (A100 X-33)
SAFE_AWAY_SEC = 2.0  # safe open while cashier is back at the desk for longer than this
DETECTOR_WINDOW_SEC = 10.0


def _v(type_id: str, t: float, observed: str, ev: Optional[List[float]] = None, conf: float = 0.8) -> Violation:
    return Violation(
        type_id=type_id, type=kb.type_name(type_id), timecode_sec=float(t), description=kb.describe(type_id, observed),
        evidence_ts=ev or [t], confidence=conf, source="events", refs=kb.refs(type_id),
    )


def _exits(log: EventLog, actor: str = "cashier") -> List[Event]:
    return log.of("person_exit", actor=actor)


def _departure_start(log: EventLog, exit_t: float) -> float:
    """When the cashier got up from the desk before this exit (falls back to the exit time)."""
    leaves = [e.t for e in log.of("cashier_zone") if e.t <= exit_t and e.zone != "desk"]
    desk = [e.t for e in log.of("cashier_zone", zone="desk") if e.t <= exit_t]
    if leaves and desk:
        after_desk = [t for t in leaves if t > max(desk)]
        if after_desk:
            return min(after_desk)
    leave_desk = [e.t for e in log.of("cashier_leaves_desk") if e.t <= exit_t]
    return max(leave_desk) if leave_desk else exit_t


def _client_sessions(log: EventLog):
    """[(enter_t, exit_t)] for each client visit."""
    enters = [e.t for e in log.of("person_enter", actor="client")]
    exits = [e.t for e in log.of("person_exit", actor="client")]
    out = []
    for a in enters:
        later = [b for b in exits if b > a]
        out.append((a, min(later) if later else log.duration))
    return out


def t01_door_not_closed(log: EventLog) -> List[Violation]:
    out = []
    door_open = log.intervals("door_open")
    if not door_open and not log.of("door_close"):
        return out  # CV produced no door states: say nothing rather than guess
    for e in log.of("person_enter", actor="cashier"):
        closed = [c for c in log.of("door_close") if e.t < c.t <= e.t + DOOR_GRACE_SEC]
        late = e.t + DOOR_GRACE_SEC
        if not closed and any(a < late + DOOR_SAMPLE_SEC and b > late for a, b in door_open):
            out.append(_v("T01", e.t, "Кассир вошла в кассовое помещение и прошла к рабочему месту, дверь за собой не закрыла",
                          [e.t, e.t + DOOR_GRACE_SEC / 2, e.t + DOOR_GRACE_SEC], 0.7 * e.conf))
    return out


def t02_outsider(log: EventLog) -> List[Violation]:
    if log.scene not in (None, "B"):
        return []
    cashier_in = bool(log.of("person_enter", actor="cashier")) or bool(log.of("cashier_zone"))
    return [_v("T02", e.t, "В кассовое помещение вошло постороннее лицо (не кассир)", [e.t, e.t + 1.5], 0.75 * e.conf)
            for e in log.of("person_enter", actor="other") if cashier_in]


def t03_valuables_left(log: EventLog) -> List[Violation]:
    out = []
    for e in _exits(log):
        left = log.objects_on_desk(e.t, VALUABLES)
        if left:
            names = {"cash": "наличные", "cash_bag": "инкассаторская сумка", "doc": "документы", "tablet": "планшет/ценности"}
            what = ", ".join(names.get(o, o) for o in left)
            out.append(_v("T03", e.t, f"Кассир покинула помещение, на столе без присмотра остались: {what}", [e.t - 1, e.t, e.t + 2], 0.8 * e.conf))
    return out


def t04_safe_left_open(log: EventLog) -> List[Violation]:
    out = []
    for a, b in log.intervals("safe_open"):
        # "Back at the desk" only counts after the cashier was seen at the safe: where the safe and desk zones look alike
        # from the camera (training room), CV said "desk" while she stood at the open safe (A100 X-25 false T04).
        at_safe = [z.t for z in log.of("cashier_zone", zone="safe") if a - SAFE_AWAY_SEC <= z.t < b]
        if not at_safe:
            continue
        desk = [z.t for z in log.of("cashier_zone", zone="desk") if max(a, min(at_safe)) < z.t < b]
        for t in desk:
            if b - t > SAFE_AWAY_SEC:
                out.append(_v("T04", t, "Кассир отошла от сейфа к рабочему столу, оставив дверцу сейфа открытой", [t, min(b, t + 3)], 0.75))
                break
    return out


def t05_key_left(log: EventLog) -> List[Violation]:
    keys = log.intervals("key_in_lock")
    return [_v("T05", e.t, "Кассир ушла, ключ остался в замке сейфа", [e.t - 1, e.t, e.t + 1], 0.7 * e.conf)
            for e in _exits(log) if in_any(e.t, keys)]


def t06_screen_unlocked(log: EventLog) -> List[Violation]:
    if log.of("lights_off"):
        return []  # end of day is T07
    screens = log.intervals("screen_on")
    out = []
    for e in _exits(log):
        if in_any(e.t, screens):
            t = _departure_start(log, e.t)
            out.append(_v("T06", t, "Кассир ушла с рабочего места, экран ноутбука остался активным и не заблокированным", [t, e.t], 0.7 * e.conf))
    return out


def t07_end_of_day(log: EventLog) -> List[Violation]:
    lights = log.of("lights_off")
    if not lights:
        return []
    tl = lights[0].t
    screens = log.intervals("screen_on")
    exits = [e.t for e in _exits(log) if e.t >= tl - 2]
    t = exits[0] if exits else tl + 2
    if in_any(tl, screens):
        return [_v("T07", t, "В конце рабочего дня кассир выключила свет и уходит, ноутбук оставлен открытым и включённым", [tl, t], 0.7)]
    return []


def t08_cash_before_id(log: EventLog) -> List[Violation]:
    out = []
    for a, b in _client_sessions(log):
        cash = [h for h in log.of("handover", obj="cash") if a <= h.t <= b and h.extra.get("to") == "cashier"]
        if not cash:
            continue
        tc = cash[0].t
        ids = [h for h in log.sorted() if h.kind == "handover" and h.obj in ID_DOCS and a <= h.t < tc]
        if not ids:
            out.append(_v("T08", tc, "Клиент передал наличные, кассир приняла их, документ, удостоверяющий личность, не предъявлялся", [tc - 1, tc, tc + 1], 0.75 * cash[0].conf))
    return out


def t09_cash_exposed(log: EventLog) -> List[Violation]:
    out = []
    enters = log.of("person_enter", actor="client")
    exits = [x.t for x in log.of("person_exit", actor="client")]
    for e in enters[1:]:
        if any(x < e.t for x in exits) and log.objects_on_desk(e.t, ("cash",)):
            out.append(_v("T09", e.t, "Предыдущий клиент ушёл, его наличные остались открыто на столе, в кабинет входит следующий клиент", [e.t - 1, e.t, e.t + 2], 0.8 * e.conf))
    return out


def t10_third_party(log: EventLog) -> List[Violation]:
    out = []
    for a, b in _client_sessions(log):
        for e in log.sorted():
            if e.kind == "person_enter" and e.actor in ("client", "other") and a < e.t < b - 1:
                out.append(_v("T10", e.t, "Во время обслуживания клиента в зону обслуживания вошёл посторонний посетитель", [e.t, e.t + 2], 0.7 * e.conf))
                break
    return out


def t11_left_before_count(log: EventLog) -> List[Violation]:
    counting = log.intervals("counting_start")
    return [_v("T11", e.t, "Клиент покинул кабинет, пока его наличные ещё пересчитываются", [e.t - 1, e.t, e.t + 1], 0.75 * e.conf)
            for e in log.of("person_exit", actor="client") if in_any(e.t, counting)]


def t12_bag_without_docs(log: EventLog) -> List[Violation]:
    out = []
    for h in log.of("handover", obj="cash_bag"):
        if h.extra.get("to") != "cashier":
            continue
        docs = [d for d in log.sorted() if d.kind == "handover" and d.obj in ID_DOCS and h.t - 60 <= d.t < h.t]
        if not docs:
            out.append(_v("T12", h.t, "Инкассатор передал сумку, кассир приняла её без проверки документов инкассатора", [h.t - 1, h.t, h.t + 2], 0.75 * h.conf))
    return out


def t14_no_detector(log: EventLog) -> List[Violation]:
    out = []
    for c in log.of("counting_end"):
        if c.extra.get("device") != "counter":
            continue
        after = [e for e in log.of("counting_start") if c.t <= e.t <= c.t + DETECTOR_WINDOW_SEC]
        if after and not any(e.extra.get("device") == "detector" for e in after) and any(e.extra.get("device") == "hand" for e in after):
            t = [e.t for e in after if e.extra.get("device") == "hand"][0]
            out.append(_v("T14", t, "После счётчика купюры пересчитываются вручную без проверки на детекторе подлинности", [t, t + 2], 0.6))
    return out


def t16_personal_items(log: EventLog) -> List[Violation]:
    return [_v("T16", e.t, "Кассир положила на рабочий стол личные вещи (сумка/верхняя одежда)", [e.t, e.t + 2], 0.75 * e.conf)
            for e in log.sorted() if e.kind == "object_placed" and e.obj in PERSONAL and e.zone in (None, "desk")][:1]


def t17_food(log: EventLog) -> List[Violation]:
    eat = [e for e in log.sorted() if e.kind == "consuming"]
    placed = [e for e in log.sorted() if e.kind == "object_placed" and e.obj in FOOD]
    e = (eat or placed or [None])[0]
    return [_v("T17", e.t, "Кассир ест/пьёт за рабочим столом рядом с наличными и оборудованием", [e.t, e.t + 2], 0.7 * e.conf)] if e else []


RULES: Dict[str, Callable[[EventLog], List[Violation]]] = {
    "T01": t01_door_not_closed, "T02": t02_outsider, "T03": t03_valuables_left, "T04": t04_safe_left_open,
    "T05": t05_key_left, "T06": t06_screen_unlocked, "T07": t07_end_of_day, "T08": t08_cash_before_id,
    "T09": t09_cash_exposed, "T10": t10_third_party, "T11": t11_left_before_count, "T12": t12_bag_without_docs,
    "T14": t14_no_detector, "T16": t16_personal_items, "T17": t17_food,
}


def run_rules(log: EventLog, only: Optional[List[str]] = None) -> List[Violation]:
    out: List[Violation] = []
    for tid, fn in RULES.items():
        if only and tid not in only:
            continue
        out.extend(v.clamp(log.duration) for v in fn(log))
    return out
