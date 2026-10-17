"""Synthetic event logs that mirror training clips. They double as the spec for the CV layer:
emit these events and the rule fires at the labelled time."""

from cashaudit.events import Event as E
from cashaudit.events import EventLog
from cashaudit.rules import run_rules


def types(vs):
    return sorted({v.type_id for v in vs})


def test_x4_cash_exposed_to_next_client():  # X-4 @57: client 1 leaves, cash on desk, client 2 enters
    log = EventLog([
        E(3, "person_enter", actor="client"), E(10, "handover", obj="id", extra={"from": "client", "to": "cashier"}),
        E(14, "handover", obj="cash", extra={"from": "client", "to": "cashier"}),
        E(20, "object_placed", obj="cash", zone="desk"), E(52, "person_exit", actor="client"),
        E(57, "person_enter", actor="client"),
    ], duration=72.7, scene="A")
    vs = run_rules(log)
    assert types(vs) == ["T09"]
    assert vs[0].timecode_sec == 57


def test_x5_clean_cash_put_away():  # X-5: same flow, cash into safe before next client
    log = EventLog([
        E(3, "person_enter", actor="client"), E(10, "handover", obj="id", extra={"from": "client", "to": "cashier"}),
        E(14, "handover", obj="cash", extra={"from": "client", "to": "cashier"}),
        E(20, "object_placed", obj="cash", zone="desk"), E(48, "object_removed", obj="cash"),
        E(58, "person_exit", actor="client"), E(64, "person_enter", actor="client"),
    ], duration=71.6, scene="A")
    assert run_rules(log) == []


def test_x2_cash_without_id():  # X-2 @10
    log = EventLog([E(3, "person_enter", actor="client"), E(10, "handover", obj="cash", extra={"from": "client", "to": "cashier"})],
                   duration=17, scene="A")
    vs = run_rules(log)
    assert types(vs) == ["T08"] and vs[0].timecode_sec == 10


def test_x26_cash_left_on_leave():  # X-26 @9: cash on desk, cashier walks out
    log = EventLog([E(0, "cashier_zone", zone="desk"), E(9, "object_placed", obj="cash", zone="desk"),
                    E(10, "cashier_zone", zone="door"), E(11, "person_exit", actor="cashier")], duration=19.8, scene="B")
    vs = run_rules(log)
    assert types(vs) == ["T03"] and abs(vs[0].timecode_sec - 11) <= 2


def test_x28_screen_left_unlocked():  # X-28 @7: gets up at 7, exits ~12, screen still on
    log = EventLog([E(0, "screen_on", obj="laptop"), E(0, "cashier_zone", zone="desk"), E(7, "cashier_zone", zone="door"),
                    E(12, "person_exit", actor="cashier")], duration=16.5, scene="B")
    vs = run_rules(log)
    assert types(vs) == ["T06"] and vs[0].timecode_sec == 7


def test_x29_clean_screen_locked():
    log = EventLog([E(0, "screen_on", obj="laptop"), E(6, "screen_locked", obj="laptop"), E(7, "cashier_zone", zone="door"),
                    E(13, "person_exit", actor="cashier")], duration=17.3, scene="B")
    assert run_rules(log) == []


def test_x32_door_not_closed_and_x31_clean():
    bad = EventLog([E(2, "door_open", zone="door"), E(5, "person_enter", actor="cashier")], duration=21, scene="B")
    good = EventLog([E(2, "door_open", zone="door"), E(5, "person_enter", actor="cashier"), E(8, "door_close", zone="door")],
                    duration=23.8, scene="B")
    assert types(run_rules(bad)) == ["T01"]
    assert run_rules(good) == []


def test_x17_bag_without_docs_and_x16_clean():
    bad = EventLog([E(2, "person_enter", actor="collector"), E(7, "handover", obj="cash_bag", extra={"from": "collector", "to": "cashier"})],
                   duration=14.7, scene="A")
    good = EventLog([E(2, "person_enter", actor="collector"), E(9, "handover", obj="id", extra={"from": "collector", "to": "cashier"}),
                     E(16, "handover", obj="cash_bag", extra={"from": "collector", "to": "cashier"})], duration=22.7, scene="A")
    assert types(run_rules(bad)) == ["T12"]
    assert run_rules(good) == []


def test_x34_client_leaves_during_count():  # X-34 @13
    log = EventLog([E(1, "person_enter", actor="client"), E(7, "handover", obj="id", extra={"from": "client", "to": "cashier"}),
                    E(8, "handover", obj="cash", extra={"from": "client", "to": "cashier"}),
                    E(11, "counting_start", obj="cash", extra={"device": "counter"}), E(13, "person_exit", actor="client"),
                    E(16, "counting_end", obj="cash", extra={"device": "counter"})], duration=17.7, scene="A")
    vs = run_rules(log)
    assert types(vs) == ["T11"] and vs[0].timecode_sec == 13


def test_x20_safe_left_open():  # X-20 @5
    log = EventLog([E(2, "safe_open", zone="safe"), E(2, "cashier_zone", zone="safe"), E(5, "cashier_zone", zone="desk")],
                   duration=11.6, scene="B")
    vs = run_rules(log)
    assert types(vs) == ["T04"] and vs[0].timecode_sec == 5


def test_no_events_no_violations():
    assert run_rules(EventLog([], duration=30)) == []


def test_descriptions_carry_citations():
    log = EventLog([E(3, "person_enter", actor="client"), E(10, "handover", obj="cash", extra={"from": "client", "to": "cashier"})],
                   duration=17, scene="A")
    v = run_rules(log)[0]
    assert "№ 231" in v.description and "п. 19" in v.description
    assert v.refs and v.refs[0]["url"].startswith("https://adilet.zan.kz")


def test_t01_door_confirmed_open_just_after_the_grace_time():  # A100 X-33: entry 5.5 s, door "yes" again from 12 s
    log = EventLog([E(5.5, "person_enter", actor="cashier"), E(12.0, "door_open", zone="door")], duration=14.3, scene="B")
    vs = run_rules(log)
    assert types(vs) == ["T01"] and vs[0].timecode_sec == 5.5
    late = EventLog([E(5.5, "person_enter", actor="cashier"), E(14.0, "door_open", zone="door")], duration=20, scene="B")
    assert run_rules(late) == []  # opened again well after: not this entry's door


def test_t04_needs_the_cashier_seen_at_the_safe():  # A100 X-25: safe opened, CV said "desk" the whole time
    log = EventLog([E(0, "cashier_zone", zone="desk"), E(8, "safe_open", zone="safe"), E(14, "cashier_zone", zone="desk"),
                    E(19, "safe_close", zone="safe")], duration=31.7, scene="B")
    assert run_rules(log) == []
