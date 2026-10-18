"""The legal side must never cite something that is not in the index (criterion 4)."""

import re

from cashaudit import kb


def _point_numbers(s):
    nums = set()
    for m in re.finditer(r"(п\.|ст\.)\s*(\d+)(?:\s*[–-]\s*(\d+))?", s):
        a = int(m.group(2))
        b = int(m.group(3)) if m.group(3) else a
        nums.update(range(a, b + 1))
    return nums


def test_every_rule_reference_exists_in_the_legal_index():
    idx = kb.load_legal_index()
    have = {}
    for e in idx["points"]:
        have.setdefault(e["src"], set()).update(_point_numbers(e["point"]))
    for t in kb.load_kb()["types"]:
        for r in t["refs"]:
            missing = _point_numbers(r["point"]) - have.get(r["src"], set())
            assert not missing, f"{t['id']} cites {r['src']} {sorted(missing)} which is not in legal_index.json"


def test_suggest_refs_finds_phone_rule():
    refs = kb.suggest_refs("Кассир фотографирует наличные на смартфон с камерой за рабочим столом")
    assert refs and "12" in refs[0]["point"]
    assert all(r["strength"] == "interpretive" for r in refs)


def test_suggest_refs_returns_nothing_for_noise():
    assert kb.suggest_refs("солнечная погода за окном") == []


def test_unseen_type_description_marks_interpretation():
    refs = kb.suggest_refs("кассир ушла с рабочего места во время обслуживания клиента")
    d = kb.describe(kb.UNSEEN_TYPE_ID, "Кассир ушла", extra_refs=refs)
    assert "по аналогии" in d and "Требуется проверка человеком" in d


def test_candidate_types_flagged_unseen():
    assert not kb.seen_in_train("T18") and kb.seen_in_train("T09")
