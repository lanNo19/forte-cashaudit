"""Rule knowledge base: violation types, visual cues and legal references (data/rules_kb.json).

Citations are only ever built from this file, so the service cannot invent a legal reference.
References marked "interpretive" are worded as such in the description (criterion 4 penalises
overstated legal claims).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from .config import settings

UNSEEN_TYPE_ID = "T00"


@lru_cache(maxsize=4)
def load_kb(path: Optional[str] = None) -> Dict:
    p = Path(path) if path else settings.kb_path
    kb = json.loads(p.read_text(encoding="utf-8"))
    kb["by_id"] = {t["id"]: t for t in kb["types"]}
    return kb


def type_ids() -> List[str]:
    return [t["id"] for t in load_kb()["types"]]


def type_name(type_id: str) -> str:
    t = load_kb()["by_id"].get(type_id)
    return t["name"] if t else "Иное возможное нарушение кассовой дисциплины"


def type_id_by_name(name: str) -> Optional[str]:
    name = (name or "").strip().lower()
    for t in load_kb()["types"]:
        if t["name"].strip().lower() == name:
            return t["id"]
    return None


def refs(type_id: str) -> List[Dict[str, str]]:
    kb = load_kb()
    t = kb["by_id"].get(type_id)
    if not t:
        return []
    out = []
    for r in t["refs"]:
        src = kb["sources"][r["src"]]
        out.append({"act": src["act"], "title": src["title"], "point": r["point"], "url": src["url"], "strength": r["strength"]})
    return out


def citation_text(type_id: str) -> str:
    """One line for the description field, e.g.
    'Основание: Постановление Правления НБРК от 29.11.2019 № 231 (рег. МЮ № 19680), п. 20 (...).'"""
    rs = refs(type_id)
    if not rs:
        return "Основание: требует ручной проверки нормативной базы (тип нарушения не из перечня)."
    parts = []
    for r in rs:
        s = f"{r['act']}, {r['point']}"
        if r["strength"] == "interpretive":
            s += " — по аналогии/толкованию"
        parts.append(s)
    return "Основание: " + "; ".join(parts) + "."


def seen_in_train(type_id: str) -> bool:
    t = load_kb()["by_id"].get(type_id)
    return bool(t and t.get("seen_in_train", True))


# ---------------------------------------------------------------- legal index lookup
LEGAL_INDEX_PATH = settings.kb_path.parent / "legal_index.json"


@lru_cache(maxsize=2)
def load_legal_index(path: Optional[str] = None) -> Dict:
    p = Path(path) if path else LEGAL_INDEX_PATH
    return json.loads(p.read_text(encoding="utf-8"))


def _stems(text: str) -> set:
    import re
    words = re.findall(r"[а-яёa-z0-9]+", (text or "").lower())
    return {w[:5] for w in words if len(w) >= 4}


def suggest_refs(observed: str, k: int = 2, min_overlap: int = 2) -> List[Dict[str, str]]:
    """Closest indexed points for free text (used for unseen types, T00). Keyword overlap on 5-letter stems:
    crude but deterministic, and it can only return points that exist in the index."""
    idx = load_legal_index()
    q = _stems(observed)
    scored = []
    for e in idx["points"]:
        score = len(q & _stems(e["kw"] + " " + e["summary"]))
        if score >= min_overlap:
            scored.append((score, e))
    scored.sort(key=lambda x: -x[0])
    out = []
    for _, e in scored[:k]:
        src = idx["sources"][e["src"]]
        out.append({"act": src["act"], "title": src["title"], "point": f"{e['point']} ({e['summary'].rstrip('.').lower()})",
                    "url": src["url"], "strength": "interpretive"})
    return out


def describe(type_id: str, observed: str, extra_refs: Optional[List[Dict[str, str]]] = None) -> str:
    """description = what is visible + why it may be a violation + legal basis."""
    observed = (observed or "").strip().rstrip(".")
    t = load_kb()["by_id"].get(type_id)
    why = f"Это может указывать на нарушение: {t['name'].lower()}." if t else "Это может указывать на нарушение кассовой дисциплины."
    if t:
        basis = citation_text(type_id)
    elif extra_refs:
        basis = "Возможное основание (по аналогии, требует проверки юристом): " + "; ".join(f"{r['act']}, {r['point']}" for r in extra_refs) + "."
    else:
        basis = citation_text(type_id)
    return f"{observed}. {why} {basis} Требуется проверка человеком."
