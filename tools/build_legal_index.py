#!/usr/bin/env python3
"""Split a full legal text into points, so citations can quote the real wording.

The service box cannot fetch adilet.zan.kz from code reliably, so: open the act on adilet in a browser,
save the page (HTML or plain text), then run

    python tools/build_legal_index.py --src R231 --file ~/Downloads/V1900019680.html

This writes data/legal_fulltext/R231.json ({"12": "full text of point 12 ...", ...}) and fills any empty
'quote' in data/legal_index.json with the opening of the real point text. Source keys: R231, R14, IS53, AML.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def to_text(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw)
    raw = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</li>|</tr>", "\n", raw)
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = html.unescape(raw)
    return "\n".join(re.sub(r"[ \t ]+", " ", line).strip() for line in raw.splitlines())


def split_points(text: str) -> dict:
    """'12. Text...' starts point 12; lines until the next point belong to it (sub-points '1) ...' included).
    Article-style acts ('Статья 5. ...') are keyed as 'ст. 5'."""
    points, cur = {}, None
    for line in text.splitlines():
        if not line:
            continue
        m = re.match(r"^(\d{1,3})\.\s+(\S.*)$", line)
        a = re.match(r"^Статья\s+(\d{1,3})\.\s*(.*)$", line)
        if a:
            cur = f"ст. {a.group(1)}"
            points[cur] = a.group(2)
        elif m and (cur is None or not cur.startswith("ст.")):
            n = m.group(1)
            # a point number must not go backwards by much (guards against numbered lists inside points)
            if cur is None or int(n) >= int(cur) - 1 or int(n) == 1:
                cur = n
                points[cur] = m.group(2)
                continue
            points[cur] += "\n" + line
        elif cur:
            points[cur] += "\n" + line
    return points


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="source key in data/legal_index.json, e.g. R231")
    ap.add_argument("--file", required=True, help="saved adilet page (.html or .txt)")
    args = ap.parse_args()

    text = to_text(Path(args.file).read_text(encoding="utf-8", errors="ignore"))
    points = split_points(text)
    out_dir = ROOT / "data" / "legal_fulltext"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.src}.json").write_text(json.dumps(points, ensure_ascii=False, indent=1), encoding="utf-8")

    idx_path = ROOT / "data" / "legal_index.json"
    idx = json.loads(idx_path.read_text(encoding="utf-8"))
    filled = 0
    for e in idx["points"]:
        if e["src"] != args.src or e.get("quote"):
            continue
        m = re.match(r"(п\.|ст\.)\s*(\d+)", e["point"])
        key = m.group(2) if m and m.group(1) == "п." else (f"ст. {m.group(2)}" if m else None)
        if key and key in points:
            e["quote"] = points[key].split("\n")[0][:160]
            filled += 1
    idx_path.write_text(json.dumps(idx, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{args.src}: {len(points)} points parsed, {filled} empty quotes filled")


if __name__ == "__main__":
    main()
