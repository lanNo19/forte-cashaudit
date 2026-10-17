#!/usr/bin/env python3
"""Fail if a banned dependency or a cloud AI endpoint shows up in code, requirements or deploy files.

Runs in CI and in `make check`. Docs (README, AGENTS.md, docs/) may name banned things to warn about them, so they are
not scanned. To allow a line on purpose, end it with `# licence-ok: <reason>` and say why in the PR.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCAN = ["cashaudit", "tools", "tests"]
SCAN_FILES = ["Dockerfile", "Dockerfile.gpu", "docker-compose.yml", ".env.example", "Makefile"]
SCAN_GLOBS = ["requirements*.txt"]
SKIP = {Path(__file__).resolve(), (ROOT / "tests" / "test_licences.py").resolve()}  # these name banned things on purpose

BANNED = [
    (r"\bultralytics\b", "ultralytics is AGPL-3.0 (also covers its YOLO/RT-DETR/trackers and weights trained with it)"),
    (r"\bboxmot\b", "boxmot is AGPL-3.0; use supervision/trackers ByteTrack"),
    (r"yolo[_-]?world", "YOLO-World is GPL-3.0"),
    (r"rfdetr\[plus\]|RFDETR(XLarge|2XLarge|Atto|Femto|Pico)\b", "RF-DETR Plus sizes are PML 1.0, not Apache"),
    (r"FLUX\.1-(dev|Kontext|Fill)", "FLUX.1-dev family is non-commercial"),
    (r"HunyuanVideo", "HunyuanVideo licence is territory-limited"),
    (r"\bnim\b.*nvcr\.io|nvcr\.io/nim", "NVIDIA NIM needs an AI Enterprise licence"),
    (r"api\.openai\.com|api\.anthropic\.com|generativelanguage\.googleapis\.com|aiplatform\.googleapis\.com|"
     r"dashscope|api\.mistral\.ai|api\.groq\.com|api\.together\.xyz|openrouter\.ai|api\.deepseek\.com",
     "cloud AI endpoint: the brief requires fully local processing"),
    (r"^\s*(import|from)\s+(openai|anthropic|google\.generativeai|google\.genai)\b",
     "cloud AI SDK import (httpx to the local vLLM is the allowed path)"),
]


def files():
    for d in SCAN:
        p = ROOT / d
        if p.exists():
            yield from (f for f in p.rglob("*") if f.is_file() and f.suffix in {".py", ".txt", ".yml", ".yaml", ".toml", ".cfg", ".sh"})
    for name in SCAN_FILES:
        if (ROOT / name).exists():
            yield ROOT / name
    for g in SCAN_GLOBS:
        yield from ROOT.glob(g)


def main() -> int:
    problems = []
    for f in sorted(set(files())):
        if f.resolve() in SKIP or "__pycache__" in f.parts:
            continue
        for i, line in enumerate(f.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if "licence-ok:" in line:
                continue
            for pat, why in BANNED:
                if re.search(pat, line, re.I):
                    problems.append(f"{f.relative_to(ROOT)}:{i}: {why}\n    {line.strip()[:120]}")
    if problems:
        print("Licence / locality check FAILED:\n" + "\n".join(problems))
        return 1
    print("Licence / locality check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
