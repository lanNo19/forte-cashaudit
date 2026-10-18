"""Client for a LOCAL vision-language model behind vLLM's OpenAI-compatible server.

Start one server per GPU on the box (docker-compose.yml does this), e.g.
    CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8001 --gpu-memory-utilization 0.65 \
        --max-model-len 32768 --limit-mm-per-prompt '{"image": 40, "video": 1}'
Nothing here talks to a cloud service; VLM_BASE_URL must point at the box itself.

Two input modes (VLM_INPUT):
- images: each window is a list of JPEGs with a small burned-in "t=12.5s" label (works everywhere);
- video:  each window is one short mp4, so Qwen3-VL can use its native text timestamps (research: better
          temporal grounding). Times come back relative to the window and are shifted here. EXPERIMENTAL: not yet
          run against a real vLLM; compare both modes with tools/eval.py before switching.

Prompts are a first version. The ML lead owns them: iterate with tools/eval.py.
"""

from __future__ import annotations

import base64
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
from typing import List, Optional, Sequence, Tuple

import httpx
import numpy as np

from . import kb
from .config import settings
from .schema import Violation
from .video import burn_timestamp, jpeg_b64

SYSTEM = (
    "You audit overhead CCTV video of a bank cashier's workplace, filmed by one fixed camera. "
    "Report only what is clearly visible. Never infer things the camera cannot show (account balances, "
    "software state, documents' contents). Do not identify people. If nothing clearly matches, return an empty list."
)

EVENT_MENU = "\n".join([
    "person_enter / person_exit: actor = cashier | client | collector | other",
    "handover: obj = cash | id | cash_bag | doc | card; from, to = cashier | client | collector",
    "object_placed / object_removed: obj = cash | cash_bag | doc | tablet | handbag | coat | food | cup; zone = desk | safe",
    "desk_clear: nothing valuable left on the desk",
    "safe_open / safe_close; key_in_lock / key_removed",
    "door_open / door_close (door of the room)",
    "cashier_zone: zone = desk | safe | door | out (where the cashier is now)",
    "counting_start / counting_end: obj = cash; device = counter | hand | detector",
    "screen_on / screen_locked: obj = laptop",
    "consuming: actor = cashier; obj = food | drink",
    "lights_off",
])
EVENT_KINDS = {
    "person_enter", "person_exit", "handover", "object_placed", "object_removed", "desk_clear", "safe_open", "safe_close",
    "key_in_lock", "key_removed", "door_open", "door_close", "cashier_zone", "counting_start", "counting_end",
    "screen_on", "screen_locked", "consuming", "lights_off",
}
SCENES = {
    "A": "a client-service desk: a cashier serves a client sitting across the desk (client chair, small safe, note counter)",
    "B": "a back-office cash room: a cashier works alone at a desk with a note counter and a banknote detector, door to the room visible",
}

Frames = List[Tuple[float, np.ndarray]]


def _card(t: dict) -> str:
    """Rule name + visual cue the VLM reads. VLM_PROMPT_LANG=en uses the English cards (name_en/cue_en in rules_kb);
    jury-facing text (type names, descriptions, legal basis) stays Russian either way."""
    if os.getenv("VLM_PROMPT_LANG", "ru") == "en" and t.get("cue_en"):
        return f"{t['name_en']}. Visual cue: {t['cue_en']}"
    return f"{t['name']}. Visual cue: {t['cue']}"


def _rule_cards(ids: Optional[Sequence[str]] = None) -> str:
    lines = []
    for t in kb.load_kb()["types"]:
        if ids and t["id"] not in ids:
            continue
        lines.append(f"{t['id']}: {_card(t)}")
    lines.append("T00: any other clearly visible breach of cash-handling or cash-room security procedures (use sparingly).")
    return "\n".join(lines)


def scene_type_ids(scene: Optional[str]) -> Optional[List[str]]:
    """Rule cards to show for a scene (research: fewer, relevant rules = fewer false alarms). None = all."""
    if scene not in SCENES:
        return None
    return [t["id"] for t in kb.load_kb()["types"] if t.get("scene") in (scene, None)]


def _extract_json(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {}


def _mp4_data_url(frames: Frames, fps: float) -> str:
    h, w = frames[0][1].shape[:2]
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        path = f.name
    try:
        proc = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:.3f}",
             "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", path],
            input=b"".join(np.ascontiguousarray(img).tobytes() for _, img in frames), capture_output=True, timeout=60,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.decode()[:200])
        with open(path, "rb") as f:
            return "data:video/mp4;base64," + base64.b64encode(f.read()).decode("ascii")
    finally:
        os.remove(path)


class VLMClient:
    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None, timeout: Optional[float] = None,
                 input_mode: Optional[str] = None):
        self.base_url = (base_url or settings.vlm_base_url).rstrip("/")
        self.model = model or settings.vlm_model
        self.input_mode = input_mode or settings.vlm_input
        self.http = httpx.Client(timeout=timeout or settings.vlm_timeout_sec)

    def available(self) -> bool:
        try:
            return self.http.get(f"{self.base_url}/models", timeout=3).status_code == 200
        except httpx.HTTPError:
            return False

    def warmup_async(self) -> None:
        """First requests on Blackwell can stall for minutes while kernels compile: pay that at startup, not on the jury's video."""
        def run():
            try:
                if self.available():
                    self.chat([(0.0, np.zeros((64, 64, 3), np.uint8))] * 2, "Reply with {}.", max_tokens=4)
            except Exception:
                pass
        threading.Thread(target=run, daemon=True, name="vlm-warmup").start()

    # ------------------------------------------------------------------ transport
    def _media(self, frames: Frames) -> List[dict]:
        if self.input_mode == "video" and len(frames) >= 2:
            span = max(frames[-1][0] - frames[0][0], 1e-3)
            fps = (len(frames) - 1) / span
            return [{"type": "video_url", "video_url": {"url": _mp4_data_url(frames, fps)}}]
        return [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{jpeg_b64(burn_timestamp(img, t), quality=85)}"}}
                for t, img in frames]

    def _time_hint(self, frames: Frames) -> str:
        t0, t1 = frames[0][0], frames[-1][0]
        if self.input_mode == "video":
            return f"This clip is {t1 - t0:.1f} s long. Give every time t in seconds from the START OF THIS CLIP."
        return f"These frames cover t={t0:.1f}s to t={t1:.1f}s; each frame shows its time in the top-left corner. Give t as read from the frame label."

    def _abs_t(self, t: float, frames: Frames) -> float:
        t0, t1 = frames[0][0], frames[-1][0]
        if self.input_mode == "video":
            t = t0 + t
        return min(max(t, t0), t1)

    def chat(self, frames: Frames, text: str, max_tokens: int = 700, temperature: float = 0.0) -> str:
        content = self._media(frames) + [{"type": "text", "text": text}]
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        r = self.http.post(f"{self.base_url}/chat/completions", json=body)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    # ------------------------------------------------------------------ tasks
    def classify_scene(self, frames: Frames) -> Optional[str]:
        """One cheap question decides which rule cards to show. Returns 'A', 'B' or None (unsure -> all rules)."""
        pick = [frames[int(i * (len(frames) - 1) / 2)] for i in range(3)] if len(frames) >= 3 else frames
        prompt = ("Which setting is this?\nA: " + SCENES["A"] + "\nB: " + SCENES["B"] +
                  '\nAnswer JSON only: {"scene": "A" | "B", "confidence": 0.0}')
        mode, self.input_mode = self.input_mode, "images"  # 3 stills are enough
        try:
            data = _extract_json(self.chat(pick, prompt, max_tokens=40))
        finally:
            self.input_mode = mode
        scene, conf = str(data.get("scene", "")).upper(), float(data.get("confidence", 0) or 0)
        return scene if scene in SCENES and conf >= 0.6 else None

    def scan_window(self, frames: Frames, ids: Optional[Sequence[str]] = None) -> List[Violation]:
        """One window -> candidate violations (source='vlm')."""
        prompt = (
            f"{self._time_hint(frames)}\n"
            f"Rules (id: name. visual cue):\n{_rule_cards(ids)}\n\n"
            "Return JSON only, no prose:\n"
            '{"violations": [{"type_id": "<rule id from the list>", "t": <seconds, read from the timestamp on the frame>, '
            '"observed": "<1-2 sentences in Russian: exactly what is visible>", "confidence": <0..1>}]}\n'
            'If no rule is clearly broken, return {"violations": []}.\n'
            "t = when the rule is first broken. confidence: 0.9+ only if it is unambiguous."
        )
        data = _extract_json(self.chat(frames, prompt))
        out = []
        for item in data.get("violations", []) or []:
            try:
                tid = str(item.get("type_id", "T00")).upper()
                tid = tid if tid in kb.type_ids() or tid == "T00" else "T00"
                t = self._abs_t(float(item.get("t", 0.0)), frames)
                conf = float(item.get("confidence", 0.5))
                observed = str(item.get("observed", "")).strip() or "Модель отметила возможное нарушение в этом фрагменте"
            except (TypeError, ValueError):
                continue
            refs = kb.refs(tid) if tid != kb.UNSEEN_TYPE_ID else kb.suggest_refs(observed)
            out.append(Violation(type_id=tid, type=kb.type_name(tid), timecode_sec=t,
                                 description=kb.describe(tid, observed, extra_refs=refs if tid == kb.UNSEEN_TYPE_ID else None),
                                 evidence_ts=[t], confidence=conf, source="vlm", refs=refs))
        return out

    def extract_events(self, frames: Frames, samples: int = 1, temperature: float = 0.0) -> List["Event"]:
        """VLM as event extractor (research: segment first, judge with rules second).
        The model fills a CLOSED list of events; rules.py decides what is a violation.
        With samples > 1 it votes: an event is kept if most runs report it (same kind/obj/actor within 3 s)."""
        from .events import Event

        prompt = (
            f"{self._time_hint(frames)} List what HAPPENS, not whether it is allowed.\n"
            f"Allowed events (kind: fields):\n{EVENT_MENU}\n\n"
            "Return JSON only: {\"events\": [{\"kind\": \"handover\", \"t\": 2.5, \"actor\": \"client\", \"obj\": \"cash\", "
            "\"from\": \"client\", \"to\": \"cashier\", \"zone\": null, \"device\": null, \"confidence\": 0.0}]}\n"
            "Report only events you can see; omit anything unclear."
        )
        runs = []
        for _ in range(max(1, samples)):
            data = _extract_json(self.chat(frames, prompt, max_tokens=900, temperature=temperature if samples > 1 else 0.0))
            evs = []
            for it in data.get("events", []) or []:
                kind = str(it.get("kind", "")).strip()
                if kind not in EVENT_KINDS:
                    continue
                try:
                    t = self._abs_t(float(it.get("t", 0.0)), frames)
                    conf = float(it.get("confidence", 0.6))
                except (TypeError, ValueError):
                    continue
                extra = {k: it[k] for k in ("from", "to", "device") if it.get(k)}
                evs.append(Event(t=t, kind=kind, actor=it.get("actor"), obj=it.get("obj"), zone=it.get("zone"),
                                 conf=conf, extra=dict(extra, source="vlm")))
            runs.append(evs)
        if len(runs) == 1:
            return runs[0]
        kept = []
        for e in [e for r in runs for e in r]:
            def same(a, e=e):
                return a.kind == e.kind and a.obj == e.obj and a.actor == e.actor and abs(a.t - e.t) <= 3
            votes = sum(any(same(a) for a in r) for r in runs)
            if votes * 2 > len(runs) and not any(same(k) for k in kept):
                kept.append(e)
        return kept

    def verify(self, frames: Frames, v: Violation) -> Optional[float]:
        """Yes/no check of one candidate on the frames around it. Returns a confidence or None on failure."""
        t = kb.load_kb()["by_id"].get(v.type_id)
        rule = _card(t) if t else v.type
        when = v.timecode_sec - frames[0][0] if self.input_mode == "video" else v.timecode_sec
        if os.getenv("VERIFY_MODE", "logprob") == "logprob":
            return self._verify_logprob(frames, v, rule, when)
        prompt = (
            f"{self._time_hint(frames)}\nRule: {rule}\nClaim: around t={when:.1f}s this rule is broken.\n"
            'Answer JSON only: {"holds": true|false, "confidence": 0.0, "observed": "one sentence in Russian"}'
        )
        try:
            data = _extract_json(self.chat(frames, prompt, max_tokens=200))
        except httpx.HTTPError:
            return None
        if "holds" not in data:
            return None
        conf = float(data.get("confidence", 0.5))
        return conf if data.get("holds") else min(conf, 1 - conf) * 0.5

    def _verify_logprob(self, frames: Frames, v: Violation, rule: str, when: float) -> Optional[float]:
        """VERIFY_MODE=logprob: one-word yes/no answer; returns P(yes) from the token probabilities instead of the
        model's self-reported confidence (unreliable on small VLMs: 0.9-0.98 for real and false candidates alike)."""
        prompt = (
            f"{self._time_hint(frames)}\nRule: {rule}\n"
            f"Look at what happens around t={when:.1f}s. Is this rule actually broken there?\n"
            "Answer with exactly one word: yes or no."
        )
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": self._media(frames) + [{"type": "text", "text": prompt}]}],
            "temperature": 0.0, "max_tokens": 1, "logprobs": True, "top_logprobs": 10,
        }
        try:
            r = self.http.post(f"{self.base_url}/chat/completions", json=body)
            r.raise_for_status()
            top = r.json()["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
            yes = sum(math.exp(x["logprob"]) for x in top if x["token"].strip().lower() in ("yes", "да"))
            no = sum(math.exp(x["logprob"]) for x in top if x["token"].strip().lower() in ("no", "нет"))
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
            return None
        if yes + no <= 0:
            return None
        p = yes / (yes + no)
        print(f"VERIFY {v.type_id} t={v.timecode_sec:.1f} p_yes={p:.3f}", file=sys.stderr, flush=True)
        return p
