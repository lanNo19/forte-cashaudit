"""Job queue: one worker thread per GPU, absolute deadlines, a watchdog, results persisted to disk.

Why deadlines are absolute: the jury may upload 5+ videos at once and counts any video without an
answer 300 s after upload as unprocessed. Time spent waiting in the queue counts, so every job carries
`deadline_at = created + JOB_BUDGET_SEC` and the analyzer degrades when it is short on time. If an
analyzer still overruns, the watchdog answers at `hard_at` with the latest partial result.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import secrets
import shutil
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .analyzers import Deadline
from .config import settings
from .schema import AnalysisResult, to_api_json
from .video import frame_at, jpeg_b64, probe

log = logging.getLogger("cashaudit.jobs")


@dataclass
class Job:
    id: str
    filename: str
    video_path: str
    created: float
    deadline_at: float
    hard_at: float
    status: str = "queued"  # queued | processing | done | failed
    error: Optional[str] = None
    started: Optional[float] = None
    finished: Optional[float] = None
    partial: Optional[AnalysisResult] = None
    response: Optional[Dict] = None  # full JSON incl. extra fields; jury API strips extras
    final: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def public_status(self) -> str:
        return "processing" if self.status in ("queued", "processing") else self.status


class JobManager:
    def __init__(self, analyzer_factory: Callable[[int], object], workers: int = 1, store: Optional[Path] = None):
        self.store = Path(store or settings.data_dir) / "jobs"
        self.store.mkdir(parents=True, exist_ok=True)
        self.jobs: Dict[str, Job] = {}
        self.q: "queue.Queue[str]" = queue.Queue()
        self.analyzers = [analyzer_factory(i) for i in range(max(1, workers))]  # build now: fail at startup, not mid-jury
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []
        self._load()

    # ---------- lifecycle ----------
    def start(self) -> None:
        for i, a in enumerate(self.analyzers):
            th = threading.Thread(target=self._worker, args=(i, a), name=f"worker-{i}", daemon=True)
            th.start()
            self._threads.append(th)
        wd = threading.Thread(target=self._watchdog, name="watchdog", daemon=True)
        wd.start()
        self._threads.append(wd)

    def stop(self) -> None:
        self._stop.set()

    # ---------- public ----------
    def submit(self, src_path: str, filename: str) -> Job:
        jid = secrets.token_hex(4)
        while jid in self.jobs:
            jid = secrets.token_hex(4)
        d = self.store / jid
        d.mkdir(parents=True, exist_ok=True)
        ext = os.path.splitext(filename)[1].lower() or ".mov"
        dst = d / f"video{ext}"
        shutil.move(src_path, dst)
        now = time.time()
        job = Job(id=jid, filename=filename, video_path=str(dst), created=now,
                  deadline_at=now + settings.job_budget_sec, hard_at=now + settings.hard_timeout_sec)
        self.jobs[jid] = job
        self._save(job)
        self.q.put(jid)
        return job

    def get(self, jid: str) -> Optional[Job]:
        return self.jobs.get(jid)

    def stats(self) -> Dict:
        by = {}
        for j in self.jobs.values():
            by[j.status] = by.get(j.status, 0) + 1
        return {"queue": self.q.qsize(), "workers": len(self.analyzers), "jobs": by,
                "analyzer": getattr(self.analyzers[0], "name", "?")}

    # ---------- internals ----------
    def _worker(self, idx: int, analyzer) -> None:
        while not self._stop.is_set():
            try:
                jid = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            job = self.jobs.get(jid)
            if not job or job.final:
                continue
            job.status, job.started = "processing", time.time()
            self._save(job)

            def report(partial: AnalysisResult, job=job) -> None:
                job.partial = partial

            try:
                result = analyzer.analyze(job.video_path, Deadline(job.deadline_at), report)
                result.meta.setdefault("worker", idx)
                self._finish(job, result)
            except Exception as e:  # never leave a job hanging
                log.error("job %s failed: %s\n%s", jid, e, traceback.format_exc())
                if job.partial is not None:
                    job.partial.meta.setdefault("warnings", []).append(f"analyzer crashed, partial result: {e}")
                    self._finish(job, job.partial)
                else:
                    self._fail(job, f"analysis error: {e}")

    def _watchdog(self) -> None:
        while not self._stop.is_set():
            now = time.time()
            for job in list(self.jobs.values()):
                if job.final or now < job.hard_at:
                    continue
                if job.partial is not None:
                    job.partial.meta.setdefault("warnings", []).append("hard timeout: answered with partial result")
                    self._finish(job, job.partial)
                else:
                    self._fail(job, "timeout")
            time.sleep(1.0)

    def _finish(self, job: Job, result: AnalysisResult) -> None:
        with job.lock:
            if job.final:
                return
            frames = {}
            for v in result.violations:
                for t in v.evidence_ts:
                    key = round(t, 2)
                    if key in frames:
                        continue
                    try:
                        frames[key] = jpeg_b64(frame_at(job.video_path, t), settings.evidence_jpeg_quality)
                    except Exception as e:
                        log.warning("job %s: evidence frame at %.2f failed: %s", job.id, t, e)
            # a violation whose frames all failed would break the schema: fall back to its timecode, then drop
            for v in result.violations:
                if not any(round(t, 2) in frames for t in v.evidence_ts):
                    try:
                        frames[round(v.timecode_sec, 2)] = jpeg_b64(frame_at(job.video_path, v.timecode_sec))
                        v.evidence_ts = [v.timecode_sec]
                    except Exception:
                        pass
            result.violations = [v for v in result.violations if any(round(t, 2) in frames for t in v.evidence_ts)]
            result.meta["elapsed_since_upload_sec"] = round(time.time() - job.created, 2)
            if "duration" not in result.meta:
                try:
                    result.meta["duration"] = round(probe(job.video_path).duration, 2)
                except Exception:
                    pass
            job.response = to_api_json(job.id, result, frames, extra=True)
            job.status, job.finished, job.final = "done", time.time(), True
            self._save(job)
            self._cleanup_video(job)

    def _fail(self, job: Job, error: str) -> None:
        with job.lock:
            if job.final:
                return
            job.status, job.error, job.finished, job.final = "failed", error, time.time(), True
            self._save(job)
            self._cleanup_video(job)

    def _cleanup_video(self, job: Job) -> None:
        # Test videos may not be kept (brief, section 10).
        if os.getenv("KEEP_VIDEOS", "false").lower() not in ("1", "true", "yes"):
            try:
                os.remove(job.video_path)
            except OSError:
                pass

    def _save(self, job: Job) -> None:
        d = self.store / job.id
        d.mkdir(parents=True, exist_ok=True)
        meta = {k: getattr(job, k) for k in ("id", "filename", "video_path", "created", "deadline_at", "hard_at",
                                             "status", "error", "started", "finished")}
        (d / "job.json").write_text(json.dumps(meta), encoding="utf-8")
        if job.response is not None:
            (d / "result.json").write_text(json.dumps(job.response, ensure_ascii=False), encoding="utf-8")

    def _load(self) -> None:
        now = time.time()
        for d in sorted(self.store.iterdir()):
            f = d / "job.json"
            if not f.exists():
                continue
            try:
                m = json.loads(f.read_text(encoding="utf-8"))
                job = Job(**{k: m[k] for k in ("id", "filename", "video_path", "created", "deadline_at", "hard_at")})
                job.status, job.error, job.started, job.finished = m["status"], m.get("error"), m.get("started"), m.get("finished")
                r = d / "result.json"
                if job.status == "done" and r.exists():
                    job.response, job.final = json.loads(r.read_text(encoding="utf-8")), True
                elif job.status == "failed":
                    job.final = True
                elif os.path.exists(job.video_path) and now < job.hard_at:
                    job.status = "queued"
                    self.q.put(job.id)  # restarted mid-job: try again within the original deadline
                else:
                    job.status, job.error, job.final = "failed", "service restarted", True
                self.jobs[job.id] = job
            except Exception as e:
                log.warning("could not load job %s: %s", d.name, e)
