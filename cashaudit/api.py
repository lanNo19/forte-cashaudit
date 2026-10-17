"""HTTP layer. Jury API exactly as in section 10 of the brief, plus the upload UI.

    POST /api/v1/analyze          multipart field `file` (.mov), header X-API-Key  -> {"job_id", "status"}
    GET  /api/v1/jobs/{job_id}    header X-API-Key -> processing | done (+ verdict, violations) | failed

The UI uses /ui/* (no key: FAQ 27 says the UI needs no auth) and gets the extra fields
(type_id, confidence, source, legal refs) that the jury API leaves out.
"""

from __future__ import annotations

import hmac
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from .analyzers import make_analyzer
from .config import settings
from .jobs import Job, JobManager
from .video import BadVideo, probe

log = logging.getLogger("cashaudit.api")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

STATIC = Path(__file__).parent / "static"
ALLOWED_EXT = {".mov", ".mp4", ".m4v", ".avi", ".mkv"}
SPEC_FIELDS = ("job_id", "status", "verdict", "violations")
SPEC_VIOLATION_FIELDS = ("type", "timecode_sec", "description", "evidence_frames")

manager: Optional[JobManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global manager
    if not settings.api_keys:
        log.warning("API_KEYS is empty: the jury API will refuse every request")
    manager = JobManager(lambda i: make_analyzer(settings.analyzer, device=i), workers=settings.workers)
    manager.start()
    log.info("started: analyzer=%s workers=%d budget=%.0fs", settings.analyzer, settings.workers, settings.job_budget_sec)
    yield
    manager.stop()


app = FastAPI(title="Cash discipline video check", version="0.1.0", lifespan=lifespan)


def require_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")) -> None:
    if not settings.api_keys:
        raise HTTPException(status_code=503, detail="API keys are not configured on the server")
    if not x_api_key or not any(hmac.compare_digest(x_api_key, k) for k in settings.api_keys):
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


async def _accept_upload(file: UploadFile) -> Job:
    name = file.filename or "video.mov"
    ext = os.path.splitext(name)[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(status_code=400, detail=f"unsupported file type {ext or '(none)'}; send a .mov video in field 'file'")
    up_dir = settings.data_dir / "uploads"
    up_dir.mkdir(parents=True, exist_ok=True)
    limit = int(settings.max_upload_mb * 1024 * 1024)
    fd, tmp = tempfile.mkstemp(suffix=ext, dir=up_dir)
    size = 0
    try:
        with os.fdopen(fd, "wb") as out:
            while True:
                chunk = await file.read(1 << 20)
                if not chunk:
                    break
                size += len(chunk)
                if size > limit:
                    raise HTTPException(status_code=413, detail=f"file larger than {settings.max_upload_mb:.0f} MB")
                out.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="empty file")
        try:
            info = probe(tmp)
        except BadVideo as e:
            raise HTTPException(status_code=400, detail=f"not a readable video: {e}")
        job = manager.submit(tmp, name)
        log.info("job %s: %s, %.1f MB, %.1fs", job.id, name, size / 1e6, info.duration)
        return job
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _jury_view(job: Job) -> dict:
    if job.status == "failed":
        return {"job_id": job.id, "status": "failed", "error": job.error or "failed"}
    if job.status != "done" or job.response is None:
        return {"job_id": job.id, "status": "processing"}
    if settings.api_extra_fields:
        return job.response
    r = {k: job.response[k] for k in SPEC_FIELDS}
    r["violations"] = [{k: v[k] for k in SPEC_VIOLATION_FIELDS} for v in job.response["violations"]]
    return r


# ---------------- jury API ----------------
@app.post("/api/v1/analyze", dependencies=[Depends(require_key)])
async def analyze(file: UploadFile = File(...)):
    job = await _accept_upload(file)
    return {"job_id": job.id, "status": "processing"}


@app.get("/api/v1/jobs/{job_id}", dependencies=[Depends(require_key)])
async def job_status(job_id: str):
    job = manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return JSONResponse(_jury_view(job))


# ---------------- UI ----------------
@app.get("/", include_in_schema=False)
async def index():
    if not settings.ui_enabled:
        raise HTTPException(status_code=404)
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


@app.post("/ui/analyze", include_in_schema=False)
async def ui_analyze(file: UploadFile = File(...)):
    if not settings.ui_enabled:
        raise HTTPException(status_code=404)
    job = await _accept_upload(file)
    return {"job_id": job.id, "status": "processing"}


@app.get("/ui/jobs/{job_id}", include_in_schema=False)
async def ui_job(job_id: str):
    job = manager.get(job_id) if settings.ui_enabled else None
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    body = job.response if (job.status == "done" and job.response) else _jury_view(job)
    body = dict(body, elapsed_sec=round((job.finished or time.time()) - job.created, 1), stage=job.status)
    return JSONResponse(body)


@app.get("/healthz", include_in_schema=False)
async def healthz():
    return {"ok": True, **manager.stats()}
