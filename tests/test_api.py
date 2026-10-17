"""End-to-end API tests with a synthetic .mov (no GPU, no training data needed)."""

import importlib
import os
import subprocess
import time

import pytest


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    p = tmp_path_factory.mktemp("v") / "sample.mov"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=duration=4:size=464x848:rate=30",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(p)], check=True)
    return p


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ.update({"API_KEYS": "test-key", "ANALYZER": "stub", "STUB_DELAY_SEC": "0.2",
                       "DATA_DIR": str(tmp_path_factory.mktemp("var"))})
    import cashaudit.config as config
    importlib.reload(config)
    import cashaudit.jobs as jobs
    importlib.reload(jobs)
    import cashaudit.api as api
    importlib.reload(api)
    from fastapi.testclient import TestClient
    with TestClient(api.app) as c:
        yield c


def _wait(client, jid, key="test-key"):
    for _ in range(100):
        b = client.get(f"/api/v1/jobs/{jid}", headers={"X-API-Key": key}).json()
        if b["status"] != "processing":
            return b
        time.sleep(0.1)
    raise AssertionError("job never finished")


def test_rejects_missing_or_wrong_key(client, video):
    with open(video, "rb") as f:
        assert client.post("/api/v1/analyze", files={"file": ("a.mov", f, "video/quicktime")}).status_code == 401
    with open(video, "rb") as f:
        r = client.post("/api/v1/analyze", headers={"X-API-Key": "nope"}, files={"file": ("a.mov", f, "video/quicktime")})
        assert r.status_code == 401
    assert client.get("/api/v1/jobs/abc", headers={"X-API-Key": "nope"}).status_code == 401


def test_full_flow_matches_spec(client, video):
    with open(video, "rb") as f:
        r = client.post("/api/v1/analyze", headers={"X-API-Key": "test-key"}, files={"file": ("video_017.mov", f, "video/quicktime")})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"job_id", "status"} and body["status"] == "processing" and len(body["job_id"]) == 8
    done = _wait(client, body["job_id"])
    assert done == {"job_id": body["job_id"], "status": "done", "verdict": "no_violation", "violations": []}


def test_unknown_job_404(client):
    assert client.get("/api/v1/jobs/deadbeef", headers={"X-API-Key": "test-key"}).status_code == 404


def test_bad_file_rejected(client, tmp_path):
    junk = tmp_path / "x.mov"
    junk.write_bytes(b"not a video at all")
    with open(junk, "rb") as f:
        r = client.post("/api/v1/analyze", headers={"X-API-Key": "test-key"}, files={"file": ("x.mov", f, "video/quicktime")})
    assert r.status_code == 400
    with open(junk, "rb") as f:
        r = client.post("/api/v1/analyze", headers={"X-API-Key": "test-key"}, files={"file": ("x.txt", f, "text/plain")})
    assert r.status_code == 400


def test_ui_routes(client, video):
    assert client.get("/").status_code == 200
    with open(video, "rb") as f:
        jid = client.post("/ui/analyze", files={"file": ("a.mov", f, "video/quicktime")}).json()["job_id"]
    for _ in range(100):
        b = client.get(f"/ui/jobs/{jid}").json()
        if b["status"] == "done":
            break
        time.sleep(0.1)
    assert b["status"] == "done" and "elapsed_sec" in b
    assert client.get("/healthz").json()["ok"] is True
