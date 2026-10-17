"""All settings come from environment variables so the same image runs on a laptop and on the GPU box."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


class Settings:
    def __init__(self) -> None:
        # Comma-separated. The jury key goes here. Empty = API refuses every request (fail closed).
        self.api_keys = [k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip()]
        # stub | oracle | pipeline
        self.analyzer = os.getenv("ANALYZER", "stub")
        # One worker per GPU. Each worker builds its own analyzer instance with device=worker index.
        self.workers = int(os.getenv("WORKERS", "1"))
        # Seconds from upload to answer. The jury counts a video as unprocessed after 300 s.
        self.job_budget_sec = float(os.getenv("JOB_BUDGET_SEC", "270"))
        # If the analyzer is still running this long after upload, the watchdog answers with the
        # latest partial result (or marks the job failed if there is none).
        self.hard_timeout_sec = float(os.getenv("HARD_TIMEOUT_SEC", "290"))
        self.data_dir = Path(os.getenv("DATA_DIR", str(ROOT / "var")))
        self.max_upload_mb = float(os.getenv("MAX_UPLOAD_MB", "300"))
        # Keep the jury API response exactly as in section 10 unless explicitly enabled.
        self.api_extra_fields = _bool("API_EXTRA_FIELDS", False)
        self.ui_enabled = _bool("UI_ENABLED", True)
        self.kb_path = Path(os.getenv("KB_PATH", str(ROOT / "data" / "rules_kb.json")))
        self.annotations_path = Path(os.getenv("ANNOTATIONS_PATH", str(ROOT / "data" / "train_annotations.csv")))
        self.train_md5_path = Path(os.getenv("TRAIN_MD5_PATH", str(ROOT / "data" / "train_md5.json")))
        # Local VLM servers (vLLM, OpenAI-compatible). Must run on the same box: cloud APIs are banned.
        # Comma-separated: one server per GPU. Worker i talks to URL i % n.
        self.vlm_base_urls = [u.strip() for u in os.getenv("VLM_BASE_URL", "http://127.0.0.1:8001/v1").split(",") if u.strip()]
        self.vlm_base_url = self.vlm_base_urls[0]
        self.vlm_model = os.getenv("VLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
        self.vlm_timeout_sec = float(os.getenv("VLM_TIMEOUT_SEC", "60"))
        # What the VLM sees (research: 2-4 fps at native resolution; the old default was ~1 fps at 640 px)
        self.vlm_fps = float(os.getenv("VLM_FPS", "2"))
        self.vlm_max_side = int(os.getenv("VLM_MAX_SIDE", "0"))  # 0 = native resolution
        self.vlm_max_frames = int(os.getenv("VLM_MAX_FRAMES", "32"))  # per 16 s window
        self.vlm_concurrency = int(os.getenv("VLM_CONCURRENCY", "4"))  # windows in flight per video
        # images = JPEG list with a small burned-in time label; video = one mp4 per window (native timestamps, experimental)
        self.vlm_input = os.getenv("VLM_INPUT", "images")
        self.vlm_scene_filter = _bool("VLM_SCENE_FILTER", True)
        self.stub_delay_sec = float(os.getenv("STUB_DELAY_SEC", "2"))
        self.evidence_jpeg_quality = int(os.getenv("EVIDENCE_JPEG_QUALITY", "85"))


settings = Settings()
