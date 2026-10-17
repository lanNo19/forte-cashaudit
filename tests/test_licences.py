import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import check_licences  # noqa: E402


def hits(line):
    return [why for pat, why in check_licences.BANNED if re.search(pat, line, re.I)]


def test_banned_things_are_caught():
    assert hits("ultralytics==8.3.0")
    assert hits("from boxmot import ByteTrack")
    assert hits("pip install rfdetr[plus]")
    assert hits("model = YOLOWorld('yolov8s-world')") or hits("yolo_world")
    assert hits("url = 'https://api.openai.com/v1/chat/completions'")
    assert hits("from anthropic import Anthropic")
    assert hits("black-forest-labs/FLUX.1-dev")


def test_allowed_things_pass():
    for ok in ["rfdetr==1.11.2", "supervision==0.31.0", "VLM_BASE_URL=http://vlm0:8001/v1", "import httpx",
               "Qwen/Qwen3-VL-8B-Instruct", "facebook/dinov2-base"]:
        assert not hits(ok), ok


def test_repo_is_clean():
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "check_licences.py")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout
