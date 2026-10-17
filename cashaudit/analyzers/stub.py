"""Always answers "no violation" after a short delay. For wiring the service before any model exists."""

import time

from ..config import settings
from ..schema import AnalysisResult
from .base import Deadline, Report


class StubAnalyzer:
    name = "stub"

    def __init__(self, device: int = 0):
        self.device = device

    def analyze(self, video_path: str, deadline: Deadline, report: Report) -> AnalysisResult:
        time.sleep(min(settings.stub_delay_sec, max(0.0, deadline.remaining())))
        return AnalysisResult(meta={"analyzer": self.name})
