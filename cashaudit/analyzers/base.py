"""The one interface every analyzer implements.

    analyze(video_path, deadline, report) -> AnalysisResult

- `deadline` is absolute (upload time + budget). Check `deadline.remaining()` between stages and degrade
  (lower fps, skip the verifier) instead of running over. The jury's 5-minute clock starts at upload,
  so time spent waiting in the queue is already gone.
- `report(partial)` publishes a best-so-far result. If the analyzer overruns the hard timeout, the
  watchdog answers the jury with the latest partial instead of a timeout. Call it after every stage.
"""

from __future__ import annotations

import time
from typing import Callable, Protocol

from ..schema import AnalysisResult

Report = Callable[[AnalysisResult], None]


class Deadline:
    def __init__(self, at: float):
        self.at = at

    def remaining(self) -> float:
        return self.at - time.time()

    def expired(self) -> bool:
        return self.remaining() <= 0


class Analyzer(Protocol):
    name: str

    def analyze(self, video_path: str, deadline: Deadline, report: Report) -> AnalysisResult: ...
