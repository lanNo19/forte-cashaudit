from .base import Analyzer, Deadline  # noqa: F401


def make_analyzer(name: str, device: int = 0):
    if name == "stub":
        from .stub import StubAnalyzer
        return StubAnalyzer(device)
    if name == "oracle":
        from .oracle import OracleAnalyzer
        return OracleAnalyzer(device)
    if name == "pipeline":
        from .pipeline import PipelineAnalyzer
        return PipelineAnalyzer(device)
    raise ValueError(f"unknown ANALYZER={name!r} (stub | oracle | pipeline)")
