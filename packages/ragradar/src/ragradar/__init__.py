"""ragradar — the single public import surface.

Users only ever write ``import ragradar``: capture entry points
(capture/start and the staged proxies), evaluation entry points
(check/evaluate/available_metrics), and the schema dataclasses are all
re-exported here. The underlying distributions (ragradar-core,
ragradar-capture, ragradar-evaluate) stay separately installable so a
production pipeline can depend on ragradar-capture alone without
pulling the evaluation stack (scipy/ragas) — but importing their
modules directly is an internal concern, not the public API.
"""

from ragradar_capture import (
    Capture,
    cache,
    capture,
    chunks,
    commit,
    context,
    history,
    response,
    set_strict,
    start,
    tool_call,
)
from ragradar_core.schema import (
    CacheEvent,
    ChunkRecord,
    RunRecord,
    TokenBudget,
    TokenUsage,
    ToolCallRecord,
    Turn,
)
from ragradar_evaluate import (
    CheckResult,
    EvalResult,
    InputQualityPolicy,
    MetricInfo,
    available_metrics,
    check,
    evaluate,
)

__all__ = [
    # Capture
    "Capture",
    "start",
    "capture",
    "set_strict",
    "chunks",
    "context",
    "history",
    "response",
    "cache",
    "tool_call",
    "commit",
    # Evaluation
    "check",
    "evaluate",
    "available_metrics",
    "CheckResult",
    "EvalResult",
    "MetricInfo",
    "InputQualityPolicy",
    # Schema dataclasses (advanced path; primitives coerce everywhere)
    "ChunkRecord",
    "TokenBudget",
    "TokenUsage",
    "Turn",
    "CacheEvent",
    "ToolCallRecord",
    "RunRecord",
]
