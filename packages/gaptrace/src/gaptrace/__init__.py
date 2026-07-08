"""gaptrace — the single public import surface.

Users only ever write ``import gaptrace``: capture entry points
(capture/start and the staged proxies), evaluation entry points
(check/evaluate/available_metrics), and the schema dataclasses are all
re-exported here. The underlying distributions (gaptrace-core,
gaptrace-capture, gaptrace-evaluate) stay separately installable so a
production pipeline can depend on gaptrace-capture alone without
pulling the evaluation stack (scipy/ragas) — but importing their
modules directly is an internal concern, not the public API.
"""

from gaptrace_capture import (
    Capture,
    cache,
    capture,
    chunks,
    commit,
    context,
    history,
    metadata_filter,
    response,
    semantic_cache,
    set_strict,
    start,
    tool_call,
)
from gaptrace_core.schema import (
    CacheEvent,
    CacheRecord,
    ChunkRecord,
    FilterRecord,
    RunRecord,
    TokenBudget,
    TokenUsage,
    ToolCallRecord,
    Turn,
)
from gaptrace_evaluate import (
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
    "semantic_cache",
    "metadata_filter",
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
    "CacheRecord",
    "ToolCallRecord",
    "RunRecord",
    "FilterRecord",
]
