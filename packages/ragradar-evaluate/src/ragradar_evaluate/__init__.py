from ragradar_core.schema import (
    CacheEvent,
    CacheRecord,
    ChunkRecord,
    RunRecord,
    TokenBudget,
    TokenUsage,
    ToolCallRecord,
    Turn,
)

from ragradar_evaluate.facade import (
    CheckResult,
    EvalResult,
    MetricInfo,
    available_metrics,
    check,
    evaluate,
)
from ragradar_evaluate.policy.schema import InputQualityPolicy

# NOTE: benchmark machinery (seeding, building, checking, exporting) is
# internal — the CLI's `benchmark` commands drive it, and check() consults
# learned thresholds automatically. It is deliberately absent here.

__all__ = [
    # User tasks
    "check",
    "evaluate",
    "available_metrics",
    # Result / config types
    "CheckResult",
    "EvalResult",
    "MetricInfo",
    "InputQualityPolicy",
    # Re-exported schema dataclasses so users need only one import.
    "ChunkRecord",
    "TokenBudget",
    "TokenUsage",
    "Turn",
    "CacheEvent",
    "CacheRecord",
    "ToolCallRecord",
    "RunRecord",
]
