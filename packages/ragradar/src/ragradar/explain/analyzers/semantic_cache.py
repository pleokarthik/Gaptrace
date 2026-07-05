from datetime import datetime, timezone

from ragradar_core.schema import RunRecord
from ragradar_evaluate.policy.schema import InputQualityPolicy


def analyze(record: RunRecord, policy: InputQualityPolicy | None = None) -> dict | None:
    if record.cache is None:
        return None
    if policy is None:
        policy = InputQualityPolicy.default()

    cache = record.cache
    borderline_hit = False
    stale_hit = False
    age_seconds = None

    if cache.hit:
        if cache.similarity_score is not None and cache.threshold is not None:
            borderline_hit = (
                abs(cache.similarity_score - cache.threshold) <= policy.cache_borderline_margin
            )
        if cache.cached_at is not None:
            try:
                cached_time = datetime.fromisoformat(cache.cached_at)
                if cached_time.tzinfo is None:
                    cached_time = cached_time.replace(tzinfo=timezone.utc)
                age_seconds = (datetime.now(timezone.utc) - cached_time).total_seconds()
                stale_hit = age_seconds > policy.cache_max_age_seconds
            except ValueError:
                age_seconds = None

    return {
        "checked": cache.checked,
        "hit": cache.hit,
        "similarity_score": cache.similarity_score,
        "threshold": cache.threshold,
        "cached_query": cache.cached_query,
        "cached_at": cache.cached_at,
        "age_seconds": age_seconds,
        "registered": cache.registered,
        "borderline_hit": borderline_hit,
        "stale_hit": stale_hit,
    }
