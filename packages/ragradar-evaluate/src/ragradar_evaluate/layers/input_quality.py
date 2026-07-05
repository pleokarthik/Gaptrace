"""Deterministic input-quality metrics.

Each metric family (relevance, duplicates, truncation, token_efficiency,
coherence) is computed by its own pure function taking a RunRecord and
returning a flat dict of RAW values. Policy checks (check_policy_violations,
and the risk asymmetry noted there) run on raw values; round_values()
produces the rounded copy used for display and persistence — a raw
0.300049 violates a 0.3 threshold even though it displays as 0.3.
score_input_quality() is the dispatcher that runs every family and
appends policy violations — its output shape is unchanged from the
original monolith. The evaluate() facade calls the family functions
individually so selecting one metric never computes the others.
"""

import math
from datetime import datetime, timezone
from typing import Callable

from ragradar_core.schema import RunRecord

from ragradar_evaluate.policy.schema import InputQualityPolicy

SEMANTIC_DUP_THRESHOLD = 0.92


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two vectors (0.0 if either is all-zero). Pure."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _detect_path_dups(chunks) -> int:
    chunk_paths: dict[str, list[str]] = {}
    for c in chunks:
        chunk_paths.setdefault(c.chunk_id, [])
        if c.retrieval_path:
            chunk_paths[c.chunk_id].append(c.retrieval_path)
    return sum(1 for paths in chunk_paths.values() if len(paths) > 1)


def _detect_window_dups(chunks) -> int:
    by_source: dict[str, list] = {}
    for c in chunks:
        by_source.setdefault(c.source_doc_id, []).append(c)

    count = 0
    for group in by_source.values():
        if len(group) < 2:
            continue
        seen: set[tuple[str, str]] = set()
        for i, a in enumerate(group):
            for b in group[i + 1 :]:
                pair = tuple(sorted([a.chunk_id, b.chunk_id]))
                if pair in seen:
                    continue
                if a.content in b.content or b.content in a.content:
                    seen.add(pair)
                    count += 1
                    continue
                tokens_a = set(a.content.lower().split())
                tokens_b = set(b.content.lower().split())
                if tokens_a and tokens_b:
                    overlap = len(tokens_a & tokens_b)
                    total = max(len(tokens_a), len(tokens_b))
                    if overlap / total > 0.5:
                        seen.add(pair)
                        count += 1
    return count


def _detect_semantic_dups(chunks, embedding_fn) -> int:
    if embedding_fn is None:
        return 0
    embeddings = [(c, embedding_fn(c.content)) for c in chunks]
    count = 0
    for i, (ca, va) in enumerate(embeddings):
        for cb, vb in embeddings[i + 1 :]:
            if ca.source_doc_id == cb.source_doc_id:
                continue
            if cosine_similarity(va, vb) > SEMANTIC_DUP_THRESHOLD:
                count += 1
    return count


# ---------------------------------------------------------------------------
# Metric family functions — each pure: RunRecord in, flat value dict out.
# All assume record.chunks is non-empty; callers gate on that. The one
# exception is score_cache_risk, which keys off record.cache instead and
# gates on it internally (returns None rather than assuming a caller did).
# ---------------------------------------------------------------------------


def score_relevance(record: RunRecord, embedding_fn: Callable | None = None) -> dict:
    """Relevance family: per-chunk relevance vs the query. Pure.

    Uses query/chunk cosine similarity when embedding_fn is given,
    otherwise falls back to rerank then retrieval scores. Returns
    relevance_scores, mean_relevance, top_chunk_score (max rerank score,
    None if no rerank scores exist).
    """
    chunks = record.chunks
    relevance_scores: list[float] = []
    if embedding_fn is not None:
        query_vec = embedding_fn(record.query)
        for c in chunks:
            chunk_vec = embedding_fn(c.content)
            relevance_scores.append(cosine_similarity(query_vec, chunk_vec))
    else:
        for c in chunks:
            if c.rerank_score is not None:
                relevance_scores.append(c.rerank_score)
            elif c.retrieval_score is not None:
                relevance_scores.append(c.retrieval_score)

    mean_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0

    rerank_scores = [c.rerank_score for c in chunks if c.rerank_score is not None]
    top_chunk_score = max(rerank_scores) if rerank_scores else None

    return {
        "relevance_scores": relevance_scores,
        "mean_relevance": mean_relevance,
        "top_chunk_score": top_chunk_score,
    }


def score_duplicates(record: RunRecord, embedding_fn: Callable | None = None) -> dict:
    """Duplicates family: path/window(/semantic) duplicate detection. Pure.

    duplicate_ratio counts path + window dups over total chunks; semantic
    dups (cosine > 0.92 across different sources, needs embedding_fn) are
    reported separately.
    """
    chunks = record.chunks
    total = len(chunks)
    path_dup_count = _detect_path_dups(chunks)
    window_dup_count = _detect_window_dups(chunks)
    semantic_dup_count = _detect_semantic_dups(chunks, embedding_fn)
    duplicate_ratio = (path_dup_count + window_dup_count) / total if total else 0.0

    return {
        "duplicate_ratio": duplicate_ratio,
        "path_dup_count": path_dup_count,
        "window_dup_count": window_dup_count,
        "semantic_dup_count": semantic_dup_count,
    }


def score_truncation(record: RunRecord) -> dict:
    """Truncation family: which chunks were trimmed and how bad it is. Pure.

    Severity is "high" when a truncated chunk had a retrieval or rerank
    score above 0.7, "low" for other truncations, "none" otherwise.
    """
    truncated = [c for c in record.chunks if c.truncated]
    truncated_count = len(truncated)
    high_score_truncations = sum(
        1 for c in truncated if (c.retrieval_score or 0) > 0.7 or (c.rerank_score or 0) > 0.7
    )
    if not truncated:
        truncation_severity = "none"
    elif high_score_truncations > 0:
        truncation_severity = "high"
    else:
        truncation_severity = "low"

    return {
        "truncated_count": truncated_count,
        "high_score_truncations": high_score_truncations,
        "truncation_severity": truncation_severity,
    }


def score_token_efficiency(record: RunRecord) -> dict:
    """Token-efficiency family: headroom and low-score chunk ratio. Pure.

    token_headroom_pct is 0.0 when no token_budget was captured (policy
    checks against it are gated on the budget's presence).
    """
    chunks = record.chunks
    total = len(chunks)

    token_headroom_pct = 0.0
    if record.token_budget and record.token_budget.total_limit > 0:
        token_headroom_pct = record.token_budget.headroom / record.token_budget.total_limit

    low_score_chunks = sum(
        1
        for c in chunks
        if (c.rerank_score is not None and c.rerank_score < 0.5)
        or (c.rerank_score is None and c.retrieval_score is not None and c.retrieval_score < 0.5)
    )
    low_score_chunk_ratio = low_score_chunks / total if total else 0.0

    return {
        "token_headroom_pct": token_headroom_pct,
        "low_score_chunk_ratio": low_score_chunk_ratio,
    }


def score_coherence(record: RunRecord) -> dict:
    """Coherence family: source spread and rerank-score variance. Pure.

    score_variance is None with fewer than two rerank scores.
    """
    chunks = record.chunks
    source_domain_count = len({c.source_doc_id for c in chunks})

    rerank_scores = [c.rerank_score for c in chunks if c.rerank_score is not None]
    score_variance = None
    if len(rerank_scores) > 1:
        mean = sum(rerank_scores) / len(rerank_scores)
        score_variance = round(sum((s - mean) ** 2 for s in rerank_scores) / len(rerank_scores), 4)

    return {
        "source_domain_count": source_domain_count,
        "score_variance": score_variance,
    }


def score_cache_risk(record: RunRecord, policy: InputQualityPolicy) -> dict | None:
    """Cache-risk family: does this run's semantic-cache check look
    trustworthy? Pure.

    Unlike the other families, this one is not applicable — and returns
    None — for a record that never checked a semantic cache (record.cache
    is None or record.cache.checked is False); callers should treat that
    the same as "skip this metric", not as a fully-computed clean result.

    borderline_hit flags a hit whose similarity landed within
    policy.cache_borderline_margin of the threshold (right on the fence
    of the cache's own cutoff). stale_hit flags a hit whose cached_at is
    older than policy.cache_max_age_seconds (a reused answer from a long
    time ago). Both are always False for a miss or an unchecked cache.
    """
    cache = record.cache
    if cache is None or not cache.checked:
        return None

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
        "cache_hit": cache.hit,
        "cache_similarity_score": cache.similarity_score,
        "cache_threshold": cache.threshold,
        "cache_age_seconds": age_seconds,
        "cache_registered": cache.registered,
        "borderline_hit": borderline_hit,
        "stale_hit": stale_hit,
    }


# Values rounded (to 4 places) for display/persistence only. Everything
# else a family returns is an int, a string, None, or already rounded at
# computation (score_variance).
_ROUND_KEYS = (
    "mean_relevance",
    "duplicate_ratio",
    "token_headroom_pct",
    "low_score_chunk_ratio",
    "cache_similarity_score",
    "cache_threshold",
)


def round_values(values: dict) -> dict:
    """Rounded copy of family-function ``values`` for display/persistence. Pure.

    Policy checks must run on the raw values, not this copy: a raw
    0.300049 violates a 0.3 threshold even though it rounds to 0.3.
    """
    out = dict(values)
    for key in _ROUND_KEYS:
        if out.get(key) is not None:
            out[key] = round(out[key], 4)
    if out.get("relevance_scores"):
        out["relevance_scores"] = [round(s, 4) for s in out["relevance_scores"]]
    return out


def check_policy_violations(
    values: dict, policy: InputQualityPolicy, record: RunRecord
) -> list[str]:
    """List the policy thresholds breached by the computed ``values``. Pure.

    ``values`` must be the RAW family-function output — pass rounded
    values and thresholds within 5e-5 of a value are checked wrong.
    (compute_risk_score, by contrast, has always run on the rounded
    values; that asymmetry is inherited from the original monolith.)

    Only checks thresholds whose backing value is present in ``values``,
    so a partial (metric-subset) evaluation is never flagged for values
    it did not compute. min_token_headroom additionally requires the
    record to have captured a token_budget; min_chunk_relevance_score
    requires at least one relevance score.
    """
    violations: list[str] = []

    val = values.get("duplicate_ratio")
    if val is not None and val > policy.max_duplicate_ratio:
        violations.append("max_duplicate_ratio")

    val = values.get("top_chunk_score")
    if val is not None and val < policy.min_top_chunk_score:
        violations.append("min_top_chunk_score")

    val = values.get("high_score_truncations")
    if val is not None and val > policy.max_high_score_truncations:
        violations.append("max_high_score_truncations")

    val = values.get("low_score_chunk_ratio")
    if val is not None and val > policy.max_low_score_chunk_ratio:
        violations.append("max_low_score_chunk_ratio")

    val = values.get("token_headroom_pct")
    if val is not None and record.token_budget and val < policy.min_token_headroom:
        violations.append("min_token_headroom")

    val = values.get("source_domain_count")
    if val is not None and val > policy.max_source_domains:
        violations.append("max_source_domains")

    val = values.get("mean_relevance")
    if (
        val is not None
        and val < policy.min_chunk_relevance_score
        and values.get("relevance_scores")
    ):
        violations.append("min_chunk_relevance_score")

    return violations


def score_input_quality(
    record: RunRecord,
    policy: InputQualityPolicy,
    embedding_fn: Callable | None = None,
) -> dict | None:
    """Run every input metric family plus policy checks. Pure.

    Returns None for a chunk-less record. Output shape is the flat dict
    the original monolith produced: every family's values merged, plus
    policy_violations and passes_policy.
    """
    if not record.chunks:
        return None

    raw: dict = {}
    raw.update(score_relevance(record, embedding_fn))
    raw.update(score_duplicates(record, embedding_fn))
    raw.update(score_truncation(record))
    raw.update(score_token_efficiency(record))
    raw.update(score_coherence(record))

    violations = check_policy_violations(raw, policy, record)
    values = round_values(raw)
    values["policy_violations"] = violations
    values["passes_policy"] = len(violations) == 0

    return values
