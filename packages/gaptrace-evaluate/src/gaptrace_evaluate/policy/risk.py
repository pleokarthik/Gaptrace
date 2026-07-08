from gaptrace_evaluate.policy.schema import InputQualityPolicy

_DEFAULT_WEIGHTS = {
    "duplicate_ratio": 0.15,
    "top_chunk_score": 0.25,
    "high_score_truncations": 0.30,
    "token_headroom_pct": 0.15,
    "source_domain_count": 0.10,
    "low_score_chunk_ratio": 0.05,
}


def compute_risk_score(
    input_scores: dict,
    policy: InputQualityPolicy,
    weights: dict | None = None,
) -> float:
    if weights is None:
        weights = _DEFAULT_WEIGHTS

    risk = 0.0

    val = input_scores.get("duplicate_ratio")
    if val is not None and val > policy.max_duplicate_ratio:
        risk += weights["duplicate_ratio"]

    val = input_scores.get("top_chunk_score")
    if val is not None and val < policy.min_top_chunk_score:
        risk += weights["top_chunk_score"]

    val = input_scores.get("high_score_truncations")
    if val is not None and val > policy.max_high_score_truncations:
        risk += weights["high_score_truncations"]

    val = input_scores.get("token_headroom_pct")
    if val is not None and val < policy.min_token_headroom:
        risk += weights["token_headroom_pct"]

    val = input_scores.get("source_domain_count")
    if val is not None and val > policy.max_source_domains:
        risk += weights["source_domain_count"]

    val = input_scores.get("low_score_chunk_ratio")
    if val is not None and val > policy.max_low_score_chunk_ratio:
        risk += weights["low_score_chunk_ratio"]

    return round(risk, 4)
