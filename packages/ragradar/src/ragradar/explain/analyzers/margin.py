from ragradar_core.schema import RunRecord
from ragradar_evaluate.policy.schema import InputQualityPolicy


def analyze(record: RunRecord, policy: InputQualityPolicy | None = None) -> dict | None:
    if not record.chunks:
        return None
    if policy is None:
        policy = InputQualityPolicy.default()

    scores = [
        c.rerank_score if c.rerank_score is not None else c.retrieval_score
        for c in record.chunks
    ]
    usable_scores = [s for s in scores if s is not None]
    if len(usable_scores) < 2:
        return None

    usable_scores.sort(reverse=True)

    return {
        "top_second_margin": round(usable_scores[0] - usable_scores[1], 4),
        "threshold_margin": round(usable_scores[0] - policy.min_top_chunk_score, 4),
    }
