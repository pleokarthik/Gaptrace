from gaptrace_core.schema import RunRecord


def analyze(record: RunRecord) -> dict | None:
    if not record.chunks:
        return None

    scores = [
        c.rerank_score if c.rerank_score is not None else c.retrieval_score
        for c in record.chunks
    ]
    usable_scores = [s for s in scores if s is not None]

    variance = None
    if len(usable_scores) > 1:
        mean = sum(usable_scores) / len(usable_scores)
        variance = round(sum((s - mean) ** 2 for s in usable_scores) / len(usable_scores), 4)

    return {
        "usable_score_count": len(usable_scores),
        "chunk_score_variance": variance,
    }
