from ragradar_core.schema import RunRecord


def analyze(record: RunRecord) -> dict | None:
    if not record.chunks:
        return None

    retrieval_scores = [c.retrieval_score for c in record.chunks if c.retrieval_score is not None]
    rerank_scores = [c.rerank_score for c in record.chunks if c.rerank_score is not None]

    if not retrieval_scores and not rerank_scores:
        return None

    top_retrieval = max(retrieval_scores) if retrieval_scores else None
    bottom_retrieval = min(retrieval_scores) if retrieval_scores else None
    top_rerank = max(rerank_scores) if rerank_scores else None
    bottom_rerank = min(rerank_scores) if rerank_scores else None

    rerank_delta = None
    if retrieval_scores and rerank_scores:
        mean_rerank = sum(rerank_scores) / len(rerank_scores)
        mean_retrieval = sum(retrieval_scores) / len(retrieval_scores)
        rerank_delta = round(mean_rerank - mean_retrieval, 4)

    total = len(record.chunks)
    low_rerank = [s for s in rerank_scores if s < 0.5]
    low_score_ratio = round(len(low_rerank) / total, 4) if total else 0.0

    return {
        "retrieval_scores": retrieval_scores,
        "rerank_scores": rerank_scores,
        "top_retrieval": top_retrieval,
        "top_rerank": top_rerank,
        "bottom_retrieval": bottom_retrieval,
        "bottom_rerank": bottom_rerank,
        "rerank_delta": rerank_delta,
        "low_score_ratio": low_score_ratio,
    }
