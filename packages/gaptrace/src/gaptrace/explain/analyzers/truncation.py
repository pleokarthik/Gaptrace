from gaptrace_core.schema import RunRecord


def analyze(record: RunRecord) -> dict | None:
    if not record.chunks:
        return None

    truncated = [c for c in record.chunks if c.truncated]
    high_score = [
        c for c in truncated if (c.retrieval_score or 0) > 0.7 or (c.rerank_score or 0) > 0.7
    ]

    if not truncated:
        severity = "none"
    elif high_score:
        severity = "high"
    else:
        severity = "low"

    return {
        "truncated_count": len(truncated),
        "truncated_chunks": [
            {
                "chunk_id": c.chunk_id,
                "score": c.retrieval_score,
                "rerank_score": c.rerank_score,
            }
            for c in truncated
        ],
        "high_score_truncations": len(high_score),
        "severity": severity,
    }
