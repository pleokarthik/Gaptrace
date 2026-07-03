from ragradar_core.schema import RunRecord


def analyze(record: RunRecord) -> dict | None:
    if not record.chunks and not record.final_prompt:
        return None

    chunks_tokens = sum(c.token_count for c in (record.chunks or []))

    history_tokens = sum(t.tokens or 0 for t in (record.history_post or []))

    system_tokens = 0
    headroom = 0
    model_limit = None
    if record.token_budget:
        system_tokens = record.token_budget.system_allocated
        headroom = record.token_budget.headroom
        model_limit = record.token_budget.total_limit

    total = chunks_tokens + history_tokens + system_tokens
    utilisation = (total / model_limit * 100) if model_limit else 0.0

    per_chunk = [
        {"chunk_id": c.chunk_id, "token_count": c.token_count} for c in (record.chunks or [])
    ]

    return {
        "total_tokens": total,
        "chunks_tokens": chunks_tokens,
        "history_tokens": history_tokens,
        "system_tokens": system_tokens,
        "headroom": headroom,
        "model_limit": model_limit,
        "utilisation_pct": round(utilisation, 1),
        "per_chunk": per_chunk,
    }
