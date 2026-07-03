from ragradar_core.schema import RunRecord


def analyze(record: RunRecord) -> dict | None:
    if not record.cache_events:
        return None

    hits = [e for e in record.cache_events if e.hit]
    misses = [e for e in record.cache_events if not e.hit]
    total = len(record.cache_events)

    return {
        "total_events": total,
        "hits": len(hits),
        "misses": len(misses),
        "hit_ratio": len(hits) / total if total else 0.0,
        "hit_chunks": [e.chunk_id for e in hits],
        "miss_chunks": [e.chunk_id for e in misses],
    }
