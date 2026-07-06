from ragradar_core.schema import RunRecord


def analyze(record: RunRecord) -> dict | None:
    if record.chunks is None:
        return None
    requested = record.requested_chunk_count
    if requested is None or requested <= 0:
        return None

    returned = len(record.chunks)

    return {
        "underfill_ratio": round((requested - returned) / requested, 4),
        "requested_chunk_count": requested,
        "returned_chunk_count": returned,
    }
