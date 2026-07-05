from ragradar_core.schema import RunRecord


def analyze(record: RunRecord) -> dict | None:
    if record.filter is None:
        return None

    filt = record.filter
    candidates = filt.candidate_count
    excluded = filt.excluded_count

    filtered_exclusion_ratio = None
    if candidates is not None and excluded is not None and candidates > 0:
        filtered_exclusion_ratio = excluded / candidates

    return {
        "applied": filt.applied,
        "candidate_count": candidates,
        "excluded_count": excluded,
        "filtered_exclusion_ratio": filtered_exclusion_ratio,
        "filters": filt.filters,
    }
