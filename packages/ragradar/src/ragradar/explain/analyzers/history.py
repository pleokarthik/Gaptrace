from ragradar_core.schema import RunRecord


def analyze(record: RunRecord) -> dict | None:
    if not record.history_pre and not record.history_post:
        return None

    pre = record.history_pre or []
    post = record.history_post or []

    pre_vals = [t.tokens for t in pre if t.tokens is not None]
    post_vals = [t.tokens for t in post if t.tokens is not None]

    post_contents = {(t.role, t.content) for t in post}
    dropped = [t for t in pre if (t.role, t.content) not in post_contents]

    return {
        "pre_turn_count": len(pre),
        "post_turn_count": len(post),
        "dropped_turn_count": len(dropped),
        "eviction_reason": record.eviction_reason,
        "dropped_turns": dropped,
        "pre_tokens": sum(pre_vals) if pre_vals else None,
        "post_tokens": sum(post_vals) if post_vals else None,
    }
