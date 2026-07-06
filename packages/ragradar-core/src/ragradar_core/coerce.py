"""Coercion of plain-Python inputs into the ragradar_core schema dataclasses.

This is the shared user-input boundary: ragradar_capture's entry points
(Capture methods, capture(), the thread-local proxies) and ragradar_evaluate's
target resolution (evaluate()/check() on a hand-built RunRecord) route
user input through these functions, so naive callers can pass primitives
— shorthand dicts, tuples, a bare int budget — without knowing the
dataclasses exist. The dataclasses (Turn, ChunkRecord, TokenBudget,
CacheEvent, TokenUsage, ToolCallRecord) remain the advanced path and
always pass through untouched; explicitly provided fields always win
over computed defaults.

Token counts are estimated with a deterministic ~4-characters-per-token
heuristic (no tokenizer dependency — ragradar-core stays stdlib-only). Pass
explicit ``tokens`` / ``token_count`` values to override.

All functions are pure and raise TypeError/KeyError on unusable input;
callers decide the failure policy (ragradar_capture swallows/logs by default
and raises in strict mode; ragradar_evaluate raises ValueError).
"""

from collections.abc import Mapping

from ragradar_core.schema import (
    CacheEvent,
    CacheRecord,
    ChunkRecord,
    FilterRecord,
    RunRecord,
    TokenBudget,
    TokenUsage,
    ToolCallRecord,
    Turn,
)


def estimate_tokens(text) -> int:
    """Deterministic token estimate: ~4 characters per token. Pure.

    Returns 0 for None/empty text, at least 1 for any non-empty text.
    Used wherever a token count is derivable but not explicitly given.
    """
    if not text:
        return 0
    return max(1, round(len(text) / 4))


def coerce_turn(turn) -> Turn:
    """Coerce one history turn. Pure.

    Accepts: a Turn (passed through untouched); a ("role", "content")
    pair; a full dict with a "role" key; or the shorthand single-entry
    dict {"user": "..."} / {"assistant": "..."} (optionally with a
    "tokens" entry alongside). Tokens are estimated from the content
    unless explicitly provided.
    """
    if isinstance(turn, Turn):
        return turn
    if isinstance(turn, (tuple, list)):
        if len(turn) != 2:
            raise TypeError(f"Turn tuples must be (role, content), got {len(turn)} items: {turn!r}")
        role, content = turn
        return Turn(role=role, content=content, tokens=estimate_tokens(content))
    if isinstance(turn, Mapping):
        d = dict(turn)
        if "role" in d:
            content = d.get("content", "")
            tokens = d["tokens"] if d.get("tokens") is not None else estimate_tokens(content)
            return Turn(role=d["role"], content=content, tokens=tokens)
        tokens = d.pop("tokens", None)
        if len(d) != 1:
            raise TypeError(
                "Shorthand turn dicts must have exactly one role entry, e.g. "
                f'{{"user": "..."}} (plus an optional "tokens"), got: {turn!r}'
            )
        ((role, content),) = d.items()
        if tokens is None:
            tokens = estimate_tokens(content)
        return Turn(role=role, content=content, tokens=tokens)
    raise TypeError(f"Cannot coerce {type(turn).__name__} into a history turn: {turn!r}")


def coerce_turns(turns) -> list[Turn]:
    """Coerce a sequence of history turns (see coerce_turn). Pure."""
    return [coerce_turn(t) for t in turns]


def coerce_chunk(chunk, index: int) -> ChunkRecord:
    """Coerce one retrieval chunk. Pure.

    Accepts a ChunkRecord (passed through untouched) or a dict; "content"
    is the only required key. Missing boilerplate is filled: chunk_id
    defaults to "chunk_{index}", source_doc_id to "unknown", token_count
    to an estimate of the content. Score/path/flag fields keep their
    dataclass defaults when absent.
    """
    if isinstance(chunk, ChunkRecord):
        return chunk
    if isinstance(chunk, Mapping):
        d = dict(chunk)
        d.setdefault("chunk_id", f"chunk_{index}")
        d.setdefault("source_doc_id", "unknown")
        if d.get("token_count") is None:
            d["token_count"] = estimate_tokens(d.get("content"))
        return ChunkRecord(**d)
    raise TypeError(f"Cannot coerce {type(chunk).__name__} into a chunk: {chunk!r}")


def coerce_chunks(chunks) -> list[ChunkRecord]:
    """Coerce a sequence of retrieval chunks (see coerce_chunk). Pure."""
    return [coerce_chunk(c, i) for i, c in enumerate(chunks)]


def coerce_token_budget(budget, final_prompt=None) -> TokenBudget:
    """Coerce a token budget. Pure.

    Accepts a TokenBudget (passed through untouched), a bare int (the
    total limit), or a dict with at least "total_limit". Allocation
    fields default to 0. A missing headroom is derived, in order of
    preference: total_limit minus the given allocations (when any
    allocation was provided), total_limit minus the estimated
    final_prompt tokens (when a prompt is available), else total_limit.
    Derived headroom may be negative — that is the over-budget signal.
    """
    if isinstance(budget, TokenBudget):
        return budget
    if isinstance(budget, bool) or not isinstance(budget, (int, Mapping)):
        raise TypeError(f"Cannot coerce {type(budget).__name__} into a token budget: {budget!r}")
    d = {"total_limit": budget} if isinstance(budget, int) else dict(budget)

    alloc_keys = ("chunks_allocated", "history_allocated", "system_allocated")
    alloc_given = any(d.get(k) is not None for k in alloc_keys)
    for k in alloc_keys:
        if d.get(k) is None:
            d[k] = 0

    if d.get("headroom") is None:
        total = d["total_limit"]
        if alloc_given:
            d["headroom"] = total - sum(d[k] for k in alloc_keys)
        elif final_prompt:
            d["headroom"] = total - estimate_tokens(final_prompt)
        else:
            d["headroom"] = total
    return TokenBudget(**d)


def coerce_cache_events(events) -> list[CacheEvent]:
    """Coerce cache events. Pure.

    Accepts a mapping of {chunk_id: hit} for the whole call, or a
    sequence whose items are CacheEvents (passed through untouched),
    dicts, or ("chunk_id", hit) pairs.
    """
    if isinstance(events, Mapping):
        return [CacheEvent(chunk_id=k, hit=bool(v)) for k, v in events.items()]
    out = []
    for e in events:
        if isinstance(e, CacheEvent):
            out.append(e)
        elif isinstance(e, Mapping):
            out.append(CacheEvent(**e))
        elif isinstance(e, (tuple, list)) and len(e) == 2:
            out.append(CacheEvent(chunk_id=e[0], hit=bool(e[1])))
        else:
            raise TypeError(f"Cannot coerce {type(e).__name__} into a cache event: {e!r}")
    return out


def coerce_cache_record(cache) -> CacheRecord:
    """Coerce a semantic-cache check. Pure.

    Accepts a CacheRecord (passed through untouched) or a dict with at
    least "checked".
    """
    if isinstance(cache, CacheRecord):
        return cache
    if isinstance(cache, Mapping):
        return CacheRecord(**cache)
    raise TypeError(f"Cannot coerce {type(cache).__name__} into a cache record: {cache!r}")


def coerce_filter_record(filt) -> FilterRecord:
    """Coerce a metadata-filter check. Pure.

    Accepts a FilterRecord (passed through untouched) or a dict with at
    least "applied".
    """
    if isinstance(filt, FilterRecord):
        return filt
    if isinstance(filt, Mapping):
        return FilterRecord(**filt)
    raise TypeError(f"Cannot coerce {type(filt).__name__} into a filter record: {filt!r}")


def coerce_token_usage(usage) -> TokenUsage:
    """Coerce token usage. Pure.

    Accepts a TokenUsage (passed through untouched) or a dict; a missing
    total_tokens is derived as input_tokens + output_tokens.
    """
    if isinstance(usage, TokenUsage):
        return usage
    if isinstance(usage, Mapping):
        d = dict(usage)
        if d.get("total_tokens") is None:
            d["total_tokens"] = d.get("input_tokens", 0) + d.get("output_tokens", 0)
        return TokenUsage(**d)
    raise TypeError(f"Cannot coerce {type(usage).__name__} into token usage: {usage!r}")


def coerce_tool_call(call) -> ToolCallRecord:
    """Coerce one tool call: a ToolCallRecord (untouched) or a dict. Pure."""
    if isinstance(call, ToolCallRecord):
        return call
    if isinstance(call, Mapping):
        return ToolCallRecord(**call)
    raise TypeError(f"Cannot coerce {type(call).__name__} into a tool call: {call!r}")


def coerce_run_record(record: RunRecord) -> RunRecord:
    """Normalized copy of ``record`` with every nested field coerced. Pure.

    RunRecord's constructor stores nested values as given, so a
    hand-built record may carry primitive chunks/turns/budget where the
    metric layers expect dataclasses. This runs each nested field
    through its coercer (dataclass instances pass through untouched)
    and returns a new RunRecord; the input is never mutated.
    """
    return RunRecord(
        query=record.query,
        response=record.response,
        chunks=(coerce_chunks(record.chunks) if record.chunks is not None else None),
        requested_chunk_count=record.requested_chunk_count,
        final_prompt=record.final_prompt,
        token_budget=(
            coerce_token_budget(record.token_budget, record.final_prompt)
            if record.token_budget is not None
            else None
        ),
        history_pre=(coerce_turns(record.history_pre) if record.history_pre is not None else None),
        history_post=(
            coerce_turns(record.history_post) if record.history_post is not None else None
        ),
        eviction_reason=record.eviction_reason,
        cache_events=(
            coerce_cache_events(record.cache_events) if record.cache_events is not None else None
        ),
        tool_calls=(
            [coerce_tool_call(c) for c in record.tool_calls]
            if record.tool_calls is not None
            else None
        ),
        model=record.model,
        token_usage=(
            coerce_token_usage(record.token_usage) if record.token_usage is not None else None
        ),
        cache=(coerce_cache_record(record.cache) if record.cache is not None else None),
        filter=(coerce_filter_record(record.filter) if record.filter is not None else None),
    )
