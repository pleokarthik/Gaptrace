"""Run record dataclasses shared by every gaptrace package.

Pure data definitions — nothing in this module touches the store. All
dataclasses are decorated with ``_flexible`` so unknown keyword arguments
are silently dropped: instrumentation with extra fields never raises
``TypeError`` in a caller's pipeline, and future fields never break old
readers.
"""

import functools
from dataclasses import asdict, dataclass, fields
from typing import Optional


def _flexible(cls):
    """Make dataclass __init__ accept and ignore unknown keyword arguments."""
    original_init = cls.__init__

    @functools.wraps(original_init)
    def init(self, *args, **kwargs):
        valid = {f.name for f in fields(cls)}
        original_init(self, *args, **{k: v for k, v in kwargs.items() if k in valid})

    cls.__init__ = init
    return cls


@_flexible
@dataclass
class ChunkRecord:
    """One retrieved chunk in a run's context window.

    The advanced/typed path — most callers never construct this directly.
    ``gaptrace.capture()``/``cap.chunks()`` accept plain dicts (only
    ``content`` is required; everything else, including ``chunk_id`` and
    ``source_doc_id``, gets a sensible default) and coerce them into this
    shape internally. Construct ``ChunkRecord`` yourself only if you want
    static typing or are round-tripping data you already have in this form.
    """

    chunk_id: str
    source_doc_id: str
    content: str
    token_count: int
    retrieval_score: Optional[float] = None
    rerank_score: Optional[float] = None
    retrieval_path: Optional[str] = None
    truncated: bool = False
    cache_hit: Optional[bool] = None


@_flexible
@dataclass
class TokenBudget:
    """How a run's token limit was allocated across chunks/history/system.

    Advanced/typed path — ``cap.context(prompt, token_budget=...)`` also
    accepts a bare int (the total limit) or a partial dict; missing
    allocations default to 0 and ``headroom`` is derived when omitted.
    """

    total_limit: int
    chunks_allocated: int
    history_allocated: int
    system_allocated: int
    headroom: int


@_flexible
@dataclass
class TokenUsage:
    """Actual token counts an LLM call reported (as opposed to the budget).

    Advanced/typed path — ``cap.response(text, token_usage=...)`` also
    accepts a dict; a missing ``total_tokens`` is derived as
    ``input_tokens + output_tokens``.
    """

    input_tokens: int
    output_tokens: int
    total_tokens: int


@_flexible
@dataclass
class Turn:
    """One turn of conversation history, before or after eviction.

    Advanced/typed path — ``cap.history(pre=..., post=...)`` also accepts
    shorthand ``{"user": "..."}`` / ``{"assistant": "..."}`` dicts or
    ``(role, content)`` tuples; a missing ``tokens`` count is estimated
    from the content.
    """

    role: str
    content: str
    tokens: Optional[int] = None


@_flexible
@dataclass
class CacheEvent:
    """Whether one chunk was served from cache for this run.

    Advanced/typed path — ``cap.cache(...)`` also accepts a whole-call
    ``{chunk_id: hit}`` mapping or ``(chunk_id, hit)`` pairs.
    """

    chunk_id: str
    hit: bool
    cache_source: Optional[str] = None


@_flexible
@dataclass
class ToolCallRecord:
    """One tool/function call made while producing a run's response.

    Advanced/typed path — ``cap.tool_call(...)`` also accepts a plain
    dict with the same field names.
    """

    tool_name: str
    arguments: dict
    result: Optional[str] = None
    error: Optional[str] = None
    latency_ms: Optional[float] = None


@_flexible
@dataclass
class CacheRecord:
    """Whether this run's query hit a semantic cache, checked before
    retrieval ran — the query-level cache check, distinct from the
    per-chunk ``CacheEvent`` list.

    Advanced/typed path — ``cap.semantic_cache(...)`` builds this from
    its keyword arguments; there is no dict/tuple shorthand since every
    field already maps 1:1 onto a keyword there.
    """

    checked: bool
    hit: bool = False
    similarity_score: Optional[float] = None
    threshold: Optional[float] = None
    cached_query: Optional[str] = None
    cached_at: Optional[str] = None
    registered: bool = False


@_flexible
@dataclass
class FilterRecord:
    """Whether a metadata filter ran before retrieval/scoring, and how many
    candidate chunks it excluded.

    Advanced/typed path — ``cap.metadata_filter(...)`` builds this from
    its keyword arguments. ``filters`` is an opaque dict of the applied
    filter criteria (e.g. {"source": "internal"}) for display only, not
    scored.
    """

    applied: bool
    candidate_count: Optional[int] = None
    excluded_count: Optional[int] = None
    filters: Optional[dict] = None


@_flexible
@dataclass
class RunRecord:
    """The complete captured record of one pipeline run.

    This is what ``gaptrace.capture()``/``Capture`` build up and persist,
    and what ``gaptrace.evaluate()``/``check()`` score. Everything past
    ``query``/``response`` is optional — instrument as much or as little
    of your pipeline as you have. Most callers never construct one by
    hand; it is assembled for you from the primitives passed to
    ``capture()`` or the staged ``Capture`` methods.
    """

    query: str
    response: str
    chunks: Optional[list[ChunkRecord]] = None
    requested_chunk_count: Optional[int] = None
    final_prompt: Optional[str] = None
    token_budget: Optional[TokenBudget] = None
    history_pre: Optional[list[Turn]] = None
    history_post: Optional[list[Turn]] = None
    eviction_reason: Optional[str] = None
    cache_events: Optional[list[CacheEvent]] = None
    tool_calls: Optional[list[ToolCallRecord]] = None
    model: Optional[str] = None
    token_usage: Optional[TokenUsage] = None
    cache: Optional[CacheRecord] = None
    filter: Optional[FilterRecord] = None

    def to_json(self) -> dict:
        """This record as a plain, JSON-serializable dict. Pure."""
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> "RunRecord":
        """Rebuild a ``RunRecord`` from ``to_json()``'s output. Pure.

        Nested dicts are reinflated into their dataclasses (``chunks``
        into ``ChunkRecord``s, etc.) so the result is fully typed, not
        just a dict of dicts.
        """
        data = dict(data)
        if data.get("chunks") is not None:
            data["chunks"] = [ChunkRecord(**c) for c in data["chunks"]]
        if data.get("token_budget") is not None:
            data["token_budget"] = TokenBudget(**data["token_budget"])
        if data.get("history_pre") is not None:
            data["history_pre"] = [Turn(**t) for t in data["history_pre"]]
        if data.get("history_post") is not None:
            data["history_post"] = [Turn(**t) for t in data["history_post"]]
        if data.get("cache_events") is not None:
            data["cache_events"] = [CacheEvent(**e) for e in data["cache_events"]]
        if data.get("tool_calls") is not None:
            data["tool_calls"] = [ToolCallRecord(**t) for t in data["tool_calls"]]
        if data.get("token_usage") is not None:
            data["token_usage"] = TokenUsage(**data["token_usage"])
        if data.get("cache") is not None:
            data["cache"] = CacheRecord(**data["cache"])
        if data.get("filter") is not None:
            data["filter"] = FilterRecord(**data["filter"])
        return cls(**data)
