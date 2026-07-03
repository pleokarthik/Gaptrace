from dataclasses import asdict

from ragradar_core.schema import (
    CacheEvent,
    ChunkRecord,
    RunRecord,
    TokenBudget,
    TokenUsage,
    ToolCallRecord,
    Turn,
)


def _full_record():
    return RunRecord(
        query="what is RRF?",
        response="RRF is reciprocal rank fusion.",
        chunks=[
            ChunkRecord(
                chunk_id="c1",
                source_doc_id="doc1",
                content="RRF combines scores",
                token_count=50,
                retrieval_score=0.9,
                rerank_score=0.85,
                retrieval_path="hybrid",
                truncated=False,
                cache_hit=True,
            ),
            ChunkRecord(
                chunk_id="c2",
                source_doc_id="doc2",
                content="BM25 baseline",
                token_count=30,
                retrieval_score=0.7,
            ),
        ],
        final_prompt="System: ...\nContext: ...\nQuery: what is RRF?",
        token_budget=TokenBudget(
            total_limit=4096,
            chunks_allocated=2000,
            history_allocated=500,
            system_allocated=800,
            headroom=796,
        ),
        history_pre=[
            Turn(role="user", content="hello", tokens=3),
            Turn(role="assistant", content="hi there", tokens=5),
        ],
        history_post=[
            Turn(role="user", content="hello", tokens=3),
        ],
        eviction_reason="token_budget",
        cache_events=[
            CacheEvent(chunk_id="c1", hit=True, cache_source="disk"),
            CacheEvent(chunk_id="c2", hit=False),
        ],
        tool_calls=[
            ToolCallRecord(
                tool_name="search",
                arguments={"query": "RRF"},
                result="3 hits",
                latency_ms=120.5,
            ),
        ],
        model="gpt-4",
        token_usage=TokenUsage(input_tokens=300, output_tokens=50, total_tokens=350),
    )


class TestMinimalRunRecord:
    def test_serialises(self):
        rec = RunRecord(query="q", response="r")
        data = rec.to_json()
        assert data["query"] == "q"
        assert data["response"] == "r"
        assert data["chunks"] is None

    def test_deserialises(self):
        data = {"query": "q", "response": "r"}
        rec = RunRecord.from_json(data)
        assert rec.query == "q"
        assert rec.response == "r"
        assert rec.chunks is None

    def test_round_trip(self):
        original = RunRecord(query="q", response="r")
        restored = RunRecord.from_json(original.to_json())
        assert original.to_json() == restored.to_json()

    def test_all_optionals_none_round_trip(self):
        original = RunRecord(query="q", response="r")
        restored = RunRecord.from_json(original.to_json())
        for field_name, value in restored.to_json().items():
            if field_name not in ("query", "response"):
                assert value is None, f"{field_name} should round-trip as None"


class TestChildDataclassRoundTrips:
    """Round-trip every child dataclass through asdict + re-construction,
    including the all-None-optionals shape."""

    def test_chunk_record(self):
        full = ChunkRecord(
            chunk_id="c1",
            source_doc_id="d1",
            content="x",
            token_count=5,
            retrieval_score=0.5,
            rerank_score=0.6,
            retrieval_path="bm25",
            truncated=True,
            cache_hit=False,
        )
        minimal = ChunkRecord(
            chunk_id="c2",
            source_doc_id="d2",
            content="y",
            token_count=3,
        )
        for original in (full, minimal):
            assert asdict(ChunkRecord(**asdict(original))) == asdict(original)
        assert minimal.retrieval_score is None
        assert minimal.cache_hit is None

    def test_token_budget(self):
        original = TokenBudget(
            total_limit=100,
            chunks_allocated=50,
            history_allocated=20,
            system_allocated=10,
            headroom=20,
        )
        assert asdict(TokenBudget(**asdict(original))) == asdict(original)

    def test_token_usage(self):
        original = TokenUsage(input_tokens=1, output_tokens=2, total_tokens=3)
        assert asdict(TokenUsage(**asdict(original))) == asdict(original)

    def test_turn(self):
        for original in (
            Turn(role="user", content="hi", tokens=2),
            Turn(role="user", content="hi"),
        ):
            assert asdict(Turn(**asdict(original))) == asdict(original)
        assert Turn(role="user", content="hi").tokens is None

    def test_cache_event(self):
        for original in (
            CacheEvent(chunk_id="c", hit=True, cache_source="disk"),
            CacheEvent(chunk_id="c", hit=False),
        ):
            assert asdict(CacheEvent(**asdict(original))) == asdict(original)
        assert CacheEvent(chunk_id="c", hit=False).cache_source is None

    def test_tool_call_record(self):
        full = ToolCallRecord(
            tool_name="t",
            arguments={"a": 1},
            result="ok",
            error=None,
            latency_ms=1.5,
        )
        minimal = ToolCallRecord(tool_name="t", arguments={})
        for original in (full, minimal):
            assert asdict(ToolCallRecord(**asdict(original))) == asdict(original)
        assert minimal.result is None
        assert minimal.error is None
        assert minimal.latency_ms is None


class TestFullRunRecord:
    def test_serialises(self):
        rec = _full_record()
        data = rec.to_json()
        assert data["query"] == "what is RRF?"
        assert len(data["chunks"]) == 2
        assert data["chunks"][0]["chunk_id"] == "c1"
        assert data["token_budget"]["total_limit"] == 4096
        assert data["token_usage"]["total_tokens"] == 350
        assert len(data["history_pre"]) == 2
        assert len(data["history_post"]) == 1
        assert len(data["cache_events"]) == 2
        assert len(data["tool_calls"]) == 1
        assert data["tool_calls"][0]["tool_name"] == "search"
        assert data["model"] == "gpt-4"
        assert data["eviction_reason"] == "token_budget"

    def test_deserialises(self):
        data = _full_record().to_json()
        rec = RunRecord.from_json(data)
        assert isinstance(rec.chunks[0], ChunkRecord)
        assert rec.chunks[0].retrieval_score == 0.9
        assert isinstance(rec.token_budget, TokenBudget)
        assert rec.token_budget.headroom == 796
        assert isinstance(rec.history_pre[0], Turn)
        assert rec.history_pre[0].tokens == 3
        assert isinstance(rec.cache_events[0], CacheEvent)
        assert rec.cache_events[0].cache_source == "disk"
        assert isinstance(rec.tool_calls[0], ToolCallRecord)
        assert rec.tool_calls[0].arguments == {"query": "RRF"}
        assert isinstance(rec.token_usage, TokenUsage)

    def test_round_trip(self):
        original = _full_record()
        restored = RunRecord.from_json(original.to_json())
        assert original.to_json() == restored.to_json()


class TestFlexibleInit:
    def test_unknown_kwargs_ignored(self):
        rec = RunRecord(query="q", response="r", unknown_field="x", another=42)
        assert rec.query == "q"
        assert rec.response == "r"
        assert not hasattr(rec, "unknown_field")

    def test_chunk_unknown_kwargs_ignored(self):
        c = ChunkRecord(
            chunk_id="c1",
            source_doc_id="d1",
            content="text",
            token_count=10,
            future_field="ignored",
        )
        assert c.chunk_id == "c1"
        assert c.token_count == 10
        assert not hasattr(c, "future_field")

    def test_toolcall_unknown_kwargs_ignored(self):
        t = ToolCallRecord(
            tool_name="search",
            arguments={"query": "RRF"},
            future_field="ignored",
        )
        assert t.tool_name == "search"
        assert t.arguments == {"query": "RRF"}
        assert not hasattr(t, "future_field")

    def test_every_dataclass_ignores_unknown_kwargs(self):
        assert (
            TokenBudget(
                total_limit=1,
                chunks_allocated=1,
                history_allocated=1,
                system_allocated=1,
                headroom=1,
                future=True,
            ).total_limit
            == 1
        )
        assert (
            TokenUsage(
                input_tokens=1,
                output_tokens=1,
                total_tokens=2,
                future=True,
            ).total_tokens
            == 2
        )
        assert Turn(role="user", content="hi", future=True).content == "hi"
        assert CacheEvent(chunk_id="c", hit=True, future=True).hit is True

    def test_from_json_ignores_unknown_fields(self):
        data = {"query": "q", "response": "r", "new_field": True}
        rec = RunRecord.from_json(data)
        assert rec.query == "q"


class TestToolCallsRoundTrip:
    def test_round_trip_preserves_tool_calls(self):
        original = RunRecord(
            query="q",
            response="r",
            tool_calls=[
                ToolCallRecord(
                    tool_name="search",
                    arguments={"query": "RRF", "top_k": 5},
                    result="3 hits",
                    latency_ms=120.5,
                ),
                ToolCallRecord(
                    tool_name="fetch_url",
                    arguments={"url": "https://example.com"},
                    error="timeout",
                ),
            ],
        )

        restored = RunRecord.from_json(original.to_json())

        assert original.to_json() == restored.to_json()
        assert len(restored.tool_calls) == 2

        first, second = restored.tool_calls
        assert isinstance(first, ToolCallRecord)
        assert first.tool_name == "search"
        assert first.arguments == {"query": "RRF", "top_k": 5}
        assert first.result == "3 hits"
        assert first.latency_ms == 120.5
        assert first.error is None

        assert isinstance(second, ToolCallRecord)
        assert second.tool_name == "fetch_url"
        assert second.error == "timeout"
        assert second.result is None

    def test_round_trip_none_tool_calls(self):
        original = RunRecord(query="q", response="r")
        restored = RunRecord.from_json(original.to_json())
        assert restored.tool_calls is None
