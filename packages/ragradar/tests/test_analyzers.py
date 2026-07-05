from datetime import datetime, timedelta, timezone

from ragradar.explain.analyzers import (
    cache,
    duplicates,
    history,
    scores,
    semantic_cache,
    tokens,
    truncation,
)
from ragradar_core.schema import CacheRecord, ChunkRecord, RunRecord, TokenBudget, Turn
from ragradar_evaluate.policy.schema import InputQualityPolicy


class TestEmptyRecord:
    """All analyzers return None on a minimal RunRecord."""

    def setup_method(self):
        self.empty = RunRecord(query="q", response="r")

    def test_tokens(self):
        assert tokens.analyze(self.empty) is None

    def test_duplicates(self):
        assert duplicates.analyze(self.empty) is None

    def test_truncation(self):
        assert truncation.analyze(self.empty) is None

    def test_history(self):
        assert history.analyze(self.empty) is None

    def test_cache(self):
        assert cache.analyze(self.empty) is None

    def test_semantic_cache(self):
        assert semantic_cache.analyze(self.empty) is None

    def test_scores(self):
        assert scores.analyze(self.empty) is None


class TestTokens:
    def test_structure(self, full_record):
        result = tokens.analyze(full_record)
        assert result is not None
        assert "total_tokens" in result
        assert "chunks_tokens" in result
        assert "history_tokens" in result
        assert "system_tokens" in result
        assert "headroom" in result
        assert "model_limit" in result
        assert "utilisation_pct" in result
        assert "per_chunk" in result

    def test_values(self, full_record):
        result = tokens.analyze(full_record)
        assert result["chunks_tokens"] == 80  # 50 + 30
        assert result["system_tokens"] == 800
        assert result["headroom"] == 796
        assert result["model_limit"] == 4096
        assert len(result["per_chunk"]) == 2

    def test_history_tokens_sums_post_only(self):
        record = RunRecord(
            query="q",
            response="r",
            chunks=[
                ChunkRecord(
                    chunk_id="c1",
                    source_doc_id="d1",
                    content="text",
                    token_count=10,
                ),
            ],
            token_budget=TokenBudget(
                total_limit=4096,
                chunks_allocated=2000,
                history_allocated=500,
                system_allocated=800,
                headroom=796,
            ),
            history_pre=[
                Turn(role="user", content="hello", tokens=10),
                Turn(role="assistant", content="hi", tokens=20),
                Turn(role="user", content="question", tokens=15),
            ],
            history_post=[
                Turn(role="user", content="hello", tokens=10),
                Turn(role="assistant", content="hi", tokens=20),
            ],
        )
        result = tokens.analyze(record)
        assert result["history_tokens"] == 30  # post only: 10+20


class TestDuplicates:
    def test_structure(self, full_record):
        result = duplicates.analyze(full_record)
        assert result is not None
        assert "path_dups" in result
        assert "window_dups" in result
        assert "semantic_dups" in result
        assert "duplicate_ratio" in result

    def test_path_dups_detected(self):
        record = RunRecord(
            query="q",
            response="r",
            chunks=[
                ChunkRecord(
                    chunk_id="c1",
                    source_doc_id="d1",
                    content="text A",
                    token_count=10,
                    retrieval_path="bm25",
                ),
                ChunkRecord(
                    chunk_id="c1",
                    source_doc_id="d1",
                    content="text A",
                    token_count=10,
                    retrieval_path="ann",
                ),
            ],
        )
        result = duplicates.analyze(record)
        assert len(result["path_dups"]) == 1
        assert result["path_dups"][0]["chunk_id"] == "c1"
        assert set(result["path_dups"][0]["paths"]) == {"bm25", "ann"}

    def test_window_dups_detected(self):
        record = RunRecord(
            query="q",
            response="r",
            chunks=[
                ChunkRecord(
                    chunk_id="c1",
                    source_doc_id="d1",
                    content="the quick brown fox jumps",
                    token_count=10,
                ),
                ChunkRecord(
                    chunk_id="c2",
                    source_doc_id="d1",
                    content="the quick brown fox jumps over the lazy dog",
                    token_count=15,
                ),
            ],
        )
        result = duplicates.analyze(record)
        assert len(result["window_dups"]) == 1
        assert "c1" in result["window_dups"][0]["chunk_ids"]
        assert "c2" in result["window_dups"][0]["chunk_ids"]

    def test_no_dups(self):
        record = RunRecord(
            query="q",
            response="r",
            chunks=[
                ChunkRecord(
                    chunk_id="c1",
                    source_doc_id="d1",
                    content="text A",
                    token_count=10,
                ),
                ChunkRecord(
                    chunk_id="c2",
                    source_doc_id="d2",
                    content="text B",
                    token_count=10,
                ),
            ],
        )
        result = duplicates.analyze(record)
        assert result["duplicate_ratio"] == 0.0


class TestTruncation:
    def test_structure(self, full_record):
        result = truncation.analyze(full_record)
        assert result is not None
        assert "truncated_count" in result
        assert "truncated_chunks" in result
        assert "high_score_truncations" in result
        assert "severity" in result

    def test_high_severity_when_high_score_truncated(self):
        record = RunRecord(
            query="q",
            response="r",
            chunks=[
                ChunkRecord(
                    chunk_id="c1",
                    source_doc_id="d1",
                    content="text",
                    token_count=10,
                    retrieval_score=0.9,
                    truncated=True,
                ),
            ],
        )
        result = truncation.analyze(record)
        assert result["severity"] == "high"
        assert result["high_score_truncations"] == 1

    def test_low_severity(self):
        record = RunRecord(
            query="q",
            response="r",
            chunks=[
                ChunkRecord(
                    chunk_id="c1",
                    source_doc_id="d1",
                    content="text",
                    token_count=10,
                    retrieval_score=0.3,
                    truncated=True,
                ),
            ],
        )
        result = truncation.analyze(record)
        assert result["severity"] == "low"

    def test_none_severity(self):
        record = RunRecord(
            query="q",
            response="r",
            chunks=[
                ChunkRecord(
                    chunk_id="c1",
                    source_doc_id="d1",
                    content="text",
                    token_count=10,
                    truncated=False,
                ),
            ],
        )
        result = truncation.analyze(record)
        assert result["severity"] == "none"


class TestHistory:
    def test_structure(self, full_record):
        result = history.analyze(full_record)
        assert result is not None
        assert result["pre_turn_count"] == 2
        assert result["post_turn_count"] == 1
        assert result["dropped_turn_count"] == 1
        assert result["eviction_reason"] == "token_budget"

    def test_dropped_turns_identified(self, full_record):
        result = history.analyze(full_record)
        assert len(result["dropped_turns"]) == 1
        assert result["dropped_turns"][0].role == "assistant"

    def test_token_sums(self, full_record):
        result = history.analyze(full_record)
        assert result["pre_tokens"] == 8  # 3 + 5
        assert result["post_tokens"] == 3

    def test_history_all_turns_evicted(self):
        record = RunRecord(
            query="q",
            response="r",
            history_pre=[
                Turn(role="user", content="hello", tokens=10),
                Turn(role="assistant", content="hi", tokens=20),
                Turn(role="user", content="question", tokens=15),
            ],
            history_post=[],
        )
        result = history.analyze(record)
        assert result is not None
        assert result["dropped_turn_count"] == 3
        assert result["post_turn_count"] == 0


class TestCache:
    def test_structure(self, full_record):
        result = cache.analyze(full_record)
        assert result is not None
        assert result["total_events"] == 2
        assert result["hits"] == 1
        assert result["misses"] == 1
        assert result["hit_ratio"] == 0.5
        assert result["hit_chunks"] == ["c1"]
        assert result["miss_chunks"] == ["c2"]


class TestScores:
    def test_structure(self, full_record):
        result = scores.analyze(full_record)
        assert result is not None
        assert result["top_retrieval"] == 0.9
        assert result["bottom_retrieval"] == 0.7
        assert result["top_rerank"] == 0.85
        assert result["bottom_rerank"] == 0.4

    def test_rerank_delta(self, full_record):
        result = scores.analyze(full_record)
        mean_rerank = (0.85 + 0.4) / 2
        mean_retrieval = (0.9 + 0.7) / 2
        expected = round(mean_rerank - mean_retrieval, 4)
        assert result["rerank_delta"] == expected

    def test_low_score_ratio(self, full_record):
        result = scores.analyze(full_record)
        # c2 has rerank 0.4 < 0.5 â†’ 1/2 = 0.5
        assert result["low_score_ratio"] == 0.5

    def test_no_scores_returns_none(self):
        record = RunRecord(
            query="q",
            response="r",
            chunks=[
                ChunkRecord(
                    chunk_id="c1",
                    source_doc_id="d1",
                    content="text",
                    token_count=10,
                ),
            ],
        )
        result = scores.analyze(record)
        assert result is None


class TestSemanticCache:
    def test_not_checked_still_renders_checked_false(self):
        record = RunRecord(query="q", response="r", cache=CacheRecord(checked=False))
        result = semantic_cache.analyze(record)
        assert result is not None
        assert result["checked"] is False

    def test_checked_miss(self):
        record = RunRecord(query="q", response="r", cache=CacheRecord(checked=True, hit=False))
        result = semantic_cache.analyze(record)
        assert result["checked"] is True
        assert result["hit"] is False
        assert result["borderline_hit"] is False
        assert result["stale_hit"] is False

    def test_checked_hit_clean(self):
        record = RunRecord(
            query="q",
            response="r",
            cache=CacheRecord(
                checked=True,
                hit=True,
                similarity_score=0.98,
                threshold=0.9,
                cached_at=datetime.now(timezone.utc).isoformat(),
                registered=True,
            ),
        )
        result = semantic_cache.analyze(record, InputQualityPolicy())
        assert result["hit"] is True
        assert result["borderline_hit"] is False
        assert result["stale_hit"] is False

    def test_checked_hit_borderline(self):
        record = RunRecord(
            query="q",
            response="r",
            cache=CacheRecord(
                checked=True,
                hit=True,
                similarity_score=0.91,
                threshold=0.9,
                cached_at=datetime.now(timezone.utc).isoformat(),
            ),
        )
        policy = InputQualityPolicy(cache_borderline_margin=0.03)
        result = semantic_cache.analyze(record, policy)
        assert result["borderline_hit"] is True
        assert result["stale_hit"] is False

    def test_checked_hit_stale(self):
        old_time = datetime.now(timezone.utc) - timedelta(days=2)
        record = RunRecord(
            query="q",
            response="r",
            cache=CacheRecord(
                checked=True,
                hit=True,
                similarity_score=0.98,
                threshold=0.9,
                cached_at=old_time.isoformat(),
            ),
        )
        policy = InputQualityPolicy(cache_max_age_seconds=3600)
        result = semantic_cache.analyze(record, policy)
        assert result["borderline_hit"] is False
        assert result["stale_hit"] is True

    def test_default_policy_used_when_none_given(self):
        record = RunRecord(
            query="q",
            response="r",
            cache=CacheRecord(checked=True, hit=True, similarity_score=0.98, threshold=0.9),
        )
        result = semantic_cache.analyze(record)
        assert result is not None
        assert result["borderline_hit"] is False
