from datetime import datetime, timedelta, timezone

from ragradar_core.schema import CacheRecord, ChunkRecord, FilterRecord, RunRecord, TokenBudget
from ragradar_evaluate.layers.input_quality import (
    check_policy_violations,
    cosine_similarity,
    score_cache_risk,
    score_filter_risk,
    score_input_quality,
    score_score_degeneracy,
    score_score_margin,
)
from ragradar_evaluate.policy.schema import InputQualityPolicy


class TestEmptyRecord:
    def test_no_chunks(self):
        rec = RunRecord(query="q", response="r")
        assert score_input_quality(rec, InputQualityPolicy()) is None

    def test_empty_chunks(self):
        rec = RunRecord(query="q", response="r", chunks=[])
        assert score_input_quality(rec, InputQualityPolicy()) is None


class TestRelevance:
    def test_uses_rerank_score_when_no_embedding(self):
        rec = RunRecord(
            query="q",
            response="r",
            chunks=[
                ChunkRecord(
                    chunk_id="c1",
                    source_doc_id="d1",
                    content="text",
                    token_count=10,
                    rerank_score=0.9,
                )
            ],
        )
        result = score_input_quality(rec, InputQualityPolicy())
        assert 0.9 in result["relevance_scores"]

    def test_uses_retrieval_score_as_fallback(self):
        rec = RunRecord(
            query="q",
            response="r",
            chunks=[
                ChunkRecord(
                    chunk_id="c1",
                    source_doc_id="d1",
                    content="text",
                    token_count=10,
                    retrieval_score=0.7,
                )
            ],
        )
        result = score_input_quality(rec, InputQualityPolicy())
        assert 0.7 in result["relevance_scores"]

    def test_embedding_fn_used_when_provided(self):
        rec = RunRecord(
            query="q",
            response="r",
            chunks=[
                ChunkRecord(
                    chunk_id="c1",
                    source_doc_id="d1",
                    content="text",
                    token_count=10,
                )
            ],
        )
        fake_embed = lambda text: [1.0, 0.0, 0.0]
        result = score_input_quality(rec, InputQualityPolicy(), embedding_fn=fake_embed)
        assert result["relevance_scores"][0] == 1.0


class TestDuplicates:
    def test_path_duplicate_detection(self):
        rec = RunRecord(
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
        result = score_input_quality(rec, InputQualityPolicy())
        assert result["path_dup_count"] == 1
        assert result["duplicate_ratio"] > 0

    def test_window_duplicate_detection(self):
        rec = RunRecord(
            query="q",
            response="r",
            chunks=[
                ChunkRecord(
                    chunk_id="c1", source_doc_id="d1", content="the quick brown fox", token_count=10
                ),
                ChunkRecord(
                    chunk_id="c2",
                    source_doc_id="d1",
                    content="the quick brown fox jumps over",
                    token_count=15,
                ),
            ],
        )
        result = score_input_quality(rec, InputQualityPolicy())
        assert result["window_dup_count"] == 1


class TestTruncation:
    def test_high_score_truncation(self):
        rec = RunRecord(
            query="q",
            response="r",
            chunks=[
                ChunkRecord(
                    chunk_id="c1",
                    source_doc_id="d1",
                    content="text",
                    token_count=10,
                    rerank_score=0.85,
                    truncated=True,
                )
            ],
        )
        result = score_input_quality(rec, InputQualityPolicy())
        assert result["high_score_truncations"] == 1
        assert result["truncation_severity"] == "high"

    def test_low_score_truncation(self):
        rec = RunRecord(
            query="q",
            response="r",
            chunks=[
                ChunkRecord(
                    chunk_id="c1",
                    source_doc_id="d1",
                    content="text",
                    token_count=10,
                    rerank_score=0.3,
                    truncated=True,
                )
            ],
        )
        result = score_input_quality(rec, InputQualityPolicy())
        assert result["truncation_severity"] == "low"


class TestPolicy:
    def test_violation_detected(self):
        rec = RunRecord(
            query="q",
            response="r",
            chunks=[
                ChunkRecord(
                    chunk_id="c1",
                    source_doc_id="d1",
                    content="a",
                    token_count=10,
                    retrieval_path="bm25",
                ),
                ChunkRecord(
                    chunk_id="c1",
                    source_doc_id="d1",
                    content="a",
                    token_count=10,
                    retrieval_path="ann",
                ),
            ],
        )
        policy = InputQualityPolicy(max_duplicate_ratio=0.0)
        result = score_input_quality(rec, policy)
        assert "max_duplicate_ratio" in result["policy_violations"]
        assert not result["passes_policy"]

    def test_passes_on_clean_record(self):
        rec = RunRecord(
            query="q",
            response="r",
            chunks=[
                ChunkRecord(
                    chunk_id="c1",
                    source_doc_id="d1",
                    content="good chunk",
                    token_count=10,
                    rerank_score=0.9,
                ),
                ChunkRecord(
                    chunk_id="c2",
                    source_doc_id="d2",
                    content="another good",
                    token_count=10,
                    rerank_score=0.85,
                ),
            ],
            token_budget=TokenBudget(
                total_limit=4096,
                chunks_allocated=2000,
                history_allocated=500,
                system_allocated=800,
                headroom=796,
            ),
        )
        result = score_input_quality(rec, InputQualityPolicy())
        assert result["passes_policy"]

    def test_violations_checked_on_raw_values_not_rounded(self):
        # One window-dup pair over 3 chunks: raw duplicate_ratio is
        # 1/3 = 0.33333..., which rounds to exactly the 0.3333 threshold.
        # The raw value is above the threshold and must still violate —
        # the monolith compared raw values before rounding.
        rec = RunRecord(
            query="q",
            response="r",
            chunks=[
                ChunkRecord(
                    chunk_id="a", source_doc_id="d1", content="alpha beta gamma", token_count=10
                ),
                ChunkRecord(
                    chunk_id="b",
                    source_doc_id="d1",
                    content="alpha beta gamma delta",
                    token_count=12,
                ),
                ChunkRecord(chunk_id="c", source_doc_id="d2", content="unrelated", token_count=5),
            ],
        )
        policy = InputQualityPolicy(max_duplicate_ratio=0.3333)
        result = score_input_quality(rec, policy)
        assert result["duplicate_ratio"] == 0.3333  # persisted value stays rounded
        assert "max_duplicate_ratio" in result["policy_violations"]

    def test_mean_relevance_checked_on_raw_value(self):
        # Raw mean_relevance is 2/3 = 0.66666... < the 0.6667 minimum,
        # but rounds to exactly 0.6667; the raw value must still violate.
        rec = RunRecord(
            query="q",
            response="r",
            chunks=[
                ChunkRecord(
                    chunk_id=f"c{i}",
                    source_doc_id=f"d{i}",
                    content=f"chunk {i}",
                    token_count=10,
                    rerank_score=s,
                )
                for i, s in enumerate([1.0, 0.5, 0.5])
            ],
        )
        policy = InputQualityPolicy(min_chunk_relevance_score=0.6667)
        result = score_input_quality(rec, policy)
        assert result["mean_relevance"] == 0.6667
        assert "min_chunk_relevance_score" in result["policy_violations"]


class TestCacheRisk:
    def _record(self, cache: CacheRecord | None) -> RunRecord:
        return RunRecord(query="q", response="r", cache=cache)

    def test_no_cache_data_returns_none(self):
        rec = self._record(None)
        assert score_cache_risk(rec, InputQualityPolicy()) is None

    def test_not_checked_returns_none(self):
        rec = self._record(CacheRecord(checked=False))
        assert score_cache_risk(rec, InputQualityPolicy()) is None

    def test_checked_miss(self):
        rec = self._record(CacheRecord(checked=True, hit=False))
        result = score_cache_risk(rec, InputQualityPolicy())
        assert result["cache_hit"] is False
        assert result["borderline_hit"] is False
        assert result["stale_hit"] is False

    def test_checked_hit_clean(self):
        rec = self._record(
            CacheRecord(
                checked=True,
                hit=True,
                similarity_score=0.98,
                threshold=0.9,
                cached_at=datetime.now(timezone.utc).isoformat(),
            )
        )
        result = score_cache_risk(rec, InputQualityPolicy())
        assert result["cache_hit"] is True
        assert result["borderline_hit"] is False
        assert result["stale_hit"] is False

    def test_checked_hit_borderline(self):
        policy = InputQualityPolicy(cache_borderline_margin=0.03)
        rec = self._record(
            CacheRecord(
                checked=True,
                hit=True,
                similarity_score=0.91,
                threshold=0.9,
                cached_at=datetime.now(timezone.utc).isoformat(),
            )
        )
        result = score_cache_risk(rec, policy)
        assert result["borderline_hit"] is True
        assert result["stale_hit"] is False

    def test_checked_hit_stale(self):
        policy = InputQualityPolicy(cache_max_age_seconds=3600)
        old_time = datetime.now(timezone.utc) - timedelta(hours=2)
        rec = self._record(
            CacheRecord(
                checked=True,
                hit=True,
                similarity_score=0.98,
                threshold=0.9,
                cached_at=old_time.isoformat(),
            )
        )
        result = score_cache_risk(rec, policy)
        assert result["borderline_hit"] is False
        assert result["stale_hit"] is True
        assert result["cache_age_seconds"] > 3600

    def test_registered_and_query_passed_through(self):
        rec = self._record(
            CacheRecord(checked=True, hit=True, cached_query="near dup", registered=True)
        )
        result = score_cache_risk(rec, InputQualityPolicy())
        assert result["cache_registered"] is True


class TestFilterRisk:
    def _record(self, filt: FilterRecord | None) -> RunRecord:
        return RunRecord(query="q", response="r", filter=filt)

    def test_no_filter_data_returns_none(self):
        rec = self._record(None)
        assert score_filter_risk(rec) is None

    def test_not_applied_returns_none(self):
        rec = self._record(FilterRecord(applied=False))
        assert score_filter_risk(rec) is None

    def test_applied_missing_candidate_count_returns_none(self):
        rec = self._record(FilterRecord(applied=True, excluded_count=3))
        assert score_filter_risk(rec) is None

    def test_applied_missing_excluded_count_returns_none(self):
        rec = self._record(FilterRecord(applied=True, candidate_count=10))
        assert score_filter_risk(rec) is None

    def test_applied_zero_candidate_count_returns_none(self):
        rec = self._record(FilterRecord(applied=True, candidate_count=0, excluded_count=0))
        assert score_filter_risk(rec) is None

    def test_applied_with_counts_computes_ratio(self):
        rec = self._record(FilterRecord(applied=True, candidate_count=10, excluded_count=4))
        result = score_filter_risk(rec)
        assert result["filtered_exclusion_ratio"] == 0.4
        assert result["filter_excluded_count"] == 4
        assert result["filter_candidate_count"] == 10

    def test_policy_violation_fires_on_high_exclusion_ratio(self):
        rec = self._record(FilterRecord(applied=True, candidate_count=10, excluded_count=4))
        policy = InputQualityPolicy(max_filtered_exclusion_ratio=0.3)
        values = score_filter_risk(rec)
        violations = check_policy_violations(values, policy, rec)
        assert "max_filtered_exclusion_ratio" in violations

    def test_policy_passes_on_low_exclusion_ratio(self):
        rec = self._record(FilterRecord(applied=True, candidate_count=10, excluded_count=1))
        policy = InputQualityPolicy(max_filtered_exclusion_ratio=0.3)
        values = score_filter_risk(rec)
        violations = check_policy_violations(values, policy, rec)
        assert "max_filtered_exclusion_ratio" not in violations


class TestScoreDegeneracy:
    def _record(self, chunks) -> RunRecord:
        return RunRecord(query="q", response="r", chunks=chunks)

    def test_variance_uses_rerank_score(self):
        rec = self._record(
            [
                ChunkRecord(
                    chunk_id="c1", source_doc_id="d1", content="a", token_count=10, rerank_score=0.9
                ),
                ChunkRecord(
                    chunk_id="c2", source_doc_id="d2", content="b", token_count=10, rerank_score=0.4
                ),
            ]
        )
        result = score_score_degeneracy(rec)
        assert result["chunk_score_variance"] == 0.0625

    def test_falls_back_to_retrieval_score(self):
        rec = self._record(
            [
                ChunkRecord(
                    chunk_id="c1", source_doc_id="d1", content="a", token_count=10,
                    retrieval_score=0.9,
                ),
                ChunkRecord(
                    chunk_id="c2", source_doc_id="d2", content="b", token_count=10,
                    retrieval_score=0.4,
                ),
            ]
        )
        result = score_score_degeneracy(rec)
        assert result["chunk_score_variance"] == 0.0625

    def test_excludes_chunks_with_no_usable_score(self):
        rec = self._record(
            [
                ChunkRecord(
                    chunk_id="c1", source_doc_id="d1", content="a", token_count=10, rerank_score=0.9
                ),
                ChunkRecord(
                    chunk_id="c2", source_doc_id="d2", content="b", token_count=10, rerank_score=0.4
                ),
                ChunkRecord(chunk_id="c3", source_doc_id="d3", content="c", token_count=10),
            ]
        )
        result = score_score_degeneracy(rec)
        assert result["chunk_score_variance"] == 0.0625

    def test_none_with_fewer_than_two_usable_scores(self):
        rec = self._record(
            [
                ChunkRecord(
                    chunk_id="c1", source_doc_id="d1", content="a", token_count=10, rerank_score=0.9
                ),
                ChunkRecord(chunk_id="c2", source_doc_id="d2", content="b", token_count=10),
            ]
        )
        result = score_score_degeneracy(rec)
        assert result["chunk_score_variance"] is None

    def test_none_with_zero_usable_scores(self):
        rec = self._record(
            [ChunkRecord(chunk_id="c1", source_doc_id="d1", content="a", token_count=10)]
        )
        result = score_score_degeneracy(rec)
        assert result["chunk_score_variance"] is None

    def test_does_not_collide_with_coherence_score_variance(self):
        # score_coherence's "score_variance" (rerank-only, needs >1 rerank
        # score) and this factor's "chunk_score_variance" (rerank falling
        # back to retrieval_score) are merged into the same flat dict by
        # score_input_quality() -- they must coexist under distinct keys
        # rather than one clobbering the other.
        rec = self._record(
            [
                ChunkRecord(
                    chunk_id="c1", source_doc_id="d1", content="a", token_count=10, rerank_score=0.9
                ),
                ChunkRecord(
                    chunk_id="c2", source_doc_id="d2", content="b", token_count=10,
                    retrieval_score=0.2,
                ),
            ]
        )
        result = score_input_quality(rec, InputQualityPolicy())
        # coherence sees only one rerank score -> undefined (needs >1)
        assert result["score_variance"] is None
        # degeneracy falls back to retrieval_score for c2 -> two usable scores
        assert result["chunk_score_variance"] == 0.1225

    def test_policy_violation_fires_on_low_variance(self):
        rec = self._record(
            [
                ChunkRecord(
                    chunk_id="c1", source_doc_id="d1", content="a", token_count=10, rerank_score=0.70
                ),
                ChunkRecord(
                    chunk_id="c2", source_doc_id="d2", content="b", token_count=10, rerank_score=0.71
                ),
            ]
        )
        policy = InputQualityPolicy(min_score_variance=0.01)
        values = score_score_degeneracy(rec)
        violations = check_policy_violations(values, policy, rec)
        assert "min_score_variance" in violations

    def test_policy_passes_on_healthy_variance(self):
        rec = self._record(
            [
                ChunkRecord(
                    chunk_id="c1", source_doc_id="d1", content="a", token_count=10, rerank_score=0.9
                ),
                ChunkRecord(
                    chunk_id="c2", source_doc_id="d2", content="b", token_count=10, rerank_score=0.4
                ),
            ]
        )
        policy = InputQualityPolicy(min_score_variance=0.01)
        values = score_score_degeneracy(rec)
        violations = check_policy_violations(values, policy, rec)
        assert "min_score_variance" not in violations

    def test_none_value_never_violates(self):
        rec = self._record(
            [ChunkRecord(chunk_id="c1", source_doc_id="d1", content="a", token_count=10)]
        )
        values = score_score_degeneracy(rec)
        violations = check_policy_violations(values, InputQualityPolicy(), rec)
        assert "min_score_variance" not in violations


class TestScoreMargin:
    def _record(self, chunks) -> RunRecord:
        return RunRecord(query="q", response="r", chunks=chunks)

    def test_normal_case(self):
        rec = self._record(
            [
                ChunkRecord(
                    chunk_id="c1", source_doc_id="d1", content="a", token_count=10, rerank_score=0.9
                ),
                ChunkRecord(
                    chunk_id="c2", source_doc_id="d2", content="b", token_count=10, rerank_score=0.4
                ),
            ]
        )
        result = score_score_margin(rec, InputQualityPolicy())
        assert result["top_second_margin"] == 0.5
        assert result["threshold_margin"] == 0.9 - InputQualityPolicy().min_top_chunk_score

    def test_falls_back_to_retrieval_score(self):
        rec = self._record(
            [
                ChunkRecord(
                    chunk_id="c1", source_doc_id="d1", content="a", token_count=10,
                    retrieval_score=0.9,
                ),
                ChunkRecord(
                    chunk_id="c2", source_doc_id="d2", content="b", token_count=10,
                    retrieval_score=0.4,
                ),
            ]
        )
        result = score_score_margin(rec, InputQualityPolicy())
        assert result["top_second_margin"] == 0.5

    def test_sorts_scores_regardless_of_chunk_order(self):
        rec = self._record(
            [
                ChunkRecord(
                    chunk_id="c1", source_doc_id="d1", content="a", token_count=10, rerank_score=0.4
                ),
                ChunkRecord(
                    chunk_id="c2", source_doc_id="d2", content="b", token_count=10, rerank_score=0.9
                ),
                ChunkRecord(
                    chunk_id="c3", source_doc_id="d3", content="c", token_count=10, rerank_score=0.6
                ),
            ]
        )
        result = score_score_margin(rec, InputQualityPolicy())
        assert round(result["top_second_margin"], 4) == 0.3

    def test_none_with_fewer_than_two_usable_scores(self):
        rec = self._record(
            [
                ChunkRecord(
                    chunk_id="c1", source_doc_id="d1", content="a", token_count=10, rerank_score=0.9
                ),
                ChunkRecord(chunk_id="c2", source_doc_id="d2", content="b", token_count=10),
            ]
        )
        assert score_score_margin(rec, InputQualityPolicy()) is None

    def test_none_with_zero_usable_scores(self):
        rec = self._record(
            [ChunkRecord(chunk_id="c1", source_doc_id="d1", content="a", token_count=10)]
        )
        assert score_score_margin(rec, InputQualityPolicy()) is None

    def test_threshold_margin_present_but_not_policy_checked(self):
        # threshold_margin rides on the same min_top_chunk_score boundary the
        # top_chunk_score factor already owns -- diagnostic-only, no
        # _CHECK_FACTORS entry or policy field of its own. A record whose
        # top score is well below min_top_chunk_score (so threshold_margin is
        # sharply negative) must not produce any violation naming it --
        # only top_second_margin can fire from this function's output.
        rec = self._record(
            [
                ChunkRecord(
                    chunk_id="c1", source_doc_id="d1", content="a", token_count=10, rerank_score=0.3
                ),
                ChunkRecord(
                    chunk_id="c2", source_doc_id="d2", content="b", token_count=10, rerank_score=0.25
                ),
            ]
        )
        policy = InputQualityPolicy(min_top_chunk_score=0.7, min_top_second_margin=0.01)
        values = score_score_margin(rec, policy)
        assert values["threshold_margin"] == 0.3 - 0.7  # present in the dict...
        violations = check_policy_violations(values, policy, rec)
        assert violations == []  # ...but produces no violation on its own
        assert not any("threshold_margin" in v for v in violations)

    def test_policy_violation_fires_on_thin_margin(self):
        rec = self._record(
            [
                ChunkRecord(
                    chunk_id="c1", source_doc_id="d1", content="a", token_count=10, rerank_score=0.9
                ),
                ChunkRecord(
                    chunk_id="c2", source_doc_id="d2", content="b", token_count=10, rerank_score=0.89
                ),
            ]
        )
        policy = InputQualityPolicy(min_top_second_margin=0.05)
        values = score_score_margin(rec, policy)
        violations = check_policy_violations(values, policy, rec)
        assert "min_top_second_margin" in violations

    def test_policy_passes_on_healthy_margin(self):
        rec = self._record(
            [
                ChunkRecord(
                    chunk_id="c1", source_doc_id="d1", content="a", token_count=10, rerank_score=0.9
                ),
                ChunkRecord(
                    chunk_id="c2", source_doc_id="d2", content="b", token_count=10, rerank_score=0.4
                ),
            ]
        )
        policy = InputQualityPolicy(min_top_second_margin=0.05)
        values = score_score_margin(rec, policy)
        violations = check_policy_violations(values, policy, rec)
        assert "min_top_second_margin" not in violations

    def test_not_included_in_score_input_quality(self):
        # score_score_margin needs policy mid-computation, so -- like
        # score_cache_risk/score_filter_risk -- it is dispatched only
        # through evaluate(), never by the policy-free score_input_quality().
        rec = self._record(
            [
                ChunkRecord(
                    chunk_id="c1", source_doc_id="d1", content="a", token_count=10, rerank_score=0.9
                ),
                ChunkRecord(
                    chunk_id="c2", source_doc_id="d2", content="b", token_count=10, rerank_score=0.4
                ),
            ]
        )
        result = score_input_quality(rec, InputQualityPolicy())
        assert "top_second_margin" not in result
        assert "threshold_margin" not in result


class TestCosineSimilarity:
    def test_identical_vectors(self):
        assert cosine_similarity([1, 0, 0], [1, 0, 0]) == 1.0

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1, 0, 0], [0, 1, 0]) == 0.0

    def test_zero_vector(self):
        assert cosine_similarity([0, 0, 0], [1, 0, 0]) == 0.0
