from ragradar_core.schema import ChunkRecord, RunRecord, TokenBudget
from ragradar_evaluate.layers.input_quality import cosine_similarity, score_input_quality
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


class TestCosineSimilarity:
    def test_identical_vectors(self):
        assert cosine_similarity([1, 0, 0], [1, 0, 0]) == 1.0

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1, 0, 0], [0, 1, 0]) == 0.0

    def test_zero_vector(self):
        assert cosine_similarity([0, 0, 0], [1, 0, 0]) == 0.0
