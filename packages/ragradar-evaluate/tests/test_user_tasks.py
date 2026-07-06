"""Task-level tests: the six user stories, end-to-end, using ONLY the
public API (ragradar_capture.capture / ragradar_evaluate.check / evaluate /
available_metrics). No raw SQL, no internal modules."""

import ragradar_capture
from ragradar_evaluate import check, evaluate
from ragradar_evaluate.facade import INPUT_METRICS


def _capture_run(pipeline: str = "tasks_pipe", good: bool = True, idx: int = 0) -> str:
    """Capture a run through the public one-liner; good/bad tweaks scores."""
    base = 0.9 if good else 0.35
    chunks = [
        {
            "chunk_id": f"c{idx}_{j}",
            "source_doc_id": f"doc_{j % (2 if good else 5)}",
            "content": f"Chunk content {idx}-{j} about retrieval quality.",
            "token_count": 100,
            "retrieval_score": base - j * 0.03,
            "rerank_score": base - j * 0.02,
            "retrieval_path": "hybrid",
            "truncated": (not good) and j == 0,
        }
        for j in range(4)
    ]
    run_id = ragradar_capture.capture(
        f"user question {idx}: how does reranking work?",
        f"answer {idx}: rerankers order chunks by relevance.",
        pipeline=pipeline,
        chunks=chunks,
        token_budget={
            "total_limit": 4096,
            "chunks_allocated": 2000,
            "history_allocated": 500,
            "system_allocated": 800,
            "headroom": 796,
        },
    )
    assert run_id is not None
    return run_id


class TestStory1CaptureThenCheck:
    def test_capture_then_check(self):
        run_id = _capture_run(good=True)
        result = check(run_id)

        assert result.verdict in ("ok", "warn", "fail")
        assert result.run_id == run_id
        assert result.risk_score is not None
        assert result.factors  # per-factor detail present

    def test_bad_run_reports_problems(self):
        run_id = _capture_run(good=False)
        result = check(run_id)

        assert result.verdict in ("warn", "fail")
        assert result.problems
        assert all(isinstance(p, str) for p in result.problems)


class TestStory2CaptureThenEvaluate:
    def test_capture_then_evaluate_free_metrics(self):
        run_id = _capture_run()
        result = evaluate(run_id, metrics=list(INPUT_METRICS))

        assert result.saved is True
        assert result.run_id == run_id
        assert result.risk_score is not None
        for name in INPUT_METRICS:
            if name == "cache_risk":
                continue  # this run never checked a semantic cache -- not applicable.
            if name == "filter_risk":
                continue  # this run never applied a metadata filter -- not applicable.
            if name == "score_underfill":
                continue  # this run never captured a requested_chunk_count -- not applicable.
            assert name in result.metrics


class TestStory3CheckFreshPipelineUsesPolicy:
    def test_policy_fallback_on_fresh_pipeline(self):
        run_id = _capture_run(pipeline="brand_new_pipe")
        result = check(run_id)

        assert result.thresholds == "policy"


class TestStory4CheckLearnsThresholds:
    def test_learned_thresholds_after_ten_evaluated_runs(self, fake_ragas):
        pipeline = "learning_pipe"
        run_ids = []
        for i in range(12):
            run_ids.append(_capture_run(pipeline=pipeline, good=(i % 2 == 0), idx=i))

        # Evaluate all of them fully (fake judge, varied scores so the
        # correlation model has signal to learn from).
        for i, run_id in enumerate(run_ids):
            fake_ragas.scores["faithfulness"] = round(0.4 + (i % 6) * 0.1, 2)
            fake_ragas.scores["answer_relevancy"] = round(0.5 + (i % 5) * 0.08, 2)
            result = evaluate(run_id)
            assert result.saved is True

        result = check(run_ids[-1])
        assert result.thresholds == "learned"
        assert result.verdict in ("ok", "warn", "fail")


class TestStory5EvaluateOneCheapMetric:
    def test_single_free_metric(self):
        run_id = _capture_run()
        result = evaluate(run_id, metrics=["duplicates"], save=False)

        assert set(result.metrics) == {"duplicates"}
        assert "duplicate_ratio" in result.metrics["duplicates"]
        assert result.skipped["relevance"] == "not requested"
        assert result.skipped["faithfulness"] == "not requested"
        assert result.saved is False


class TestStory6FullEvaluateWithMockedJudge:
    def test_complete_eval(self, fake_ragas):
        run_id = _capture_run()
        result = evaluate(run_id, ground_truth="rerankers reorder chunks")

        assert result.saved is True
        assert result.errors == {}
        for name in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
            assert isinstance(result.metrics[name], float)
        for name in INPUT_METRICS:
            if name == "cache_risk":
                continue  # this run never checked a semantic cache -- not applicable.
            if name == "filter_risk":
                continue  # this run never applied a metadata filter -- not applicable.
            if name == "score_underfill":
                continue  # this run never captured a requested_chunk_count -- not applicable.
            assert name in result.metrics
        assert result.risk_score is not None


class TestChunklessCheckDoesNotCrash:
    def test_verdict_explains_insufficient_data(self):
        run_id = ragradar_capture.capture("bare question", "bare answer")
        result = check(run_id)

        assert result.verdict in ("ok", "warn", "fail")
        assert result.risk_score is None
        assert result.factors == {}
        assert any("chunk" in p.lower() for p in result.problems)
