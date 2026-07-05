"""Engine-level tests for the atomic-metric facade: available_metrics()
discovery, metric selection atomicity, the unified errors channel,
risk None-vs-0.0 semantics, and save/persistence parity with the CLI."""

import json
import sqlite3
from unittest import mock

import pytest
from click.testing import CliRunner
from ragradar_core import store
from ragradar_core.schema import RunRecord
from ragradar_evaluate import EvalResult, available_metrics, check, evaluate
from ragradar_evaluate.cli import main
from ragradar_evaluate.facade import INPUT_METRICS, OUTPUT_METRICS
from ragradar_evaluate.layers import input_quality, output_quality
from ragradar_evaluate.policy.persistence import load_policy as _real_load_policy
from ragradar_evaluate.policy.persistence import save_policy
from ragradar_evaluate.policy.schema import InputQualityPolicy

ALL_METRICS = {
    "relevance",
    "duplicates",
    "truncation",
    "token_efficiency",
    "coherence",
    "cache_risk",
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
}


class TestAvailableMetrics:
    def test_all_ten_present(self):
        metrics = available_metrics()
        assert set(metrics) == ALL_METRICS

    def test_layers_and_costs(self):
        metrics = available_metrics()
        for name in INPUT_METRICS:
            assert metrics[name].layer == "input"
            assert metrics[name].cost == "free"
        for name in OUTPUT_METRICS:
            assert metrics[name].layer == "output"
            assert metrics[name].cost == "llm"

    def test_context_recall_requires_ground_truth(self):
        assert "ground_truth" in available_metrics()["context_recall"].requires

    def test_every_name_accepted_by_evaluate(self, full_record, fake_ragas):
        for name in available_metrics():
            result = evaluate(full_record, metrics=[name], ground_truth="gt", save=False)
            assert isinstance(result, EvalResult)
            assert name in result.metrics or name in result.skipped


class TestAtomicity:
    def test_single_input_metric_computes_nothing_else(self, full_record, monkeypatch):
        called = []

        def spy(name, real):
            def wrapper(record, *a, **k):
                called.append(name)
                return real(record, *a, **k)

            return wrapper

        for family in ["relevance", "duplicates", "truncation", "token_efficiency", "coherence"]:
            fn_name = f"score_{family}"
            monkeypatch.setattr(
                input_quality, fn_name, spy(family, getattr(input_quality, fn_name))
            )

        def no_output(*a, **k):
            raise AssertionError("output layer must not run for an input-only selection")

        monkeypatch.setattr(output_quality, "score_output_quality", no_output)

        result = evaluate(full_record, metrics=["duplicates"], save=False)

        assert called == ["duplicates"]
        assert "duplicates" in result.metrics
        assert result.metrics["duplicates"]["duplicate_ratio"] is not None
        assert result.skipped["relevance"] == "not requested"
        assert result.skipped["faithfulness"] == "not requested"

    def test_single_ragas_metric_passes_exactly_that_object(self, full_record, fake_ragas):
        result = evaluate(full_record, metrics=["faithfulness"], save=False)

        assert len(fake_ragas.calls) == 1
        passed = fake_ragas.calls[0]["metrics"]
        assert passed == [fake_ragas.metric_objects["faithfulness"]]
        assert result.metrics["faithfulness"] == 0.9
        assert result.risk_score is None  # no input metrics computed


class TestCompleteEval:
    def test_metrics_none_computes_everything_applicable(self, full_record, fake_ragas):
        result = evaluate(full_record, save=False)

        for name in INPUT_METRICS:
            if name == "cache_risk":
                continue  # full_record never checked a semantic cache -- not applicable.
            assert name in result.metrics, f"{name} missing from complete eval"
        for name in ("faithfulness", "answer_relevancy", "context_precision"):
            assert result.metrics[name] == fake_ragas.scores[name]

        # cache_risk is legitimately not-applicable (no cache data), not missing data.
        assert result.skipped["cache_risk"] == "not applicable: run never checked a semantic cache"
        # context_recall needs ground_truth -- skipped with the reason.
        assert result.skipped["context_recall"] == "requires ground_truth"
        assert result.risk_score is not None

    def test_ground_truth_unlocks_context_recall(self, full_record, fake_ragas):
        result = evaluate(full_record, ground_truth="the truth", save=False)
        assert result.metrics["context_recall"] == 0.75
        assert "context_recall" not in result.skipped


class TestChunklessRecord:
    def test_input_metrics_skipped_and_risk_none(self):
        rec = RunRecord(query="q", response="r")
        result = evaluate(rec, save=False)

        for name in INPUT_METRICS:
            if name == "cache_risk":
                # No chunks AND no cache data -- skipped for the cache reason,
                # not the chunks reason.
                assert "semantic cache" in result.skipped[name]
                continue
            assert "no chunks" in result.skipped[name]
        for name in OUTPUT_METRICS:
            assert "no chunks" in result.skipped[name]
        # Regression for the None-vs-0.0 ambiguity: nothing was computed,
        # so risk must be None, not "0.0 = no risk".
        assert result.risk_score is None
        assert result.metrics == {}


class TestUnifiedErrors:
    def test_ragas_not_installed_lands_in_errors(self, full_record, monkeypatch):
        import sys

        monkeypatch.setitem(sys.modules, "ragas", None)
        monkeypatch.setitem(sys.modules, "ragas.metrics", None)
        monkeypatch.setitem(sys.modules, "datasets", None)

        result = evaluate(full_record, metrics=["faithfulness"], save=False)

        assert "faithfulness" in result.errors
        assert "RAGAS not installed" in result.errors["faithfulness"]
        assert "faithfulness" not in result.metrics

    def test_ragas_runtime_failure_lands_in_errors_same_shape(self, full_record, fake_ragas):
        fake_ragas.raise_on_evaluate = RuntimeError("judge backend unreachable")

        result = evaluate(full_record, metrics=["faithfulness"], save=False)

        assert result.errors["faithfulness"] == "judge backend unreachable"
        assert "faithfulness" not in result.metrics
        # Same channel, same shape (metric -> str), no divergent keys.
        assert all(isinstance(v, str) for v in result.errors.values())


class TestMetricValidation:
    def test_unknown_metric_raises_listing_valid_names(self, full_record):
        with pytest.raises(ValueError, match="duplicates"):
            evaluate(full_record, metrics=["dupelicates"], save=False)

    def test_empty_metrics_list_raises(self, full_record):
        with pytest.raises(ValueError, match="metrics"):
            evaluate(full_record, metrics=[], save=False)


class TestPolicyOverride:
    """policy= must be honored, not silently ignored for a loaded default —
    the CLI's session batch path depends on it for its per-pipeline cache.
    Mirrors main's TestEvaluateRunPolicyOverride."""

    def test_evaluate_skips_load_policy_when_policy_given(self, full_record):
        custom_policy = InputQualityPolicy(min_top_chunk_score=0.99)
        with mock.patch("ragradar_evaluate.facade.load_policy") as mock_load:
            result = evaluate(
                full_record,
                metrics=list(INPUT_METRICS),
                policy=custom_policy,
                save=False,
            )

        mock_load.assert_not_called()
        # min_top_chunk_score=0.99 is violated by the fixture's rerank
        # scores (max 0.85) — proves the passed-in policy is the one
        # actually used, not silently ignored in favor of a default.
        assert "min_top_chunk_score" in result.policy_violations

    def test_evaluate_calls_load_policy_when_policy_omitted(self, full_record):
        with mock.patch(
            "ragradar_evaluate.facade.load_policy", wraps=_real_load_policy
        ) as mock_load:
            evaluate(full_record, metrics=list(INPUT_METRICS), save=False)

        mock_load.assert_called_once_with("__default")

    def test_check_skips_load_policy_and_uses_given_policy(self, full_record):
        custom_policy = InputQualityPolicy(min_top_chunk_score=0.99)
        with mock.patch("ragradar_evaluate.facade.load_policy") as mock_load:
            result = check(full_record, policy=custom_policy)

        mock_load.assert_not_called()
        assert result.thresholds == "policy"
        assert result.factors["top_chunk_score"]["status"] == "fail"
        assert result.verdict in ("warn", "fail")

    def test_evaluate_pipeline_selects_that_pipelines_policy(self, full_record):
        # Mirrors check(pipeline=...): evaluate scores against the named
        # pipeline's configured policy instead of the run's own.
        save_policy("strict_pipe", InputQualityPolicy(min_top_chunk_score=0.99))

        default_result = evaluate(full_record, metrics=list(INPUT_METRICS), save=False)
        strict_result = evaluate(
            full_record,
            metrics=list(INPUT_METRICS),
            pipeline="strict_pipe",
            save=False,
        )
        assert "min_top_chunk_score" not in default_result.policy_violations
        assert "min_top_chunk_score" in strict_result.policy_violations


class TestPrimitiveTargets:
    """Hand-built RunRecords with primitive nested data (the same shapes
    ragradar_capture.capture() accepts) must score, not AttributeError."""

    def _record(self):
        return RunRecord(
            query="q",
            response="r",
            chunks=[
                {"content": "alpha beta gamma", "rerank_score": 0.9},
                {"content": "delta epsilon", "rerank_score": 0.4},
            ],
            history_pre=[{"user": "hi"}],
            token_budget={
                "total_limit": 4096,
                "chunks_allocated": 2000,
                "history_allocated": 500,
                "system_allocated": 800,
            },
        )

    def test_evaluate_accepts_primitive_chunks(self):
        result = evaluate(self._record(), metrics=list(INPUT_METRICS), save=False)
        assert result.metrics["relevance"]["top_chunk_score"] == 0.9
        assert result.risk_score is not None

    def test_check_accepts_primitive_chunks(self):
        result = check(self._record())
        assert result.verdict in ("ok", "warn", "fail")
        assert result.factors["top_chunk_score"]["value"] == 0.9

    def test_input_record_is_not_mutated(self):
        rec = self._record()
        evaluate(rec, metrics=["relevance"], save=False)
        assert isinstance(rec.chunks[0], dict)  # caller's object untouched

    def test_uncoercible_record_raises_value_error(self):
        rec = RunRecord(query="q", response="r", chunks=[42])
        with pytest.raises(ValueError, match="coerce"):
            evaluate(rec, metrics=["relevance"], save=False)


class TestSavePersistence:
    def test_save_true_persists_what_it_returns(self, migrated_db):
        result = evaluate("s2r1", metrics=list(INPUT_METRICS), save=True)
        assert result.saved is True
        assert result.run_id == "s2r1"

        row = store.get_run(2, 1)
        assert json.loads(row["eval_scores"]) == result.to_eval_scores()
        assert row["risk_score"] == result.risk_score
        assert row["evaluated_at"] is not None

    def test_facade_matches_cli_input_only_persistence(self, migrated_db):
        # CLI --input-only persists for s2r1...
        cli_result = CliRunner().invoke(main, ["run", "s2r1", "--input-only"])
        assert cli_result.exit_code == 0
        with sqlite3.connect(str(migrated_db)) as conn:
            conn.row_factory = sqlite3.Row
            cli_row = conn.execute(
                "SELECT eval_scores, risk_score FROM runs WHERE session_id = 2 AND run_seq = 1"
            ).fetchone()

        # ...and the facade with the input-metric set persists the same.
        result = evaluate("s2r1", metrics=list(INPUT_METRICS), save=True)

        assert json.loads(cli_row["eval_scores"]) == result.to_eval_scores()
        assert cli_row["risk_score"] == result.risk_score

    def test_save_with_identityless_record_raises(self, full_record):
        with pytest.raises(ValueError, match="save=False"):
            evaluate(full_record, metrics=["duplicates"], save=True)

    def test_save_false_with_record_returns_unsaved(self, full_record):
        result = evaluate(full_record, metrics=["duplicates"], save=False)
        assert result.saved is False
        assert result.run_id is None


class TestTargetResolution:
    def test_missing_run_raises_with_id(self, migrated_db):
        with pytest.raises(ValueError, match="s9r9"):
            evaluate("s9r9", save=False)

    def test_bad_format_raises(self):
        with pytest.raises(ValueError, match="sNrN"):
            evaluate("not-a-run", save=False)

    def test_uncommitted_capture_like_object_raises(self):
        class FakeCapture:
            run_id = None

        with pytest.raises(ValueError, match="commit"):
            evaluate(FakeCapture(), save=False)


class TestCacheRiskIntegration:
    """cache_risk is the one input metric that keys off record.cache
    instead of record.chunks -- these prove evaluate()'s per-metric gate
    actually threads that through, notably for the cache-hit-skips-
    retrieval scenario where chunks is None but cache data exists."""

    def test_fires_with_no_chunks_when_cache_was_checked(self):
        from ragradar_core.schema import CacheRecord

        rec = RunRecord(
            query="q",
            response="r",
            cache=CacheRecord(checked=True, hit=True, similarity_score=0.98, threshold=0.9),
        )
        result = evaluate(rec, metrics=["cache_risk"], save=False)

        assert "cache_risk" not in result.skipped
        assert result.metrics["cache_risk"]["cache_hit"] is True

    def test_skipped_as_not_applicable_when_cache_never_checked(self, full_record):
        result = evaluate(full_record, metrics=["cache_risk"], save=False)
        assert result.skipped["cache_risk"] == "not applicable: run never checked a semantic cache"
        assert "cache_risk" not in result.metrics

    def test_borderline_and_stale_flagged_through_evaluate(self):
        from datetime import datetime, timedelta, timezone

        from ragradar_core.schema import CacheRecord

        old_time = datetime.now(timezone.utc) - timedelta(days=2)
        rec = RunRecord(
            query="q",
            response="r",
            cache=CacheRecord(
                checked=True,
                hit=True,
                similarity_score=0.91,
                threshold=0.9,
                cached_at=old_time.isoformat(),
            ),
        )
        policy = InputQualityPolicy(cache_borderline_margin=0.03, cache_max_age_seconds=3600)
        result = evaluate(rec, metrics=["cache_risk"], policy=policy, save=False)

        assert result.metrics["cache_risk"]["borderline_hit"] is True
        assert result.metrics["cache_risk"]["stale_hit"] is True

    def test_other_input_metrics_unaffected_by_cache_gate(self, full_record):
        # A record with both chunks and no cache data still computes the
        # 5 chunk-based metrics exactly as before cache_risk existed.
        result = evaluate(full_record, metrics=list(INPUT_METRICS), save=False)
        for name in INPUT_METRICS:
            if name == "cache_risk":
                continue
            assert name in result.metrics
