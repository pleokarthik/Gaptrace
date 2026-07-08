import json
import sqlite3

import pytest
from gaptrace_core import store
from gaptrace_evaluate.benchmark import builder, checker, exporter, seeder


def _insert_evaluated_run(db_path, session_id, run_seq, pipeline, input_scores, output_scores):
    """Insert a run with eval_scores already set."""
    from gaptrace_core.schema import ChunkRecord, RunRecord

    rec = RunRecord(
        query=f"query {run_seq}",
        response=f"response {run_seq}",
        chunks=[
            ChunkRecord(
                chunk_id=f"c{run_seq}",
                source_doc_id=f"d{run_seq}",
                content=f"chunk content {run_seq}",
                token_count=50,
                rerank_score=input_scores.get("top_chunk_score", 0.8),
            ),
        ],
    )

    eval_scores = {"input": input_scores, "output": output_scores}

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO sessions VALUES (?, NULL, ?, '2026-06-10T10:00:00+00:00')",
            (session_id, pipeline),
        )
        conn.execute(
            "INSERT INTO runs (session_id, run_seq, query, pipeline, created_at, "
            "run_data, eval_scores, risk_score, evaluated_at) "
            "VALUES (?, ?, ?, ?, '2026-06-10T10:00:00+00:00', ?, ?, 0.0, '2026-06-10T10:00:00+00:00')",
            (
                session_id,
                run_seq,
                rec.query,
                pipeline,
                json.dumps(rec.to_json()),
                json.dumps(eval_scores),
            ),
        )


class TestBuilder:
    def test_requires_minimum_10_runs(self, migrated_db):
        with pytest.raises(ValueError, match="at least 10"):
            builder.build("pipe_a")

    def test_build_with_exactly_10_runs(self, migrated_db):
        for i in range(10):
            score_val = 0.5 + i * 0.04
            _insert_evaluated_run(
                migrated_db,
                session_id=20,
                run_seq=i + 1,
                pipeline="boundary_pipe",
                input_scores={
                    "duplicate_ratio": 0.05 * (i % 4),
                    "top_chunk_score": score_val,
                    "mean_relevance": score_val,
                },
                output_scores={
                    "faithfulness": score_val + 0.05,
                    "answer_relevancy": score_val,
                },
            )
        result = builder.build("boundary_pipe")
        assert result["run_count"] == 10
        assert len(result["factors"]) > 0

    def test_build_with_9_runs_raises(self, migrated_db):
        for i in range(9):
            _insert_evaluated_run(
                migrated_db,
                session_id=21,
                run_seq=i + 1,
                pipeline="nine_pipe",
                input_scores={"top_chunk_score": 0.8},
                output_scores={"faithfulness": 0.9},
            )
        with pytest.raises(ValueError, match="at least 10"):
            builder.build("nine_pipe")

    def test_produces_correlations(self, migrated_db):
        for i in range(12):
            score_val = 0.5 + i * 0.03
            _insert_evaluated_run(
                migrated_db,
                session_id=10,
                run_seq=i + 1,
                pipeline="bench_pipe",
                input_scores={
                    "duplicate_ratio": 0.1 * (i % 3),
                    "top_chunk_score": score_val,
                    "mean_relevance": score_val,
                    "high_score_truncations": i % 2,
                    "token_headroom_pct": 0.2,
                    "source_domain_count": 2,
                    "low_score_chunk_ratio": 0.1,
                },
                output_scores={
                    "faithfulness": score_val + 0.05,
                    "answer_relevancy": score_val,
                },
            )
        result = builder.build("bench_pipe")
        assert result["run_count"] == 12
        assert len(result["factors"]) > 0


class TestSeeder:
    def test_creates_correct_count(self, migrated_db):
        n = seeder.seed("test_pipe", count=10)
        assert n == 10

    def test_marks_pipeline(self, migrated_db):
        seeder.seed("test_pipe", count=4)
        with sqlite3.connect(str(migrated_db)) as conn:
            rows = conn.execute(
                "SELECT pipeline FROM runs WHERE pipeline = 'test_pipe__seeded'"
            ).fetchall()
        assert len(rows) == 4


class TestExporter:
    def test_produces_valid_jsonl(self, migrated_db, tmp_path):
        _insert_evaluated_run(
            migrated_db,
            session_id=10,
            run_seq=1,
            pipeline="export_pipe",
            input_scores={"mean_relevance": 0.8},
            output_scores={"faithfulness": 0.9},
        )
        path = exporter.export("export_pipe", tmp_path / "out.jsonl")
        assert path.exists()
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert "question" in data
        assert "answer" in data
        assert "contexts" in data

    def test_skips_seeded_runs(self, migrated_db, tmp_path):
        _insert_evaluated_run(
            migrated_db,
            session_id=10,
            run_seq=1,
            pipeline="real_pipe",
            input_scores={},
            output_scores={},
        )
        _insert_evaluated_run(
            migrated_db,
            session_id=10,
            run_seq=2,
            pipeline="real_pipe__seeded",
            input_scores={},
            output_scores={},
        )
        path = exporter.export(output_path=tmp_path / "out.jsonl")
        with open(path) as f:
            lines = f.readlines()
        for line in lines:
            assert "__seeded" not in json.loads(line).get("pipeline", "")


class TestChecker:
    def test_ok_on_good_run(self, migrated_db):
        _insert_evaluated_run(
            migrated_db,
            session_id=10,
            run_seq=1,
            pipeline="check_pipe",
            input_scores={
                "duplicate_ratio": 0.0,
                "top_chunk_score": 0.9,
                "high_score_truncations": 0,
                "token_headroom_pct": 0.2,
                "source_domain_count": 2,
                "low_score_chunk_ratio": 0.1,
            },
            output_scores={"faithfulness": 0.9},
        )
        result = checker.check(10, 1, "check_pipe")
        assert result["overall"] == "ok"

    def test_fail_on_bad_run(self, migrated_db):
        _insert_evaluated_run(
            migrated_db,
            session_id=10,
            run_seq=1,
            pipeline="check_pipe",
            input_scores={
                "duplicate_ratio": 0.5,
                "top_chunk_score": 0.3,
                "high_score_truncations": 5,
                "token_headroom_pct": 0.05,
                "source_domain_count": 10,
                "low_score_chunk_ratio": 0.8,
            },
            output_scores={"faithfulness": 0.2},
        )
        store.write_eval_scores(10, 1, {}, 0.9)
        result = checker.check(10, 1, "check_pipe")
        assert result["overall"] == "fail"

    def test_null_risk_score_treated_as_unknown(self, migrated_db):
        # evaluate() persists risk_score NULL when input metrics were not
        # computed; check must skip the risk threshold, not compare None.
        _insert_evaluated_run(
            migrated_db,
            session_id=10,
            run_seq=1,
            pipeline="check_pipe",
            input_scores={},
            output_scores={},
        )
        store.write_eval_scores(10, 1, {"input": None, "output": None}, None)
        result = checker.check(10, 1, "check_pipe")
        assert result["risk_score"] is None
        assert result["overall"] == "ok"
