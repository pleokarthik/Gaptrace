import json
import sqlite3
from unittest import mock

from click.testing import CliRunner
from ragradar_core import store
from ragradar_core.schema import RunRecord
from ragradar_evaluate.cli import main


class TestMigrationOnCommand:
    def test_migration_runs(self, v1_db):
        runner = CliRunner()
        runner.invoke(main, ["policy", "show"])
        with sqlite3.connect(str(v1_db)) as conn:
            conn.row_factory = sqlite3.Row
            ver = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        assert ver["value"] == "3"


class TestRunCommand:
    def test_input_only(self, migrated_db):
        result = CliRunner().invoke(main, ["run", "--input-only"])
        assert result.exit_code == 0

    def test_specific_target(self, migrated_db):
        result = CliRunner().invoke(main, ["run", "s2r1", "--input-only"])
        assert result.exit_code == 0
        assert "Input Quality" in result.output

    def test_no_runs(self, ragradar_home):
        result = CliRunner().invoke(main, ["run"])
        assert "No runs found" in result.output


class TestPolicyCommands:
    def test_show_uses_defaults(self, migrated_db):
        result = CliRunner().invoke(main, ["policy", "show"])
        assert result.exit_code == 0
        assert "0.7" in result.output
        assert "0.2" in result.output

    def test_set_valid_field(self, migrated_db):
        result = CliRunner().invoke(main, ["policy", "set", "min_top_chunk_score", "0.8"])
        assert result.exit_code == 0
        assert "0.8" in result.output

    def test_set_invalid_field(self, migrated_db):
        result = CliRunner().invoke(main, ["policy", "set", "unknown_field", "0.8"])
        assert result.exit_code != 0

    def test_reset(self, migrated_db):
        CliRunner().invoke(main, ["policy", "set", "min_top_chunk_score", "0.99"])
        result = CliRunner().invoke(main, ["policy", "reset"])
        assert result.exit_code == 0
        assert "reset" in result.output.lower()


class TestBenchmarkCommands:
    def test_seed_then_show(self, migrated_db):
        result = CliRunner().invoke(main, ["benchmark", "seed", "test_pipe"])
        assert result.exit_code == 0
        assert "Seeded" in result.output

    def test_build_fails_under_10(self, migrated_db):
        result = CliRunner().invoke(main, ["benchmark", "build"])
        assert result.exit_code != 0

    def test_export(self, migrated_db):
        store.write_eval_scores(2, 1, {"input": {}, "output": {}}, 0.1)
        result = CliRunner().invoke(main, ["benchmark", "export"])
        assert result.exit_code == 0
        assert "Exported" in result.output

    def test_benchmark_show_after_build(self, migrated_db):
        CliRunner().invoke(main, ["benchmark", "seed", "show_pipe"])
        with sqlite3.connect(str(migrated_db)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT session_id, run_seq FROM runs WHERE pipeline = ?",
                ("show_pipe__seeded",),
            ).fetchall()
        for r in rows:
            store.write_eval_scores(
                r["session_id"],
                r["run_seq"],
                {
                    "input": {
                        "mean_relevance": 0.8,
                        "top_chunk_score": 0.9,
                        "duplicate_ratio": 0.0,
                        "high_score_truncations": 0,
                        "token_headroom_pct": 0.17,
                        "source_domain_count": 2,
                        "low_score_chunk_ratio": 0.0,
                    },
                    "output": {"faithfulness": 0.9, "answer_relevancy": 0.85},
                },
                0.1,
            )
        CliRunner().invoke(main, ["benchmark", "build", "--pipeline", "show_pipe__seeded"])
        result = CliRunner().invoke(main, ["benchmark", "show", "--pipeline", "show_pipe__seeded"])
        assert result.exit_code == 0
        output = result.output.lower()
        assert any(
            f in output
            for f in [
                "duplicate_ratio",
                "top_chunk_score",
                "mean_relevance",
                "token_headroom_pct",
                "low_score_chunk_ratio",
            ]
        )

    def test_benchmark_check_nonexistent_run(self, migrated_db):
        result = CliRunner().invoke(main, ["benchmark", "check", "s99r99"])
        assert result.exit_code == 1
        assert "Error" in result.output or "not found" in result.output.lower()
        assert "Traceback" not in result.output

    def test_benchmark_check_existing_run(self, migrated_db):
        CliRunner().invoke(main, ["benchmark", "seed", "check_pipe"])
        with sqlite3.connect(str(migrated_db)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT session_id, run_seq FROM runs WHERE pipeline = ?",
                ("check_pipe__seeded",),
            ).fetchall()
        for r in rows:
            store.write_eval_scores(
                r["session_id"],
                r["run_seq"],
                {
                    "input": {
                        "mean_relevance": 0.8,
                        "top_chunk_score": 0.9,
                        "duplicate_ratio": 0.0,
                        "high_score_truncations": 0,
                        "token_headroom_pct": 0.17,
                        "source_domain_count": 2,
                        "low_score_chunk_ratio": 0.0,
                    },
                    "output": {"faithfulness": 0.9, "answer_relevancy": 0.85},
                },
                0.1,
            )
        CliRunner().invoke(main, ["benchmark", "build", "--pipeline", "check_pipe__seeded"])
        first = rows[0]
        target = f"s{first['session_id']}r{first['run_seq']}"
        result = CliRunner().invoke(
            main, ["benchmark", "check", target, "--pipeline", "check_pipe__seeded"]
        )
        assert result.exit_code == 0
        output = result.output.lower()
        assert "ok" in output or "warn" in output or "fail" in output

    def test_benchmark_check_null_risk_renders_dash(self, migrated_db):
        # A run evaluated without input metrics has risk_score NULL;
        # check must render "-" instead of crashing on None.
        store.write_eval_scores(2, 1, {"input": None, "output": None}, None)
        result = CliRunner().invoke(main, ["benchmark", "check", "s2r1"])
        assert result.exit_code == 0
        assert "Risk score: -" in result.output
        assert "Traceback" not in result.output


class TestInputOnlyMinimalRecord:
    def test_input_only_minimal_record(self, migrated_db):
        rec = RunRecord(query="minimal", response="minimal response")
        with sqlite3.connect(str(migrated_db)) as conn:
            conn.execute(
                "INSERT INTO sessions VALUES (99, NULL, 'min_pipe', '2026-06-20T10:00:00+00:00')"
            )
            conn.execute(
                "INSERT INTO runs (session_id, run_seq, query, pipeline, created_at, run_data) "
                "VALUES (99, 1, ?, 'min_pipe', '2026-06-20T10:00:00+00:00', ?)",
                (rec.query, json.dumps(rec.to_json())),
            )
        result = CliRunner().invoke(main, ["run", "s99r1", "--input-only"])
        assert result.exit_code == 0


class TestBothFlagsError:
    def test_input_only_with_output_only_is_a_clean_error(self, migrated_db):
        # Previously this combination silently computed nothing; it now
        # errors out with a clear message and non-zero exit.
        result = CliRunner().invoke(main, ["run", "s2r1", "--input-only", "--output-only"])
        assert result.exit_code != 0
        assert "exclude each other" in result.output
        assert "Traceback" not in result.output


class TestOutputOnlyWithoutRagas:
    def test_output_only_without_ragas(self, migrated_db):
        with mock.patch.dict(
            "sys.modules", {"ragas": None, "ragas.metrics": None, "datasets": None}
        ):
            result = CliRunner().invoke(main, ["run", "s2r1", "--output-only"])
        assert result.exit_code == 0
        assert "RAGAS not installed" in result.output
