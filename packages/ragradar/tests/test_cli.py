import json
from datetime import datetime, timedelta, timezone

from click.testing import CliRunner
from ragradar.cli import main
from ragradar_core import store as core_store
from ragradar_core.schema import CacheRecord, RunRecord


class TestList:
    def test_sessions_exits_0(self, populated_db):
        result = CliRunner().invoke(main, ["list"])
        assert result.exit_code == 0
        assert "s2" in result.output
        assert "s1" in result.output

    def test_runs_in_session(self, populated_db):
        result = CliRunner().invoke(main, ["list", "s2"])
        assert result.exit_code == 0
        assert "r1" in result.output
        assert "r2" in result.output

    def test_empty_db(self):
        result = CliRunner().invoke(main, ["list"])
        assert result.exit_code == 0
        assert "No sessions found" in result.output


class TestFind:
    def test_exits_0(self, populated_db):
        result = CliRunner().invoke(main, ["find", "score"])
        assert result.exit_code == 0

    def test_shows_results(self, populated_db):
        result = CliRunner().invoke(main, ["find", "RRF"])
        assert result.exit_code == 0
        assert "RRF" in result.output

    def test_no_match(self, populated_db):
        result = CliRunner().invoke(main, ["find", "zzzznonexistent"])
        assert result.exit_code == 0
        assert "No matching runs found" in result.output

    def test_recent(self, populated_db):
        result = CliRunner().invoke(main, ["find", "--recent", "1"])
        assert result.exit_code == 0

    def test_pipeline_filter(self, populated_db):
        result = CliRunner().invoke(main, ["find", "--pipeline", "pipe_a"])
        assert result.exit_code == 0

    def test_session_filter(self, populated_db):
        result = CliRunner().invoke(main, ["find", "--session", "s2", "score"])
        assert result.exit_code == 0

    def test_on_empty_database(self):
        result = CliRunner().invoke(main, ["find", "anything"])
        assert result.exit_code == 0
        assert "No matching runs found" in result.output

    def test_exact_flag(self, populated_db):
        result = CliRunner().invoke(main, ["find", "score scale differences", "--exact"])
        assert result.exit_code == 0
        assert "Search results (1)" in result.output

        result = CliRunner().invoke(main, ["find", "score ANN", "--exact"])
        assert result.exit_code == 0
        assert "No matching runs found" in result.output

        result = CliRunner().invoke(main, ["find", "score ANN"])
        assert result.exit_code == 0
        assert "No matching runs found" not in result.output

    def test_date_filters(self, ragradar_home):
        rec = RunRecord(query="date test query", response="r")
        conn = core_store.connect()
        conn.execute(
            "INSERT INTO sessions (session_id, title, pipeline, created_at) "
            "VALUES (1, NULL, 'p', '2026-06-10T10:00:00+00:00')"
        )
        conn.execute(
            "INSERT INTO runs (session_id, run_seq, query, pipeline, created_at, run_data) "
            "VALUES (1, 1, ?, 'p', '2026-06-10T10:00:00+00:00', ?)",
            (rec.query, json.dumps(rec.to_json())),
        )
        conn.execute(
            "INSERT INTO runs (session_id, run_seq, query, pipeline, created_at, run_data) "
            "VALUES (1, 2, ?, 'p', '2026-06-15T10:00:00+00:00', ?)",
            ("later query", json.dumps(RunRecord(query="later query", response="r").to_json())),
        )
        conn.commit()
        conn.close()

        result = CliRunner().invoke(main, ["find", "--from", "2026-06-14"])
        assert result.exit_code == 0
        assert "later query" in result.output
        assert "date test query" not in result.output

        result = CliRunner().invoke(main, ["find", "--from", "2026-06-10", "--to", "2026-06-12"])
        assert result.exit_code == 0
        assert "date test query" in result.output
        assert "later query" not in result.output

    def test_disambiguation_screen(self, populated_db):
        result = CliRunner().invoke(main, ["find", "BM25"])
        assert result.exit_code == 0
        assert result.output.count("BM25") >= 2


class TestExplain:
    def test_exits_0(self, populated_db):
        result = CliRunner().invoke(main, ["explain"])
        assert result.exit_code == 0

    def test_specific_target(self, populated_db):
        result = CliRunner().invoke(main, ["explain", "s2r1"])
        assert result.exit_code == 0
        assert "RRF" in result.output

    def test_full_mode(self, populated_db):
        result = CliRunner().invoke(main, ["explain", "s2r1", "--full"])
        assert result.exit_code == 0

    def test_html_output(self, populated_db, ragradar_home):
        result = CliRunner().invoke(main, ["explain", "s2r1", "--html"])
        assert result.exit_code == 0
        assert "Report written to" in result.output
        reports_dir = ragradar_home / ".ragradar" / "reports"
        assert reports_dir.exists()
        html_files = list(reports_dir.glob("*.html"))
        assert len(html_files) == 1

    def test_no_runs(self):
        result = CliRunner().invoke(main, ["explain"])
        assert "No runs found" in result.output

    def test_factors_skip_on_empty_record(self, populated_db):
        result = CliRunner().invoke(main, ["explain", "s1r1"])
        assert result.exit_code == 0
        assert "BM25" in result.output


class TestDiff:
    def test_exits_0(self, populated_db):
        result = CliRunner().invoke(main, ["diff", "s2r1", "s2r2"])
        assert result.exit_code == 0
        assert "Comparing" in result.output

    def test_nonexistent_target(self, populated_db):
        result = CliRunner().invoke(main, ["diff", "s99r99", "s2r1"])
        assert "Could not resolve" in result.output


class TestBudget:
    def test_exits_0(self, populated_db):
        result = CliRunner().invoke(main, ["budget", "s2r1"])
        assert result.exit_code == 0
        assert "Token Usage" in result.output

    def test_no_budget_data(self, populated_db):
        result = CliRunner().invoke(main, ["budget", "s1r1"])
        assert "No token budget data" in result.output


class TestExplainHtmlMinimalRecord:
    def test_explain_html_minimal_record(self, ragradar_home):
        rec = RunRecord(query="minimal query", response="minimal response")
        conn = core_store.connect()
        conn.execute(
            "INSERT INTO sessions (session_id, title, pipeline, created_at) "
            "VALUES (1, NULL, 'p', '2026-06-10T10:00:00+00:00')"
        )
        conn.execute(
            "INSERT INTO runs (session_id, run_seq, query, pipeline, created_at, run_data) "
            "VALUES (1, 1, ?, 'p', '2026-06-10T10:00:00+00:00', ?)",
            (rec.query, json.dumps(rec.to_json())),
        )
        conn.commit()
        conn.close()
        result = CliRunner().invoke(main, ["explain", "s1r1", "--html"])
        assert result.exit_code == 0
        assert "Report written to" in result.output
        reports_dir = ragradar_home / ".ragradar" / "reports"
        html_files = list(reports_dir.glob("*.html"))
        assert len(html_files) == 1
        content = html_files[0].read_text(encoding="utf-8")
        assert "<html>" in content


class TestExplainEvalScores:
    def test_explain_shows_eval_scores_when_present(self, ragradar_home):
        rec = RunRecord(query="eval query", response="eval response")
        conn = core_store.connect()
        conn.execute(
            "INSERT INTO sessions (session_id, title, pipeline, created_at) "
            "VALUES (1, NULL, 'p', '2026-06-10T10:00:00+00:00')"
        )
        eval_scores = json.dumps(
            {
                "input": {"policy_violations": [], "mean_relevance": 0.8},
            }
        )
        conn.execute(
            "INSERT INTO runs (session_id, run_seq, query, pipeline, created_at, run_data, eval_scores, risk_score, evaluated_at) "
            "VALUES (1, 1, ?, 'p', '2026-06-10T10:00:00+00:00', ?, ?, 0.15, '2026-06-10T10:05:00+00:00')",
            (rec.query, json.dumps(rec.to_json()), eval_scores),
        )
        conn.commit()
        conn.close()
        result = CliRunner().invoke(main, ["explain", "s1r1"])
        assert result.exit_code == 0
        assert "Evaluation Scores" in result.output


class TestExplainSemanticCache:
    def _insert(self, cache: CacheRecord, session_id: int = 1):
        rec = RunRecord(query="cache query", response="cache response", cache=cache)
        conn = core_store.connect()
        conn.execute(
            "INSERT INTO sessions (session_id, title, pipeline, created_at) "
            "VALUES (?, NULL, 'p', '2026-06-10T10:00:00+00:00')",
            (session_id,),
        )
        conn.execute(
            "INSERT INTO runs (session_id, run_seq, query, pipeline, created_at, run_data) "
            "VALUES (?, 1, ?, 'p', '2026-06-10T10:00:00+00:00', ?)",
            (session_id, rec.query, json.dumps(rec.to_json())),
        )
        conn.commit()
        conn.close()

    def test_no_cache_data_skips_panel_silently(self, ragradar_home):
        rec = RunRecord(query="q", response="r")
        conn = core_store.connect()
        conn.execute(
            "INSERT INTO sessions (session_id, title, pipeline, created_at) "
            "VALUES (1, NULL, 'p', '2026-06-10T10:00:00+00:00')"
        )
        conn.execute(
            "INSERT INTO runs (session_id, run_seq, query, pipeline, created_at, run_data) "
            "VALUES (1, 1, ?, 'p', '2026-06-10T10:00:00+00:00', ?)",
            (rec.query, json.dumps(rec.to_json())),
        )
        conn.commit()
        conn.close()
        result = CliRunner().invoke(main, ["explain", "s1r1"])
        assert result.exit_code == 0
        assert "Cache behavior" not in result.output

    def test_checked_miss_renders_panel(self, ragradar_home):
        self._insert(CacheRecord(checked=True, hit=False))
        result = CliRunner().invoke(main, ["explain", "s1r1"])
        assert result.exit_code == 0
        assert "Cache behavior" in result.output
        assert "miss" in result.output

    def test_checked_hit_borderline_explains_why(self, ragradar_home):
        self._insert(
            CacheRecord(
                checked=True,
                hit=True,
                similarity_score=0.91,
                threshold=0.9,
                cached_at=datetime.now(timezone.utc).isoformat(),
            )
        )
        result = CliRunner().invoke(main, ["explain", "s1r1"])
        assert result.exit_code == 0
        assert "Borderline hit" in result.output

    def test_checked_hit_stale_explains_why(self, ragradar_home):
        old_time = datetime.now(timezone.utc) - timedelta(days=2)
        self._insert(
            CacheRecord(
                checked=True,
                hit=True,
                similarity_score=0.98,
                threshold=0.9,
                cached_at=old_time.isoformat(),
            )
        )
        result = CliRunner().invoke(main, ["explain", "s1r1"])
        assert result.exit_code == 0
        assert "Stale hit" in result.output


class TestSessionRename:
    def test_renames(self, populated_db):
        result = CliRunner().invoke(main, ["session", "rename", "s1", "My Title"])
        assert result.exit_code == 0
        assert "My Title" in result.output

        result = CliRunner().invoke(main, ["list"])
        assert "My Title" in result.output
