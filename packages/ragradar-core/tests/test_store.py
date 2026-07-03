import json
import sqlite3
from datetime import datetime, timedelta, timezone

from ragradar_core import store
from ragradar_core.schema import RunRecord


class TestConnectFreshDb:
    def test_creates_dir_and_db(self):
        conn = store.connect()
        conn.close()
        path = store.db_path()
        assert path.exists()
        assert path.name == "runs.db"
        assert path.parent.name == ".ragradar"

    def test_fresh_db_is_at_latest_version(self):
        store.connect().close()
        with sqlite3.connect(str(store.db_path())) as conn:
            row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        assert row is not None
        assert row[0] == store.SCHEMA_VERSION

    def test_fresh_db_has_all_tables_and_fts(self):
        store.connect().close()
        with sqlite3.connect(str(store.db_path())) as conn:
            tables = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        for table in ("meta", "sessions", "runs", "benchmark", "policies", "runs_fts"):
            assert table in tables, f"fresh DB missing table {table}"

    def test_fresh_db_has_eval_columns(self):
        store.connect().close()
        with sqlite3.connect(str(store.db_path())) as conn:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(runs)").fetchall()]
        for col in ("eval_scores", "risk_score", "evaluated_at"):
            assert col in cols

    def test_idempotent(self):
        store.connect().close()
        store.connect().close()
        with sqlite3.connect(str(store.db_path())) as conn:
            rows = conn.execute("SELECT COUNT(*) FROM meta").fetchone()
        assert rows[0] == 1

    def test_ensure_store_returns_db_path(self):
        assert store.ensure_store() == store.db_path()
        assert store.db_path().exists()


class TestConnectMigratesOldDb:
    def test_v1_db_migrated_to_latest_with_data_intact(self, v1_db):
        with sqlite3.connect(str(v1_db)) as raw:
            before = raw.execute(
                "SELECT run_data FROM runs WHERE session_id = 2 AND run_seq = 1"
            ).fetchone()[0]

        conn = store.connect()
        try:
            ver = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()[0]
            assert ver == store.SCHEMA_VERSION

            tables = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            assert {"benchmark", "policies", "runs_fts"} <= tables

            after = conn.execute(
                "SELECT run_data FROM runs WHERE session_id = 2 AND run_seq = 1"
            ).fetchone()[0]
            assert json.loads(before) == json.loads(after)
        finally:
            conn.close()

    def test_unsupported_version_raises(self, v1_db):
        with sqlite3.connect(str(v1_db)) as raw:
            raw.execute("UPDATE meta SET value = '99' WHERE key = 'schema_version'")

        try:
            store.connect()
            assert False, "connect() should have raised for unsupported version"
        except RuntimeError as e:
            assert "99" in str(e)


class TestGetOrCreateSession:
    def test_creates_new_session(self):
        sid = store.get_or_create_session("test_pipe")
        assert sid >= 1

    def test_reuses_session_within_gap(self):
        s1 = store.get_or_create_session("test_pipe")
        s2 = store.get_or_create_session("test_pipe")
        assert s1 == s2

    def test_creates_new_session_after_gap(self):
        s1 = store.get_or_create_session("test_pipe")
        old_time = (datetime.now(timezone.utc) - timedelta(minutes=60)).isoformat()
        with sqlite3.connect(str(store.db_path())) as conn:
            conn.execute(
                "UPDATE sessions SET created_at = ? WHERE session_id = ?",
                (old_time, s1),
            )
        s2 = store.get_or_create_session("test_pipe")
        assert s2 != s1

    def test_separate_pipelines_get_separate_sessions(self):
        s1 = store.get_or_create_session("pipe_a")
        s2 = store.get_or_create_session("pipe_b")
        assert s1 != s2

    def test_none_pipeline(self):
        s1 = store.get_or_create_session(None)
        s2 = store.get_or_create_session(None)
        assert s1 == s2


class TestWriteRun:
    def test_write_and_get_round_trip(self):
        sid = store.get_or_create_session("test_pipe")
        seq = store.next_run_seq(sid)
        rec = RunRecord(query="test query", response="test response")
        store.write_run(sid, seq, rec, "test_pipe")

        row = store.get_run(sid, seq)
        assert row is not None
        assert row["query"] == "test query"
        assert row["pipeline"] == "test_pipe"
        restored = RunRecord.from_json(json.loads(row["run_data"]))
        assert restored.to_json() == rec.to_json()

    def test_get_run_missing_returns_none(self):
        assert store.get_run(99, 99) is None

    def test_next_run_seq_increments(self):
        sid = store.get_or_create_session("test_pipe")
        assert store.next_run_seq(sid) == 1
        rec = RunRecord(query="q", response="r")
        store.write_run(sid, 1, rec, "test_pipe")
        assert store.next_run_seq(sid) == 2

    def test_write_runs_batch(self):
        sid = store.get_or_create_session("batch_pipe")
        records = [RunRecord(query=f"q{i}", response=f"r{i}") for i in range(3)]
        store.write_runs_batch(sid, 1, records, "batch_pipe")

        rows = store.get_runs_in_session(sid)
        assert len(rows) == 3
        assert {r["query"] for r in rows} == {"q0", "q1", "q2"}
        assert store.next_run_seq(sid) == 4

    def test_get_latest_run(self):
        sid = store.get_or_create_session("test_pipe")
        store.write_run(sid, 1, RunRecord(query="first", response="a"), "test_pipe")
        latest = store.get_latest_run()
        assert latest is not None
        assert latest["query"] == "first"

    def test_get_latest_run_empty_store(self):
        assert store.get_latest_run() is None


class TestEvalScores:
    def _write_one(self, pipeline: str = "eval_pipe") -> tuple[int, int]:
        sid = store.get_or_create_session(pipeline)
        seq = store.next_run_seq(sid)
        store.write_run(sid, seq, RunRecord(query="q", response="r"), pipeline)
        return sid, seq

    def test_write_and_get_round_trip(self):
        sid, seq = self._write_one()
        scores = {"input": {"duplicate_ratio": 0.1}, "output": None}
        store.write_eval_scores(sid, seq, scores, 0.42)

        result = store.get_eval_scores(sid, seq)
        assert result is not None
        assert result["input"] == {"duplicate_ratio": 0.1}
        assert result["risk_score"] == 0.42

    def test_get_eval_scores_unevaluated_returns_none(self):
        sid, seq = self._write_one()
        assert store.get_eval_scores(sid, seq) is None

    def test_write_eval_scores_batch(self):
        sid, seq1 = self._write_one()
        seq2 = store.next_run_seq(sid)
        store.write_run(sid, seq2, RunRecord(query="q2", response="r2"), "eval_pipe")

        store.write_eval_scores_batch(
            [
                (sid, seq1, {"input": {"a": 1}}, 0.1),
                (sid, seq2, {"input": {"a": 2}}, 0.2),
            ]
        )

        assert store.get_eval_scores(sid, seq1)["risk_score"] == 0.1
        assert store.get_eval_scores(sid, seq2)["risk_score"] == 0.2

    def test_get_all_evaluated_runs_filters(self):
        sid_a, seq_a = self._write_one("pipe_a")
        store.write_eval_scores(sid_a, seq_a, {"input": {}}, 0.5)

        sid_b, seq_b = self._write_one("pipe_b")
        store.write_eval_scores(sid_b, seq_b, {"input": {}}, 0.6)

        # An unevaluated run must never appear.
        seq_c = store.next_run_seq(sid_a)
        store.write_run(sid_a, seq_c, RunRecord(query="q3", response="r3"), "pipe_a")

        all_runs = store.get_all_evaluated_runs()
        assert len(all_runs) == 2

        pipe_a_runs = store.get_all_evaluated_runs("pipe_a")
        assert len(pipe_a_runs) == 1
        assert pipe_a_runs[0]["pipeline"] == "pipe_a"

        assert store.get_all_evaluated_runs("nonexistent") == []


class TestBenchmark:
    def test_write_and_get(self):
        store.write_benchmark_entry("pipe_a", "duplicate_ratio", 0.2, -0.8, 12)
        rows = store.get_benchmark("pipe_a")
        assert len(rows) == 1
        assert rows[0]["factor"] == "duplicate_ratio"
        assert rows[0]["threshold"] == 0.2
        assert rows[0]["correlation"] == -0.8
        assert rows[0]["sample_count"] == 12

    def test_upsert_replaces(self):
        store.write_benchmark_entry("pipe_a", "duplicate_ratio", 0.2, -0.8, 12)
        store.write_benchmark_entry("pipe_a", "duplicate_ratio", 0.3, -0.9, 15)
        rows = store.get_benchmark("pipe_a")
        assert len(rows) == 1
        assert rows[0]["threshold"] == 0.3

    def test_batch_write(self):
        store.write_benchmark_entries_batch(
            [
                ("pipe_a", "duplicate_ratio", 0.2, -0.8, 12),
                ("pipe_a", "top_chunk_score", 0.7, 0.6, 12),
            ]
        )
        rows = store.get_benchmark("pipe_a")
        assert len(rows) == 2

    def test_pipeline_scoping(self):
        store.write_benchmark_entry("pipe_a", "duplicate_ratio", 0.2, -0.8, 12)
        assert store.get_benchmark("pipe_b") == []


class TestPolicy:
    def test_write_and_get(self):
        store.write_policy("pipe_a", {"max_duplicate_ratio": 0.5})
        assert store.get_policy("pipe_a") == {"max_duplicate_ratio": 0.5}

    def test_get_missing_returns_none(self):
        assert store.get_policy("nonexistent") is None

    def test_delete(self):
        store.write_policy("pipe_a", {"max_duplicate_ratio": 0.5})
        store.delete_policy("pipe_a")
        assert store.get_policy("pipe_a") is None

    def test_delete_missing_is_noop(self):
        store.delete_policy("nonexistent")
