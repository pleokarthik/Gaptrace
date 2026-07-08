import json
import sqlite3

import pytest
from gaptrace_core.store import ensure_store

# Same v1 layout as conftest.V1_SCHEMA (not importable under
# --import-mode=importlib, so restated here).
V1_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title      TEXT,
    pipeline   TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    session_id  INTEGER NOT NULL REFERENCES sessions(session_id),
    run_seq     INTEGER NOT NULL,
    query       TEXT NOT NULL,
    pipeline    TEXT,
    created_at  TEXT NOT NULL,
    run_data    TEXT NOT NULL,
    PRIMARY KEY (session_id, run_seq)
);

CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at);
CREATE INDEX IF NOT EXISTS idx_runs_query      ON runs(query);
CREATE INDEX IF NOT EXISTS idx_runs_pipeline   ON runs(pipeline);
"""


# ---------------------------------------------------------------------------
# Helper: build a genuine v2 database (v1→v2 applied, v2→v3 NOT yet applied)
# ---------------------------------------------------------------------------
def _make_v2_db(db_path):
    """Return path to a v2 database with one session and two runs."""
    with sqlite3.connect(str(db_path)) as conn:
        conn.executescript(V1_SCHEMA)  # creates idx_runs_query
        conn.execute("INSERT INTO meta VALUES ('schema_version', '1')")
        conn.execute("INSERT INTO sessions VALUES (1, NULL, 'pipe_a', '2026-06-08T10:00:00+00:00')")
        conn.execute(
            "INSERT INTO runs (session_id, run_seq, query, pipeline, created_at, run_data) "
            "VALUES (1, 1, 'what is token budget', 'pipe_a', '2026-06-08T10:05:00+00:00', '{}')"
        )
        conn.execute(
            "INSERT INTO runs (session_id, run_seq, query, pipeline, created_at, run_data) "
            "VALUES (1, 2, 'explain BM25 ranking', 'pipe_a', '2026-06-08T10:06:00+00:00', '{}')"
        )
        # v1 → v2 manually (same as what the migration chain does for the v1 block)
        conn.execute("ALTER TABLE runs ADD COLUMN eval_scores TEXT")
        conn.execute("ALTER TABLE runs ADD COLUMN risk_score  REAL")
        conn.execute("ALTER TABLE runs ADD COLUMN evaluated_at TEXT")
        conn.execute(
            """CREATE TABLE IF NOT EXISTS benchmark (
                pipeline TEXT NOT NULL, factor TEXT NOT NULL,
                threshold REAL, correlation REAL,
                sample_count INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (pipeline, factor)
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS policies (
                pipeline TEXT PRIMARY KEY,
                policy_data TEXT NOT NULL, updated_at TEXT NOT NULL
            )"""
        )
        conn.execute("UPDATE meta SET value = '2' WHERE key = 'schema_version'")
    return db_path


@pytest.fixture
def v2_db(gaptrace_home):
    """Genuine v2 database: has idx_runs_query, no runs_fts."""
    db_path = gaptrace_home / ".gaptrace" / "runs.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return _make_v2_db(db_path)


class TestMigration:
    def test_migration_from_v1(self, v1_db):
        ensure_store()

        with sqlite3.connect(str(v1_db)) as conn:
            conn.row_factory = sqlite3.Row
            cols = [r["name"] for r in conn.execute("PRAGMA table_info(runs)").fetchall()]
            assert "eval_scores" in cols
            assert "risk_score" in cols
            assert "evaluated_at" in cols

            tables = [
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            ]
            assert "benchmark" in tables
            assert "policies" in tables
            assert "runs_fts" in tables

            indexes = [
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index'"
                ).fetchall()
            ]
            assert "idx_runs_query" not in indexes

            ver = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
            assert ver["value"] == "3"

    def test_existing_data_intact(self, v1_db):
        with sqlite3.connect(str(v1_db)) as conn:
            before = conn.execute(
                "SELECT run_data FROM runs WHERE session_id = 2 AND run_seq = 1"
            ).fetchone()[0]

        ensure_store()

        with sqlite3.connect(str(v1_db)) as conn:
            after = conn.execute(
                "SELECT run_data FROM runs WHERE session_id = 2 AND run_seq = 1"
            ).fetchone()[0]

        assert json.loads(before) == json.loads(after)

    def test_idempotent(self, v1_db):
        ensure_store()
        ensure_store()

        with sqlite3.connect(str(v1_db)) as conn:
            conn.row_factory = sqlite3.Row
            ver = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
            assert ver["value"] == "3"

    def test_v3_is_noop(self, migrated_db):
        ensure_store()

    def test_unsupported_version_raises(self, v1_db):
        with sqlite3.connect(str(v1_db)) as conn:
            conn.execute("UPDATE meta SET value = '99' WHERE key = 'schema_version'")

        try:
            ensure_store()
            assert False, "Should have raised"
        except RuntimeError as e:
            assert "99" in str(e)

    def test_new_columns_nullable(self, migrated_db):
        with sqlite3.connect(str(migrated_db)) as conn:
            row = conn.execute(
                "SELECT eval_scores, risk_score, evaluated_at FROM runs WHERE session_id = 1 AND run_seq = 1"
            ).fetchone()
            assert row[0] is None
            assert row[1] is None
            assert row[2] is None

    # ------------------------------------------------------------------
    # v2 → v3 path
    # ------------------------------------------------------------------

    def test_migration_from_v2_reaches_v3(self, v2_db):
        """v2 start: ensure_store() must reach schema v3."""
        ensure_store()

        with sqlite3.connect(str(v2_db)) as conn:
            conn.row_factory = sqlite3.Row
            ver = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
            assert ver["value"] == "3"

    def test_migration_from_v2_drops_idx_runs_query(self, v2_db):
        """idx_runs_query must be present before migration and absent after."""
        with sqlite3.connect(str(v2_db)) as conn:
            before = [
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index'"
                ).fetchall()
            ]
        assert "idx_runs_query" in before, "precondition: v2 DB must have idx_runs_query"

        ensure_store()

        with sqlite3.connect(str(v2_db)) as conn:
            after = [
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index'"
                ).fetchall()
            ]
        assert "idx_runs_query" not in after

    def test_migration_from_v2_creates_runs_fts(self, v2_db):
        """runs_fts virtual table must not exist before and must exist after."""
        with sqlite3.connect(str(v2_db)) as conn:
            before = [
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            ]
        assert "runs_fts" not in before, "precondition: v2 DB must not have runs_fts"

        ensure_store()

        with sqlite3.connect(str(v2_db)) as conn:
            after = [
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            ]
        assert "runs_fts" in after

    def test_migration_from_v2_fts_populated_from_existing_rows(self, v2_db):
        """Existing rows at v2 must be searchable via FTS5 after migration."""
        ensure_store()

        with sqlite3.connect(str(v2_db)) as conn:
            # "what is token budget" should be in the index
            token_hits = conn.execute(
                "SELECT rowid FROM runs_fts WHERE runs_fts MATCH 'token'",
            ).fetchall()
            bm25_hits = conn.execute(
                "SELECT rowid FROM runs_fts WHERE runs_fts MATCH 'BM25'",
            ).fetchall()
        assert len(token_hits) == 1, f"expected 1 hit for 'token', got {len(token_hits)}"
        assert len(bm25_hits) == 1, f"expected 1 hit for 'BM25', got {len(bm25_hits)}"

    def test_migration_from_v2_tolerates_missing_idx_runs_query(self, v2_db):
        """DROP INDEX IF EXISTS must not error even if idx_runs_query is already gone."""
        with sqlite3.connect(str(v2_db)) as conn:
            conn.execute("DROP INDEX IF EXISTS idx_runs_query")

        ensure_store()  # must not raise

        with sqlite3.connect(str(v2_db)) as conn:
            ver = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()[0]
        assert ver == "3"

    def test_v2_to_v3_is_idempotent(self, v2_db):
        """Calling ensure_store() twice from v2 must end at v3 without error."""
        ensure_store()
        ensure_store()

        with sqlite3.connect(str(v2_db)) as conn:
            ver = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()[0]
        assert ver == "3"
