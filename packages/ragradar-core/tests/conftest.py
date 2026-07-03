import json
import sqlite3

import pytest
from ragradar_core.schema import RunRecord

# The schema as ragradar-capture 0.1.0 created it (meta.schema_version "1"),
# kept verbatim so migration tests exercise a genuinely old database.
V1_SCHEMA = """
PRAGMA journal_mode=WAL;

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


@pytest.fixture(autouse=True)
def ragradar_home(tmp_path, monkeypatch):
    ragradar_dir = tmp_path / ".ragradar"
    monkeypatch.setattr("ragradar_core.store._ragradar_dir", lambda: ragradar_dir)
    return tmp_path


@pytest.fixture
def v1_db(ragradar_home):
    """A hand-built v1 database (schema_version '1') with two runs."""
    db_path = ragradar_home / ".ragradar" / "runs.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    rec_min = RunRecord(query="what is BM25", response="a ranking function")
    rec = RunRecord(
        query="does RRF handle score scale differences",
        response="Yes, RRF normalizes scores via reciprocal rank.",
    )

    with sqlite3.connect(str(db_path)) as conn:
        conn.executescript(V1_SCHEMA)
        conn.execute("INSERT INTO meta VALUES ('schema_version', '1')")
        conn.execute("INSERT INTO sessions VALUES (1, NULL, 'pipe_a', '2026-06-08T10:00:00+00:00')")
        conn.execute(
            "INSERT INTO sessions VALUES (2, 'RRF investigation', 'pipe_a', '2026-06-09T10:00:00+00:00')"
        )
        conn.execute(
            "INSERT INTO runs (session_id, run_seq, query, pipeline, created_at, run_data) "
            "VALUES (1, 1, ?, 'pipe_a', '2026-06-08T10:05:00+00:00', ?)",
            (rec_min.query, json.dumps(rec_min.to_json())),
        )
        conn.execute(
            "INSERT INTO runs (session_id, run_seq, query, pipeline, created_at, run_data) "
            "VALUES (2, 1, ?, 'pipe_a', '2026-06-09T10:05:00+00:00', ?)",
            (rec.query, json.dumps(rec.to_json())),
        )

    return db_path


@pytest.fixture
def migrated_db(v1_db):
    """v1 database after ensure_store() has walked it to the latest version."""
    from ragradar_core.store import ensure_store

    ensure_store()
    return v1_db
