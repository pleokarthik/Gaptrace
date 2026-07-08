import json
import sqlite3
import sys
import types

import pytest
from gaptrace_core.schema import (
    CacheEvent,
    ChunkRecord,
    RunRecord,
    TokenBudget,
    TokenUsage,
    Turn,
)

# The schema as gaptrace-capture 0.1.0 created it (meta.schema_version "1"),
# kept verbatim so the v1_db fixture builds a genuinely old database.
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


@pytest.fixture(autouse=True)
def gaptrace_home(tmp_path, monkeypatch):
    gaptrace_dir = tmp_path / ".gaptrace"
    # gaptrace_core.store._gaptrace_dir is the one canonical home — every package
    # routes store access through it, so patching it here is sufficient.
    monkeypatch.setattr("gaptrace_core.store._gaptrace_dir", lambda: gaptrace_dir)
    return tmp_path


def _full_record():
    return RunRecord(
        query="does RRF handle score scale differences",
        response="Yes, RRF normalizes scores via reciprocal rank.",
        chunks=[
            ChunkRecord(
                chunk_id="c1",
                source_doc_id="d1",
                content="RRF normalizes retrieval scores across methods",
                token_count=50,
                retrieval_score=0.9,
                rerank_score=0.85,
                retrieval_path="bm25",
                truncated=False,
                cache_hit=True,
            ),
            ChunkRecord(
                chunk_id="c2",
                source_doc_id="d2",
                content="Score fusion combines signals from multiple retrievers",
                token_count=30,
                retrieval_score=0.7,
                rerank_score=0.4,
                retrieval_path="ann",
                truncated=True,
                cache_hit=False,
            ),
        ],
        final_prompt="System: answer\nContext: ...\nQuery: RRF",
        token_budget=TokenBudget(
            total_limit=4096,
            chunks_allocated=2000,
            history_allocated=500,
            system_allocated=800,
            headroom=796,
        ),
        history_pre=[
            Turn(role="user", content="hello", tokens=3),
            Turn(role="assistant", content="hi there", tokens=5),
        ],
        history_post=[Turn(role="user", content="hello", tokens=3)],
        eviction_reason="token_budget",
        cache_events=[
            CacheEvent(chunk_id="c1", hit=True, cache_source="disk"),
            CacheEvent(chunk_id="c2", hit=False),
        ],
        model="gpt-4",
        token_usage=TokenUsage(input_tokens=300, output_tokens=50, total_tokens=350),
    )


@pytest.fixture
def v1_db(gaptrace_home):
    """Create a v1 schema database with existing data."""
    db_path = gaptrace_home / ".gaptrace" / "runs.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    rec = _full_record()
    rec_min = RunRecord(query="what is BM25", response="a ranking function")

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
    """v1 database after gaptrace_core has walked it to the latest version."""
    from gaptrace_core.store import ensure_store

    ensure_store()
    return v1_db


@pytest.fixture
def full_record():
    return _full_record()


class _FakeRagas:
    """Recorder handed to tests using the fake_ragas fixture."""

    def __init__(self):
        self.calls: list[dict] = []
        self.scores = {
            "faithfulness": 0.9,
            "answer_relevancy": 0.85,
            "context_precision": 0.8,
            "context_recall": 0.75,
        }
        self.metric_objects = {name: object() for name in self.scores}
        self.raise_on_evaluate: Exception | None = None


@pytest.fixture
def fake_ragas(monkeypatch):
    """Install an in-memory ragas/datasets stand-in so output metrics run
    without the heavyweight dependency or an LLM. Returns a recorder:
    .calls captures each ragas.evaluate invocation (dataset + metric
    objects), .scores is mutable, .raise_on_evaluate simulates runtime
    failure, .metric_objects maps metric name -> the fake metric object.
    """
    rec = _FakeRagas()

    fake_ragas_mod = types.ModuleType("ragas")
    fake_metrics_mod = types.ModuleType("ragas.metrics")
    for name, obj in rec.metric_objects.items():
        setattr(fake_metrics_mod, name, obj)

    def fake_evaluate(dataset, metrics):
        rec.calls.append({"dataset": dataset, "metrics": list(metrics)})
        if rec.raise_on_evaluate is not None:
            raise rec.raise_on_evaluate
        names = [name for name, obj in rec.metric_objects.items() if obj in metrics]
        return {n: rec.scores[n] for n in names}

    fake_ragas_mod.evaluate = fake_evaluate
    fake_ragas_mod.metrics = fake_metrics_mod

    fake_datasets_mod = types.ModuleType("datasets")

    class _FakeDataset:
        def __init__(self, data):
            self.data = data

        @classmethod
        def from_dict(cls, data):
            return cls(data)

    fake_datasets_mod.Dataset = _FakeDataset

    monkeypatch.setitem(sys.modules, "ragas", fake_ragas_mod)
    monkeypatch.setitem(sys.modules, "ragas.metrics", fake_metrics_mod)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets_mod)
    return rec
