import json

import pytest
from ragradar_core import store as core_store
from ragradar_core.schema import (
    CacheEvent,
    ChunkRecord,
    RunRecord,
    TokenBudget,
    TokenUsage,
    Turn,
)


@pytest.fixture(autouse=True)
def ragradar_home(tmp_path, monkeypatch):
    ragradar_dir = tmp_path / ".ragradar"
    # ragradar_core.store._ragradar_dir is the one canonical home — every package
    # routes store access through it, so patching it here is sufficient.
    monkeypatch.setattr("ragradar_core.store._ragradar_dir", lambda: ragradar_dir)
    return tmp_path


def _full_record():
    return RunRecord(
        query="does RRF handle score scale differences",
        response="Yes, RRF normalizes scores via reciprocal rank.",
        chunks=[
            ChunkRecord(
                chunk_id="c1",
                source_doc_id="d1",
                content="RRF normalizes retrieval scores",
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
                content="Score fusion combines signals",
                token_count=30,
                retrieval_score=0.7,
                rerank_score=0.4,
                retrieval_path="ann",
                truncated=True,
                cache_hit=False,
            ),
        ],
        final_prompt="System: answer\nContext: ...\nQuery: does RRF handle score scale differences",
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
        history_post=[
            Turn(role="user", content="hello", tokens=3),
        ],
        eviction_reason="token_budget",
        cache_events=[
            CacheEvent(chunk_id="c1", hit=True, cache_source="disk"),
            CacheEvent(chunk_id="c2", hit=False),
        ],
        model="gpt-4",
        token_usage=TokenUsage(input_tokens=300, output_tokens=50, total_tokens=350),
    )


def insert_run(conn, session_id, run_seq, record, pipeline, created_at):
    """Insert a run row with an explicit created_at (write_run stamps now())."""
    conn.execute(
        "INSERT INTO runs (session_id, run_seq, query, pipeline, created_at, run_data) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (session_id, run_seq, record.query, pipeline, created_at, json.dumps(record.to_json())),
    )


@pytest.fixture
def populated_db(ragradar_home):
    # connect() creates the database at the latest schema version.
    conn = core_store.connect()

    rec_minimal = RunRecord(query="what is BM25", response="BM25 is a ranking function")
    rec_full = _full_record()
    rec_other = RunRecord(
        query="why does BM25 score differ from ANN score",
        response="Different scoring methods",
    )

    conn.execute(
        "INSERT INTO sessions (session_id, title, pipeline, created_at) "
        "VALUES (1, NULL, 'pipe_a', '2026-06-08T10:00:00+00:00')"
    )
    conn.execute(
        "INSERT INTO sessions (session_id, title, pipeline, created_at) "
        "VALUES (2, 'RRF investigation', 'pipe_a', '2026-06-09T10:00:00+00:00')"
    )
    insert_run(conn, 1, 1, rec_minimal, "pipe_a", "2026-06-08T10:05:00+00:00")
    insert_run(conn, 2, 1, rec_full, "pipe_a", "2026-06-09T10:05:00+00:00")
    insert_run(conn, 2, 2, rec_other, "pipe_a", "2026-06-09T10:10:00+00:00")
    conn.commit()
    conn.close()

    return core_store.db_path()


@pytest.fixture
def full_record():
    return _full_record()
