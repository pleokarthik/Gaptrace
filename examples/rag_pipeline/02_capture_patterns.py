"""
gaptrace-capture patterns beyond the quickstart. Each pattern_*() function is
runnable independently (`python -c "import importlib; ..."` or via a
REPL) or all together via __main__ below.

Everything here is plain Python — dicts, tuples, ints. The schema
dataclasses exist for strict typing but are never needed for capture.
"""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import gaptrace

PIPELINE = "rag_example"


def _sample_chunks():
    """Four chunks engineered to trigger gaptrace explain's window-dup, truncation, and low-score signals."""
    return [
        {
            "chunk_id": "rrf_norm_1",
            "source_doc_id": "rrf_paper_2024",
            "content": "Reciprocal Rank Fusion normalizes scores from different retrieval systems",
            "token_count": 180,
            "retrieval_score": 0.85,
            "rerank_score": 0.92,
            "retrieval_path": "hybrid",
            "cache_hit": True,
        },
        {
            "chunk_id": "rrf_norm_2",
            "source_doc_id": "rrf_paper_2024",
            "content": (
                "Reciprocal Rank Fusion normalizes scores from different "
                "retrieval systems and ranks documents accordingly."
            ),
            "token_count": 160,
            "retrieval_score": 0.71,
            "rerank_score": 0.78,
            "retrieval_path": "bm25",
            "cache_hit": False,
        },
        {
            "chunk_id": "bm25_tf_idf",
            "source_doc_id": "ir_textbook_ch3",
            "content": "BM25 computes relevance using term frequency and inverse document frequency.",
            "token_count": 140,
            "retrieval_score": 0.82,
            "rerank_score": 0.88,
            "retrieval_path": "bm25",
            "truncated": True,
            "cache_hit": False,
        },
        {
            "chunk_id": "ctx_window",
            "source_doc_id": "rag_patterns",
            "content": "Context window management determines which chunks survive token budget constraints.",
            "token_count": 145,
            "retrieval_score": 0.48,
            "rerank_score": 0.39,
            "retrieval_path": "bm25",
            "cache_hit": False,
        },
    ]


def pattern_full_fields():
    """Populate every optional RunRecord field -- metadata filter, chunks, context, history, cache, tool calls, response -- in one staged capture."""
    cap = gaptrace.start(query="what is RRF and how does it normalize scores?", pipeline=PIPELINE)

    # Metadata filter runs before retrieval; excluded candidates never reach scoring.
    cap.metadata_filter(
        applied=True,
        candidate_count=6,
        excluded_count=2,
        filters={"source": "internal"},
    )

    chunks = _sample_chunks()
    # requested_count=6: the retriever asked for 6 candidates but only 4
    # came back -- triggers gaptrace explain's candidate-underfill signal.
    cap.chunks(chunks, requested_count=6)

    prompt = (
        "System: answer using context.\n\nContext:\n"
        + "\n".join(f"[{i}] {c['content']}" for i, c in enumerate(chunks, 1))
        + "\n\nQuery: what is RRF?"
    )
    # Budget as a plain dict; headroom is derived from the allocations.
    cap.context(
        prompt,
        {
            "total_limit": 4096,
            "chunks_allocated": 2800,
            "history_allocated": 600,
            "system_allocated": 500,
        },
    )

    # History turns as {"role": "content"} shorthand; token counts are estimated.
    cap.history(
        pre=[
            {"user": "Can you help me understand retrieval systems?"},
            {"assistant": "Of course!"},
            {"user": "Start with BM25."},
            {"assistant": "BM25 ranks by term frequency."},
        ],
        post=[
            {"user": "Can you help me understand retrieval systems?"},
            {"assistant": "Of course!"},
        ],
        eviction_reason="token_budget",
    )

    # Cache events as one {chunk_id: hit} mapping.
    cap.cache({c["chunk_id"]: bool(c["cache_hit"]) for c in chunks})

    cap.tool_call(
        {
            "tool_name": "rerank",
            "arguments": {"chunk_ids": [c["chunk_id"] for c in chunks]},
            "result": "reranked 4 chunks",
            "latency_ms": 42.0,
        }
    )

    run_id = cap.response(
        "RRF replaces raw retrieval scores with rank-based reciprocal values, "
        "making it robust to score-scale differences across retrievers.",
        token_usage={"input_tokens": 1850, "output_tokens": 40},  # total derived
        model="gpt-4-turbo",
    )
    # cap.commit() already called by cap.response()
    print(f"Captured {run_id} — try: gaptrace explain {run_id}")


def _backdate_pipeline_runs(db_path: Path, pipeline: str, minutes: int) -> None:
    """test/demo-only: rewrites timestamps directly via raw SQL to simulate
    an idle gap. NOT part of the public gaptrace-capture API -- real pipelines
    never touch runs.db directly; session gaps happen naturally over
    wall-clock time between calls to gaptrace.start().
    """
    with sqlite3.connect(str(db_path)) as conn:
        for table, key_cols in [("sessions", ["session_id"]), ("runs", ["session_id", "run_seq"])]:
            rows = conn.execute(
                f"SELECT {', '.join(key_cols)}, created_at FROM {table} WHERE pipeline = ?",
                (pipeline,),
            ).fetchall()
            for row in rows:
                keys, old_ts = row[:-1], datetime.fromisoformat(row[-1])
                new_ts = (old_ts - timedelta(minutes=minutes)).isoformat()
                where = " AND ".join(f"{c} = ?" for c in key_cols)
                conn.execute(f"UPDATE {table} SET created_at = ? WHERE {where}", (new_ts, *keys))


def pattern_multi_session_gap():
    """Capture two query groups 31 minutes apart to trigger gaptrace-capture's auto session split."""
    for q in ["what is RRF?", "why does BM25 differ from vector similarity?"]:
        cap = gaptrace.start(query=q, pipeline=PIPELINE)
        cap.chunks(_sample_chunks())
        cap.response(f"Answer to: {q}")

    db_path = Path.home() / ".gaptrace" / "runs.db"
    _backdate_pipeline_runs(db_path, PIPELINE, minutes=31)

    for q in ["what does a cross-encoder compute?", "when should reranking be skipped?"]:
        cap = gaptrace.start(query=q, pipeline=PIPELINE)
        cap.chunks(_sample_chunks())
        cap.response(f"Answer to: {q}")


def pattern_thread_local_proxy():
    """Capture via module-level gaptrace.chunks()/response() -- no capture object threaded through the call stack."""
    gaptrace.start(query="does rerank order affect final context assembly?", pipeline="proxy_demo")
    gaptrace.chunks(
        [
            {
                "content": "Rerank order changes which chunks survive truncation.",
                "retrieval_score": 0.8,
                "rerank_score": 0.83,
            },
        ]
    )
    run_id = gaptrace.response(
        "Yes -- rerank order determines what gets truncated when the budget is tight."
    )
    print(f"Captured {run_id} — try: gaptrace explain {run_id}")


if __name__ == "__main__":
    # pattern_full_fields() runs last so it ends up as the latest run --
    # `gaptrace explain` with no target shows the most recently captured run,
    # and this is the one that lights up every analysis factor.
    pattern_multi_session_gap()
    pattern_thread_local_proxy()
    pattern_full_fields()
    print("Captured capture-pattern demo runs. Try: gaptrace list")
