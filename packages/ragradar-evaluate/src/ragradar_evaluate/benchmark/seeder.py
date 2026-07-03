import random

from ragradar_core import store
from ragradar_core.schema import ChunkRecord, RunRecord, TokenBudget, Turn


def seed(pipeline: str, count: int = 20) -> int:
    """Generate synthetic run records as day-zero baseline.

    Half known-good, half known-bad. Seeded runs do NOT have RAGAS scores --
    they serve as input quality baseline only.
    """
    seeded_pipeline = f"{pipeline}__seeded"
    half = count // 2

    records = [_good_record(i) for i in range(half)]
    records += [_bad_record(i) for i in range(count - half)]

    session_id = store.get_or_create_session(seeded_pipeline)
    start_seq = store.next_run_seq(session_id)
    store.write_runs_batch(session_id, start_seq, records, seeded_pipeline)

    return count


def _good_record(idx: int) -> RunRecord:
    chunks = [
        ChunkRecord(
            chunk_id=f"seed_c{idx}_{j}",
            source_doc_id=f"seed_doc_{j % 3}",
            content=f"Synthetic high-quality chunk content for topic {idx}, variant {j}.",
            token_count=150,
            retrieval_score=0.85 + random.uniform(0, 0.1),
            rerank_score=0.90 + random.uniform(0, 0.08),
            retrieval_path="hybrid",
            truncated=False,
            cache_hit=True,
        )
        for j in range(4)
    ]
    return RunRecord(
        query=f"Synthetic good query {idx}: what is the best practice?",
        response=f"Synthetic good response {idx}: comprehensive answer.",
        chunks=chunks,
        token_budget=TokenBudget(
            total_limit=4096,
            chunks_allocated=2400,
            history_allocated=400,
            system_allocated=600,
            headroom=696,
        ),
        history_pre=[
            Turn(role="user", content="context question", tokens=5),
            Turn(role="assistant", content="context answer", tokens=10),
        ],
        history_post=[
            Turn(role="user", content="context question", tokens=5),
            Turn(role="assistant", content="context answer", tokens=10),
        ],
    )


def _bad_record(idx: int) -> RunRecord:
    chunks = [
        ChunkRecord(
            chunk_id=f"seed_bad_c{idx}_{j}",
            source_doc_id=f"seed_doc_{j % 8}",
            content=f"Low quality chunk {idx}_{j}.",
            token_count=200,
            retrieval_score=0.3 + random.uniform(0, 0.15),
            rerank_score=0.25 + random.uniform(0, 0.15),
            retrieval_path="bm25",
            truncated=(j % 2 == 0),
            cache_hit=False,
        )
        for j in range(6)
    ]
    return RunRecord(
        query=f"Synthetic bad query {idx}: vague unclear question?",
        response=f"Synthetic bad response {idx}: incomplete.",
        chunks=chunks,
        token_budget=TokenBudget(
            total_limit=4096,
            chunks_allocated=3800,
            history_allocated=100,
            system_allocated=100,
            headroom=96,
        ),
    )
