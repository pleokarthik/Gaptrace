"""
ragradar-capture quickstart -- the whole capture API surface, fast.

Run this, then: ragradar list && ragradar explain
"""

import ragradar

# One-liner: capture query + response, nothing else.
run_id = ragradar.capture("what is 2+2?", "4")
print(f"Captured {run_id} — try: ragradar explain {run_id}")

# Staged: start a capture, feed it stages as they happen, then respond.
# Chunks are plain dicts — only "content" is required; ids and token
# counts are filled in for you.
cap = ragradar.start(query="what is RRF?", pipeline="quickstart")

cap.chunks(
    [
        {
            "content": "Reciprocal Rank Fusion combines rankings from multiple retrievers.",
            "retrieval_score": 0.9,
            "rerank_score": 0.95,
        },
    ]
)

run_id = cap.response("RRF combines rankings from multiple retrievers into one ranked list.")
# cap.commit() already called automatically by cap.response()
print(f"Captured {run_id} — try: ragradar explain {run_id}")
