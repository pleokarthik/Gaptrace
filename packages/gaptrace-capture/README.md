# gaptrace-capture

Developer-side SDK that instruments a RAG or agentic pipeline at key
stages and writes structured run records to a local SQLite store
(`~/.gaptrace/runs.db`). Capturing is the verb — the stored data are *runs*,
addressed as `sNrN` ids (e.g. `s2r3`).

```
pip install gaptrace-capture
```

Stdlib-only at runtime (plus the tiny `gaptrace-core` kernel it shares with
the other gaptrace packages).

## The one-liner

```python
import gaptrace_capture

run_id = gaptrace_capture.capture("what is RRF?", "RRF fuses rankings.")
print(run_id)   # "s1r1"
```

Two fields are enough. Everything else is optional keyword-only:
`chunks`, `final_prompt`, `token_budget`, `history_pre`, `history_post`,
`eviction_reason`, `cache_events`, `tool_calls`, `model`, `token_usage`,
`pipeline`. A misspelled keyword fails immediately with `TypeError` —
the signature is explicit, not `**kwargs`.

## The staged pattern

`start()` returns a `Capture` — the action object for one pipeline run.
Every argument below takes plain Python — dicts, tuples, a bare int —
never a schema type you have to import:

```python
import gaptrace_capture

cap = gaptrace_capture.start(query="what is RRF?", pipeline="my_project")

cap.chunks([                                    # retrieval stage — only
    {"content": "RRF combines rankings from multiple retrievers.",   # "content" is required
     "retrieval_score": 0.9, "rerank_score": 0.95},
])
cap.context(                                    # assembly stage — a bare
    "System: answer using context.\nContext: ...\nQuery: what is RRF?",
    4096,                                        # int = total token limit; headroom is derived
)
cap.history(                                    # history management stage
    pre=[{"user": "hello"}],
    post=[{"user": "hello"}],
    eviction_reason="token_budget",
)
cap.cache({"c1": True})                            # any time pre-commit
cap.tool_call({"tool_name": "rerank",              # appends, once per call
               "arguments": {"chunk_ids": ["c1"]}})
run_id = cap.response(                          # LLM output stage — auto-commits
    "RRF merges ranked lists into one.",
    token_usage={"input_tokens": 300, "output_tokens": 40},  # total derived
)
print(run_id)
```

The schema dataclasses (`ChunkRecord`, `TokenBudget`, `Turn`, `CacheEvent`,
`TokenUsage`, `ToolCallRecord` — tables below) still exist for callers who
want static typing or already have data in that shape; pass one in
anywhere a dict is shown above and it's used as-is.

`cap.response()` auto-commits and returns the run id; `cap.commit()` is
only needed if you skip `response()`. Commit is idempotent — a second
call returns the same id without writing again.

### The returned run id

Every committed capture hands back the run's `sNrN` id — also available
as `cap.run_id` (which is `None` before commit). Feed it straight to the
other tools:

```
gaptrace explain s2r3          # analyze the run
gaptrace-evaluate run s2r3     # score it
```

If an internal failure was swallowed (see below), the id is `None`
instead — check for that before passing it on.

### Thread-local proxies

After `gaptrace_capture.start()`, module-level functions
(`gaptrace_capture.chunks()`, `.context()`, `.history()`, `.cache()`,
`.tool_call()`, `.response()`, `.commit()`) route to the same capture
from anywhere on that thread — no need to pass the `Capture` object
through every function signature. With no active capture they log an
error and no-op.

## Dataclasses

These are the advanced/typed path — every capture call above accepts
plain dicts (with sensible defaults filled in — e.g. `chunks` only
needs `content`) or shorthand forms (`{"user": "..."}` turns, a bare int
budget), not these types directly. Reach for them only if you want
static typing or are round-tripping data already in this shape. All of
them tolerate unknown keyword arguments (silently dropped), so adding
fields later never breaks existing instrumentation.

### RunRecord

| Field | Type | Notes |
|---|---|---|
| `query` | `str` | required |
| `response` | `str` | required |
| `chunks` | `list[ChunkRecord]` | optional |
| `final_prompt` | `str` | optional |
| `token_budget` | `TokenBudget` | optional |
| `history_pre` / `history_post` | `list[Turn]` | optional |
| `eviction_reason` | `str` | optional |
| `cache_events` | `list[CacheEvent]` | optional |
| `tool_calls` | `list[ToolCallRecord]` | optional |
| `model` | `str` | optional |
| `token_usage` | `TokenUsage` | optional |

### ChunkRecord

| Field | Type | Notes |
|---|---|---|
| `chunk_id` | `str` | required |
| `source_doc_id` | `str` | required |
| `content` | `str` | required |
| `token_count` | `int` | required |
| `retrieval_score` | `float` | optional |
| `rerank_score` | `float` | optional |
| `retrieval_path` | `str` | optional — e.g. `"bm25"`, `"ann"`, `"hybrid"` |
| `truncated` | `bool` | default `False` |
| `cache_hit` | `bool` | optional |

### TokenBudget

| Field | Type |
|---|---|
| `total_limit` | `int` |
| `chunks_allocated` | `int` |
| `history_allocated` | `int` |
| `system_allocated` | `int` |
| `headroom` | `int` |

### TokenUsage

| Field | Type |
|---|---|
| `input_tokens` | `int` |
| `output_tokens` | `int` |
| `total_tokens` | `int` |

### Turn

| Field | Type | Notes |
|---|---|---|
| `role` | `str` | e.g. `"user"`, `"assistant"` |
| `content` | `str` | |
| `tokens` | `int` | optional |

### CacheEvent

| Field | Type | Notes |
|---|---|---|
| `chunk_id` | `str` | |
| `hit` | `bool` | |
| `cache_source` | `str` | optional — e.g. `"disk"` |

### ToolCallRecord

| Field | Type | Notes |
|---|---|---|
| `tool_name` | `str` | |
| `arguments` | `dict` | |
| `result` | `str` | optional |
| `error` | `str` | optional |
| `latency_ms` | `float` | optional |

## The never-raise philosophy (and strict mode)

In production, instrumentation must never take down the pipeline it
observes. Every capture call is wrapped internally: conversion errors,
store failures — all swallowed, logged to `~/.gaptrace/errors.log`, and the
call returns (`None` where a run id was expected). Your pipeline never
sees an exception from gaptrace-capture.

During development you usually want the opposite. Strict mode makes
those same errors raise:

```python
import gaptrace_capture

gaptrace_capture.set_strict(True)          # in code
# or, without a code change:
# GAPTRACE_CAPTURE_STRICT=1 python my_pipeline.py
```

The only error that raises regardless of mode is a misspelled keyword
to `capture()` — that's a bug at the call site, caught by Python itself
as `TypeError`.

## Scaffold CLI

```
gaptrace-capture init
```

Generates a starter `gaptrace_pipeline.py` in the current directory with
capture calls pre-positioned at the right pipeline stages (refuses to
overwrite an existing one).

## Store

First capture creates `~/.gaptrace/runs.db` (schema managed by `gaptrace-core`;
migrations are automatic). Sessions group runs automatically on a
30-minute idle gap — no developer action needed. Browse with `gaptrace`,
score with `gaptrace-evaluate`.
