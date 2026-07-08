# RAG Pipeline Example

Three small scripts demonstrating gaptrace-capture, gaptrace, and gaptrace-evaluate
end to end, using fake retrieval data (no external services).

All three scripts use only `import gaptrace` — no submodule imports, no
schema-type construction. Every capture argument below the surface is a
plain dict, tuple, or int; gaptrace coerces it.

| File | Demonstrates |
|---|---|
| `01_quickstart.py` | The whole capture API surface in under 30 lines: the `gaptrace.capture()` one-liner (returns the run id) and the staged `gaptrace.start()` → `cap.chunks()` → `cap.response()` pattern. |
| `02_capture_patterns.py` | Three named patterns: `pattern_full_fields()` (every optional `RunRecord` field populated), `pattern_multi_session_gap()` (auto session-splitting after a 30-minute idle gap), `pattern_thread_local_proxy()` (`gaptrace.chunks()`/`gaptrace.response()` without threading a capture object through the call stack). |
| `03_evaluate.py` | The two evaluation tasks: `gaptrace.check(run_id)` ("is this run healthy?" — free, instant, no LLM) and `gaptrace.evaluate(run_id)` (complete or atomic-metric scoring), plus `gaptrace.available_metrics()` discovery. |

## Quick start

Install from workspace root:

```bash
cd <repo root>
uv sync
```

Run the three scripts in order:

```bash
cd examples/rag_pipeline
python 01_quickstart.py
python 02_capture_patterns.py
python 03_evaluate.py
```

`01_quickstart.py` and `02_capture_patterns.py` only capture runs — the
last run captured (`pattern_full_fields()`, run last on purpose) is the
one engineered to trigger every `gaptrace explain` analysis factor.
`03_evaluate.py` captures one demonstration run of its own and walks it
through `check()` and `evaluate()` — it runs standalone, but running 02
first gives you more runs to browse.

## Browse runs with gaptrace

```bash
gaptrace list
gaptrace list s4               # session numbers will vary run to run
gaptrace explain                # latest run — all seven factors
gaptrace explain s4r3 --full
gaptrace explain s4r3 --html
gaptrace find "reranking"
gaptrace diff s4r1 s4r3
gaptrace budget s4r3
gaptrace session rename s4 "Retrieval mechanics"
```

Session/run numbers depend on what else has run against your local
`~/.gaptrace/runs.db` — use `gaptrace list` to see the actual IDs on your machine.

## Evaluate with gaptrace-evaluate

```bash
python 03_evaluate.py
```

Or use the CLI directly:

```bash
gaptrace-evaluate run --input-only
gaptrace-evaluate policy show
gaptrace-evaluate benchmark show
gaptrace-evaluate benchmark check s4r3
```

## What pattern_full_fields()'s run shows in gaptrace explain

`pattern_full_fields()` in `02_capture_patterns.py` is engineered to
trigger every factor:

| Factor | What you'll see |
|---|---|
| Token usage | Headroom 196/4096 — budget is tight by design |
| Duplicates | Window dup between `rrf_norm_1` and `rrf_norm_2`, which share the `rrf_paper_2024` source |
| Chunk scores | Distribution across 4 chunks (0.39–0.92 rerank) |
| Truncation | Severity: high — `bm25_tf_idf` truncated with rerank 0.88 |
| Dropped history | 4 → 2 turns, 2 dropped, reason: `token_budget` |
| Cache hits | 1/4 hit (`rrf_norm_1`) |
| Final prompt | Assembled system + context + history + query |

## Clean slate

To start fresh, remove the local store:

```bash
rm -rf ~/.gaptrace
```
