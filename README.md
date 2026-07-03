# ragradar

Local-first observability for RAG and agentic LLM pipelines.

Most RAG failures are silent. A pipeline can return duplicate chunks,
truncate its best content, evict critical history, or blow the token
budget — and still produce a plausible-looking response. There is no
standard tool that surfaces what actually went into the context window
and flags what went wrong.

`ragradar` fills that gap.

---

## The problem

```
Query → [Retrieve] → [Assemble] → CONTEXT WINDOW → [LLM] → Response
                                         ↑                        ↑
                               nothing here              everything here
```

Every existing evaluation tool fires after the LLM call — measuring
whether the response was good. None of them inspect the context window
before the expensive call. None of them flag mechanical failures:
duplicate chunks, high-score truncations, dropped history, token
misallocation.

`ragradar` is the observability layer that lives before the LLM call.

---

## Install

```bash
pip install ragradar
```

```python
import ragradar
```

That one import is the whole public surface — capturing runs,
evaluating them, and checking their health all live on the `ragradar`
module. Nothing else needs importing for everyday use.

(`ragradar-core`, `ragradar-capture`, and `ragradar-evaluate` are
separately installable too — see [Architecture](#architecture) for why
you'd reach for those directly instead.)

---

## Quick start

```bash
# install
uv sync

# run the examples — capture runs, then check/evaluate them
python examples/rag_pipeline/01_quickstart.py
python examples/rag_pipeline/02_capture_patterns.py
python examples/rag_pipeline/03_evaluate.py

# browse what was captured
ragradar list
ragradar list s4

# inspect the latest run — all seven factors
ragradar explain --full

# inspect a specific run
ragradar explain s4r3
ragradar explain s4r3 --full
ragradar explain s4r3 --html        # snapshot to ~/.ragradar/reports/

# search runs by query text
ragradar find "reranking"
ragradar find "RRF" --session s4

# compare two runs
ragradar diff s4r1 s4r3

# evaluate input quality — no LLM required
ragradar-evaluate run --input-only

# see benchmark thresholds built from accumulated runs
ragradar-evaluate benchmark show

# check a specific run against benchmark
ragradar-evaluate benchmark check s4r3

# see active quality policy
ragradar-evaluate policy show
```

---

## What ragradar explain shows

Seven analysis factors, computed deterministically from captured data.
Each factor is skipped silently if the required data was not captured.

```
Token usage       — per-section breakdown, headroom, model limit
Duplicate chunks  — path dups, window dups, semantic dups
Chunk scores      — retrieval + rerank score distribution
Truncation        — which chunks were trimmed, at what score
Dropped history   — what was evicted, why, what survived
Cache hits        — hit/miss ratio per chunk
Final prompt      — assembled prompt as-is
```

The example pipeline is designed to trigger all seven factors visibly:
low headroom (4.8%), window duplicates, one high-score truncation
(rerank 0.88, truncated=True), two evicted history turns, one cache hit.

---

## Instrumenting your own pipeline

Minimum instrumentation — two fields, and you get the run's id back:

```python
import ragradar

run_id = ragradar.capture("what is RRF?", "RRF fuses rankings.")
print(f"Captured {run_id} — try: ragradar explain {run_id}")
```

Everything past `query`/`response` takes plain Python — dicts, tuples,
a bare int — never a schema type you have to import and construct.
Full staged instrumentation, one call per pipeline stage:

```python
import ragradar

cap = ragradar.start(query="what is RRF?", pipeline="my_project")

cap.chunks([                                    # after retrieval — only
    {"content": "RRF combines rankings from multiple retrievers.",   # "content" is required
     "retrieval_score": 0.9, "rerank_score": 0.95},
])
cap.context(                                    # after assembly — a bare
    "System: answer using context.\nContext: ...\nQuery: what is RRF?",
    4096,                                        # int = total token limit; headroom is derived
)
cap.history(                                    # after history management
    pre=[{"user": "hello"}],
    post=[{"user": "hello"}],
    eviction_reason="token_budget",
)
run_id = cap.response(                          # after LLM call — auto-commits
    "RRF merges ranked lists into one.",
    token_usage={"input_tokens": 300, "output_tokens": 40},  # total derived
)
print(f"Captured {run_id}")
```

Every field except `query` and `response` is optional. More
instrumentation unlocks more analysis. Nothing breaks at any level —
capture failures are logged to `~/.ragradar/errors.log`, never raised
into your pipeline (flip that with `ragradar.set_strict(True)` in
development).

The schema dataclasses (`ChunkRecord`, `TokenBudget`, `Turn`, ...) still
exist and are re-exported from `ragradar` for callers who want static
typing or are round-tripping data already in that shape — pass one in
anywhere a dict is shown above and it's used as-is.

```bash
# greenfield — generates a scaffold with capture calls pre-positioned
pip install ragradar-capture   # capture alone, no analyst/eval deps
ragradar-capture init
```

---

## Architecture

```
your pipeline
  └── ragradar_capture (ragradar-capture)  →  ~/.ragradar/runs.db
                                     ↑
                    ragradar (analyst CLI) ┤
                    ragradar-evaluate      ┘
                          ↑
      all three depend on ragradar-core (schema + store + sNrN parser)
             all three are re-exported as one `import ragradar`
```

Four distributions, one public import. `ragradar-core`,
`ragradar-capture`, and `ragradar-evaluate` stay separately installable
because they have genuinely different footprints: `ragradar-capture` is
what runs *inside* your production pipeline and is deliberately
stdlib-only (beyond `ragradar-core`) so it never introduces a dependency
conflict there; `ragradar-evaluate` pulls `ragas` + `scipy` for scoring
and is not something a production hot path should have to import. The
`ragradar` distribution depends on all three and re-exports their public
functions from `ragradar/__init__.py`, so day-to-day use is just
`import ragradar` — the split only matters if you're choosing what to
install where.

```
ragradar/
  packages/
    ragradar-core/         # shared kernel — schema, SQLite store, zero deps
    ragradar-capture/      # instrumentation SDK — stdlib only
    ragradar/              # umbrella package (re-exports everything) + analyst CLI
    ragradar-evaluate/     # evaluation layer — ragas, scipy, sentence-transformers
  examples/
    rag_pipeline/     # end-to-end working example
  docs/
    internal/         # design doc, scope doc
```

---

## Tool reference

### ragradar (analyst CLI)

```bash
pip install ragradar

ragradar list                          # list sessions
ragradar list <session>                # list runs in session
ragradar find <hint>                   # search by query text
ragradar find <hint> --today           # with date filter
ragradar find <hint> --pipeline <name> # with pipeline filter
ragradar explain                       # latest run
ragradar explain <target>              # e.g. s2r3
ragradar explain <target> --full       # expanded
ragradar explain <target> --html       # HTML snapshot
ragradar diff <target> <target>        # compare two runs
ragradar budget <target>               # token waterfall only
ragradar session rename <id> <title>   # rename a session
```

Optional semantic search:

```bash
pip install ragradar[semantic]         # enables embedding-based search
```

### ragradar.check() / ragradar.evaluate()

Python API — two tasks, both reachable through the one `ragradar` import:

```python
import ragradar

run_id = ragradar.capture(
    "what is RRF?", "RRF fuses rankings.",
    chunks=[{"content": "RRF combines rankings.", "rerank_score": 0.9}],
)

# Is this run healthy? Free, deterministic, instant.
result = ragradar.check(run_id)
print(result.verdict, result.problems, result.thresholds)

# Score it fully — or pick exactly the metrics you want.
full = ragradar.evaluate(run_id)                                    # everything applicable
one = ragradar.evaluate(run_id, metrics=["duplicates"], save=False) # only this metric
print(one.metrics["duplicates"]["duplicate_ratio"])
```

CLI (the underlying `ragradar-evaluate` distribution's entry point):

```bash
ragradar-evaluate run <target>                    # both layers
ragradar-evaluate run <target> --input-only       # no LLM required
ragradar-evaluate run <target> --output-only      # RAGAS only
ragradar-evaluate run --session <id>              # all runs in session

ragradar-evaluate benchmark show                  # learned per-factor thresholds
ragradar-evaluate benchmark build                 # rebuild from 10+ evaluated runs
ragradar-evaluate benchmark check <target>        # ok / warn / fail per factor
ragradar-evaluate benchmark export                # RAGAS-compatible JSONL

ragradar-evaluate policy show                     # active thresholds
ragradar-evaluate policy set <field> <value>      # override a threshold
ragradar-evaluate policy reset                    # restore defaults
```

---

## Evaluation metrics — atomic, individually selectable

Every metric is selectable on its own via
`ragradar.evaluate(run_id, metrics=[...])`; `metrics=None` runs
everything applicable. Discover them with `ragradar.available_metrics()`.

**Input metrics (free, deterministic, no LLM)**

```
relevance             chunk similarity vs query (uses existing rerank scores)
duplicates            path + window + semantic duplicate detection
truncation            were high-score chunks cut?
token_efficiency      headroom, low-score chunk ratio
coherence             source domain count, score variance
```

**Output metrics (RAGAS, LLM-as-judge — costs LLM calls)**

```
faithfulness          is the response grounded in retrieved context?
answer_relevancy      does the response address the query?
context_precision     how much retrieved content was actually used?
context_recall        was the necessary information present? (needs ground_truth)
```

**Current standards**

`ragradar.check()` compares runs against learned thresholds once 10+
evaluated runs exist for the pipeline (built automatically from your own
history), falling back to policy defaults before that —
`CheckResult.thresholds` says which was used. The system becomes
pipeline-specific over time — your data, your thresholds.

---

## Roadmap

```
v0.1.0   ragradar-capture + ragradar         ✓ shipped
v0.2.0   ragradar-evaluate + examples        ✓ shipped
v0.3.0   ragradar-improve (input quality     — planned
         improvement before LLM call)
```

---

## Development

```bash
git clone <repo>
cd ragradar
uv sync

# run all tests
uv run pytest

# run per package
uv run pytest packages/ragradar-core/tests/
uv run pytest packages/ragradar-capture/tests/
uv run pytest packages/ragradar/tests/
uv run pytest packages/ragradar-evaluate/tests/

# format + lint
uv run ruff format packages examples
uv run ruff check packages examples
```

290 tests across four packages. All pass.

---

## Why ragradar

| | ragradar | LangSmith | RAGAS | Print debugging |
|---|---|---|---|---|
| Pre-call inspection | ✓ | ✗ | ✗ | manual |
| Local / offline | ✓ | ✗ | partial | ✓ |
| Zero infrastructure | ✓ | ✗ | ✓ | ✓ |
| Pipeline agnostic | ✓ | partial | ✓ | ✓ |
| Persistent run store | ✓ | ✓ | ✗ | ✗ |
| Benchmark over time | ✓ | ✓ | ✗ | ✗ |
| No LLM for navigation | ✓ | ✓ | ✗ | ✓ |

ragradar is not a replacement for LangSmith or RAGAS. It occupies the
pre-call mechanical observability position that neither covers.
The three tools compose: ragradar captures, RAGAS scores, LangSmith traces.
