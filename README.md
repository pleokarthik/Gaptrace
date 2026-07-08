# gaptrace

Local-first observability for RAG and agentic LLM pipelines.

Most RAG failures are silent. A pipeline can return duplicate chunks,
truncate its best content, evict critical history, or blow the token
budget — and still produce a plausible-looking response. There is no
standard tool that surfaces what actually went into the context window
and flags what went wrong.

`gaptrace` fills that gap.

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

`gaptrace` is the observability layer that lives before the LLM call.

---

## Install

```bash
pip install gaptrace
```

```python
import gaptrace
```

That one import is the whole public surface — capturing runs,
evaluating them, and checking their health all live on the `gaptrace`
module. Nothing else needs importing for everyday use.

(`gaptrace-core`, `gaptrace-capture`, and `gaptrace-evaluate` are
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
gaptrace list
gaptrace list s4

# inspect the latest run
gaptrace explain --full

# inspect a specific run
gaptrace explain s4r3
gaptrace explain s4r3 --full
gaptrace explain s4r3 --html        # snapshot to ~/.gaptrace/reports/

# search runs by query text
gaptrace find "reranking"
gaptrace find "RRF" --session s4

# compare two runs
gaptrace diff s4r1 s4r3

# evaluate input quality — no LLM required
gaptrace-evaluate run --input-only

# see benchmark thresholds built from accumulated runs
gaptrace-evaluate benchmark show

# check a specific run against benchmark
gaptrace-evaluate benchmark check s4r3

# see active quality policy
gaptrace-evaluate policy show
```

---

## What gaptrace explain shows

Eleven analysis factors, computed deterministically from captured data
(plus the assembled prompt, rendered as-is rather than analyzed). Each
factor is skipped silently if the required data was not captured.

```
Token usage         — per-section breakdown, headroom, model limit
Duplicate chunks    — path dups, window dups, semantic dups
Chunk scores        — retrieval + rerank score distribution
Truncation          — which chunks were trimmed, at what score
Score degeneracy    — chunk-score variance; near-zero means scores aren't discriminating
Score margin        — top-vs-second chunk score gap; thin means the top pick isn't decisively best
Dropped history     — what was evicted, why, what survived
Cache hits          — hit/miss ratio per chunk
Cache behavior      — semantic-cache hit/miss, borderline similarity, stale-hit age
Metadata filter     — candidates excluded before scoring, exclusion ratio
Candidate underfill — returned chunk count vs the requested top_k ask
Final prompt        — assembled prompt as-is
```

Both `examples/rag_pipeline/02_capture_patterns.py`'s `pattern_full_fields()`
run and `03_evaluate.py`'s `capture_demo_run()` are designed to trigger ten
of these eleven factors visibly: low headroom (4.8%), window duplicates,
one high-score truncation (rerank 0.88, truncated=True), two evicted
history turns, one cache hit, two candidates excluded by a metadata
filter (33% exclusion), a thin score margin (0.04, just under the 0.05
policy default), plus chunk-score variance and chunk scores rendering
alongside them. Candidate underfill fires on both too — flagged on 02's
run (6 requested, 4 returned) and a clean exact match on 03's (4
requested, 4 returned), deliberately contrasting the two. Because
`03_evaluate.py` now captures the same breadth of fields as `02`'s demo
(it used to be much thinner), following the Quick start commands above
in order leaves the reader's actual latest run — `gaptrace explain --full`,
no target needed, currently `s4r4` — firing all ten directly rather than
needing `s4r3` specifically. Only "Cache behavior" is absent from both:
no example script calls `cap.semantic_cache()`.

---

## Instrumenting your own pipeline

Minimum instrumentation — two fields, and you get the run's id back:

```python
import gaptrace

run_id = gaptrace.capture("what is RRF?", "RRF fuses rankings.")
print(f"Captured {run_id} — try: gaptrace explain {run_id}")
```

Everything past `query`/`response` takes plain Python — dicts, tuples,
a bare int — never a schema type you have to import and construct.
Full staged instrumentation, one call per pipeline stage:

```python
import gaptrace

cap = gaptrace.start(query="what is RRF?", pipeline="my_project")

cap.metadata_filter(                            # before retrieval — candidates
    applied=True,                               # excluded by a metadata filter
    candidate_count=12,
    excluded_count=3,
    filters={"source": "internal"},
)
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
capture failures are logged to `~/.gaptrace/errors.log`, never raised
into your pipeline (flip that with `gaptrace.set_strict(True)` in
development).

The schema dataclasses (`ChunkRecord`, `TokenBudget`, `Turn`, ...) still
exist and are re-exported from `gaptrace` for callers who want static
typing or are round-tripping data already in that shape — pass one in
anywhere a dict is shown above and it's used as-is.

```bash
# greenfield — generates a scaffold with capture calls pre-positioned
pip install gaptrace-capture   # capture alone, no analyst/eval deps
gaptrace-capture init
```

---

## Architecture

```
your pipeline
  └── gaptrace_capture (gaptrace-capture)  →  ~/.gaptrace/runs.db
                                     ↑
                    gaptrace (analyst CLI) ┤
                    gaptrace-evaluate      ┘
                          ↑
      all three depend on gaptrace-core (schema + store + sNrN parser)
             all three are re-exported as one `import gaptrace`
```

Four distributions, one public import. `gaptrace-core`,
`gaptrace-capture`, and `gaptrace-evaluate` stay separately installable
because they have genuinely different footprints: `gaptrace-capture` is
what runs *inside* your production pipeline and is deliberately
stdlib-only (beyond `gaptrace-core`) so it never introduces a dependency
conflict there; `gaptrace-evaluate` pulls `ragas` + `scipy` for scoring
and is not something a production hot path should have to import. The
`gaptrace` distribution depends on all three and re-exports their public
functions from `gaptrace/__init__.py`, so day-to-day use is just
`import gaptrace` — the split only matters if you're choosing what to
install where.

```
gaptrace/
  packages/
    gaptrace-core/         # shared kernel — schema, SQLite store, zero deps
    gaptrace-capture/      # instrumentation SDK — stdlib only
    gaptrace/              # umbrella package (re-exports everything) + analyst CLI
    gaptrace-evaluate/     # evaluation layer — ragas, scipy, sentence-transformers
  examples/
    rag_pipeline/     # end-to-end working examples
  docs/
    ARCHITECTURE.md, design-doc.md, scope.md, execution-flow.md
```

---

## Tool reference

### gaptrace (analyst CLI)

```bash
pip install gaptrace

gaptrace list                          # list sessions
gaptrace list <session>                # list runs in session
gaptrace find <hint>                   # search by query text
gaptrace find <hint> --today           # with date filter
gaptrace find <hint> --pipeline <name> # with pipeline filter
gaptrace explain                       # latest run
gaptrace explain <target>              # e.g. s2r3
gaptrace explain <target> --full       # expanded
gaptrace explain <target> --html       # HTML snapshot
gaptrace diff <target> <target>        # compare two runs
gaptrace budget <target>               # token waterfall only
gaptrace session rename <id> <title>   # rename a session
```

Optional semantic search:

```bash
pip install gaptrace[semantic]         # enables embedding-based search
```

### gaptrace.check() / gaptrace.evaluate()

Python API — two tasks, both reachable through the one `gaptrace` import:

```python
import gaptrace

run_id = gaptrace.capture(
    "what is RRF?", "RRF fuses rankings.",
    chunks=[{"content": "RRF combines rankings.", "rerank_score": 0.9}],
)

# Is this run healthy? Free, deterministic, instant.
result = gaptrace.check(run_id)
print(result.verdict, result.problems, result.thresholds)

# Score it fully — or pick exactly the metrics you want.
full = gaptrace.evaluate(run_id)                                    # everything applicable
one = gaptrace.evaluate(run_id, metrics=["duplicates"], save=False) # only this metric
print(one.metrics["duplicates"]["duplicate_ratio"])
```

CLI (the underlying `gaptrace-evaluate` distribution's entry point):

```bash
gaptrace-evaluate run <target>                    # both layers
gaptrace-evaluate run <target> --input-only       # no LLM required
gaptrace-evaluate run <target> --output-only      # RAGAS only
gaptrace-evaluate run --session <id>              # all runs in session

gaptrace-evaluate benchmark show                  # learned per-factor thresholds
gaptrace-evaluate benchmark build                 # rebuild from 10+ evaluated runs
gaptrace-evaluate benchmark check <target>        # ok / warn / fail per factor
gaptrace-evaluate benchmark export                # RAGAS-compatible JSONL

gaptrace-evaluate policy show                     # active thresholds
gaptrace-evaluate policy set <field> <value>      # override a threshold
gaptrace-evaluate policy reset                    # restore defaults
```

---

## Evaluation metrics — atomic, individually selectable

Every metric is selectable on its own via
`gaptrace.evaluate(run_id, metrics=[...])`; `metrics=None` runs
everything applicable. Discover them with `gaptrace.available_metrics()`
(fourteen today: ten input, four output).

**Input metrics (free, deterministic, no LLM)**

```
relevance             chunk similarity vs query (uses existing rerank scores)
duplicates            path + window + semantic duplicate detection
truncation            were high-score chunks cut?
token_efficiency      headroom, low-score chunk ratio
coherence             source domain count, score variance
cache_risk            borderline/stale semantic-cache hits
filter_risk           candidate-exclusion ratio from metadata filtering
score_degeneracy      chunk-score variance; near-zero means scores aren't discriminating
score_margin          top-vs-second chunk score gap (plus a threshold-margin diagnostic)
score_underfill       returned chunk count vs the requested top_k ask
```

**Output metrics (RAGAS, LLM-as-judge — costs LLM calls)**

```
faithfulness          is the response grounded in retrieved context?
answer_relevancy      does the response address the query?
context_precision     how much retrieved content was actually used?
context_recall        was the necessary information present? (needs ground_truth)
```

**Current standards**

`gaptrace.check()` compares runs against learned thresholds once 10+
evaluated runs exist for the pipeline (built automatically from your own
history), falling back to policy defaults before that —
`CheckResult.thresholds` says which was used. The system becomes
pipeline-specific over time — your data, your thresholds.

---

## Roadmap

```
v0.1.0   gaptrace-capture + gaptrace         ✓ shipped
v0.2.0   gaptrace-evaluate + examples        ✓ shipped
v0.3.0   gaptrace-improve (input quality     — planned
         improvement before LLM call)
```

---

## Development

```bash
git clone <repo>
cd gaptrace
uv sync

# run all tests
uv run pytest

# run per package
uv run pytest packages/gaptrace-core/tests/
uv run pytest packages/gaptrace-capture/tests/
uv run pytest packages/gaptrace/tests/
uv run pytest packages/gaptrace-evaluate/tests/

# format + lint
uv run ruff format packages examples
uv run ruff check packages examples
```

413 tests across four packages. All pass.

---

## Why gaptrace

| | gaptrace | LangSmith | RAGAS | Print debugging |
|---|---|---|---|---|
| Pre-call inspection | ✓ | ✗ | ✗ | manual |
| Local / offline | ✓ | ✗ | partial | ✓ |
| Zero infrastructure | ✓ | ✗ | ✓ | ✓ |
| Pipeline agnostic | ✓ | partial | ✓ | ✓ |
| Persistent run store | ✓ | ✓ | ✗ | ✗ |
| Benchmark over time | ✓ | ✓ | ✗ | ✗ |
| No LLM for navigation | ✓ | ✓ | ✗ | ✓ |

gaptrace is not a replacement for LangSmith or RAGAS. It occupies the
pre-call mechanical observability position that neither covers.
The three tools compose: gaptrace captures, RAGAS scores, LangSmith traces.
