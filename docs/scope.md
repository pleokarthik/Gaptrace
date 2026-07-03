# ragradar — Scope Document

**Status:** Draft v0.1 — original scope document, retained for delivery history.
Tools 1 and 2 (ragradar-capture, ragradar) and Tool 3 (ragradar-evaluate) are
delivered; Tool 4 (ragradar-improve) is still deferred. Store/schema
ownership described below predates the later `ragradar-core` extraction —
see `docs/ARCHITECTURE.md` and `design-doc.md` for the current package
picture and the `Capture` rename.  
**Author:** Leo Karthik Paramasivan  
**Date:** 2026-06-26

---

## Tool 1 + Tool 2 — `ragradar-capture` + `ragradar`

Delivered together. Single release. Shared store contract.

---

### ragradar-capture

**What it is**

Developer-side SDK. Instruments a RAG or agentic pipeline at key stages. Writes structured run records to a local SQLite store. Invisible on failure.

**What it is not**

Not an analysis tool. Not a CLI. Not opinionated about the pipeline stack.

---

**Delivery scope**

**Core capture API**

```python
ragradar.start(query, pipeline)          # begins a run, creates session if needed
cap.chunks(chunks)                       # retrieval stage
cap.context(prompt, budget)              # assembly stage
cap.history(pre, post, eviction_reason)  # history management stage
cap.response(response, usage)            # LLM output stage
cap.cache(events)                        # cache events
cap.commit()                             # writes to store
```

Thread-local active capture — `ragradar.*` accessible across files without passing the `Capture` object.

Auto-commit on `cap.response()` if `cap.commit()` not called explicitly.

**Single-line fallback**

```python
ragradar.capture(query, response)    # minimum viable — two fields only
```

**Scaffold generator**

```bash
ragradar-capture init
```

Generates a starter `ctx_pipeline.py` with capture calls pre-positioned at correct pipeline stages. For Dev 1 — greenfield only.

**Session management**

Auto-session grouping on 30-minute idle gap. New session created automatically. No developer action required.

**Store initialisation**

`~/.ragradar/runs.db` created on first capture. Schema migrations handled internally. `meta` table holds schema version.

**Failure contract**

All capture calls wrapped in try/except internally. Failures logged to `~/.ragradar/errors.log`. Pipeline never interrupted under any circumstance.

---

**Schema — owned by ragradar-capture at delivery time** (later centralized into the `ragradar-core` kernel; see `ARCHITECTURE.md`)

```sql
meta(key, value)
sessions(session_id, title, pipeline, created_at)
runs(session_id, run_seq, query, pipeline, created_at, run_data JSON)
```

Indexes on `created_at`, `query`, `pipeline`.

---

**Data types**

```
RunRecord       — top-level run container
ChunkRecord     — per-chunk retrieval data
TokenBudget     — assembly budget breakdown
TokenUsage      — LLM token consumption
Turn            — single history turn
CacheEvent      — per-chunk cache hit/miss
```

All fields except `query` and `response` optional. `**kwargs` on capture methods — future fields never break existing instrumentation.

---

**Out of scope**

- No analysis
- No rendering
- No search
- No evaluation
- No improvement
- No network calls
- No cloud sync

---

**Dependencies**

```
stdlib only — sqlite3, dataclasses, typing, threading
```

Zero third-party dependencies.

---

**Deliverables**

```
ragradar_capture/
  api.py              # public surface — ragradar.start(), cap.*, ragradar.capture()
  schema.py           # RunRecord and child dataclasses
  store.py            # SQLite write, schema init, migrations
  thread_local.py     # active capture registry
  scaffold/
    template.py       # ragradar-capture init generator
pyproject.toml
README.md
```

(`schema.py` and `store.py` as planned here were later extracted into the
shared `ragradar-core` kernel once `ragradar` and `ragradar-evaluate` needed
the same schema and store — see `ARCHITECTURE.md`. `ragradar_capture` today
holds only `api.py`, `thread_local.py`, and `scaffold/`.)

---

**Acceptance criteria**

- `pip install ragradar-capture` works, zero dependencies beyond stdlib
- `ragradar-capture init` generates a runnable scaffold
- Capture with two fields works without error
- Capture with all fields works without error
- Pipeline never raises on capture failure
- `~/.ragradar/runs.db` created on first run
- Schema version present in `meta` table
- Sessions auto-created on 30-minute idle gap
- Thread-local run accessible across module boundaries

---
---

### ragradar

**What it is**

Standalone analyst CLI. Reads from `~/.ragradar/runs.db`. Browse sessions, search runs, explain a specific run. Read-only consumer of the store.

**What it is not**

Not a capture tool. Not an evaluation tool. Never writes to the store.

---

**Delivery scope**

**Command surface**

```bash
ragradar list                          # list sessions, most recent first
ragradar list s2                       # list runs inside session 2
ragradar find <hint>                   # search runs by query text
ragradar find <hint> --exact           # phrase match instead of token match
ragradar find <hint> --from <date>     # date filter
ragradar find <hint> --to <date>       # date filter
ragradar find <hint> --today           # shorthand date filter
ragradar find <hint> --session <id>    # scope to session
ragradar find <hint> --pipeline <name> # scope to pipeline
ragradar find --recent                 # latest N runs, no hint
ragradar explain                       # latest run
ragradar explain <target>              # specific run — s2r3
ragradar explain <target> --full       # expanded output
ragradar explain <target> --html       # snapshot to ~/.ragradar/reports/
ragradar diff <target> <target>        # compare two runs
ragradar budget <target>               # token waterfall only
ragradar session rename <id> <title>   # rename a session
```

---

**Target addressing**

Resolution order — same for all commands:

```
1. Exact ID (s2r3)          →  direct lookup
2. No arg                   →  latest run in latest session
3. Quoted hint              →  search → single match → proceed
                               multiple matches → disambiguation screen
                               no match → suggest closest
```

---

**Search — ragradar find**

Token match default — hint split into terms, OR logic across query text:

```sql
WHERE query LIKE '%score%' OR query LIKE '%fusion%'
ORDER BY created_at DESC
```

All filters are pre-query SQL. LLM not involved anywhere in the navigation path.

Semantic search — optional, enabled when Ollama or sentence-transformers configured. BM25 weighted 0.7, semantic 0.3, fused via RRF. Falls back to BM25-only gracefully when no model available.

Disambiguation screen when multiple matches:

```
  s2 r3   2026-06-08   RRF investigation   — "does RRF handle score scale differences"
  s2 r2   2026-06-08   RRF investigation   — "why does BM25 score differ from ANN score"

  Pick (s2r3 / s2r2) or refine:
```

---

**Analysis — ragradar explain**

Seven factors. Each computed deterministically at read time from captured run data. Skipped silently if required data not present.

```
Token usage       — per-section breakdown, headroom, model limit
Duplicate chunks  — path dups, window dups, semantic dups (if embedding available)
Chunk scores      — retrieval score + rerank score distribution
Truncation        — which chunks trimmed, at what boundary, score of truncated chunks
Dropped history   — pre/post eviction diff, eviction reason
Cache hits        — hit/miss ratio, which chunks came from cache
Final prompt      — assembled prompt as-is
```

Duplicate detection tiers:

```
[PATH DUP]     same chunk_id, retrieved via both BM25 and ANN
[WINDOW DUP]   same source_doc_id, overlapping token window
[SEMANTIC DUP] cosine sim above threshold (requires embedding model)
```

Output modes:

```
default    —  compact, one screen
--full     —  all sections, all chunks
--html     —  snapshot to ~/.ragradar/reports/<run_id>.html
```

---

**Diff — ragradar diff**

Two confirmed targets. Deterministic comparison:

```
query delta
chunks added / removed
score delta per chunk
token budget delta
history delta
truncation delta
```

Primary use: comparing two iterations of the same query within a session.

---

**ragradar list output**

```
Sessions view:
  ID    RUNS   PIPELINE   CREATED      TITLE
  s5    3      rkis       2h ago       auto: "does RRF handle score..."
  s4    2      rkis       1d ago       Cross-encoder reranking

Runs view (ragradar list s2):
  s2 r4   2026-06-19   "does rerank order depend on retrieval scores"
  s2 r3   2026-06-19   "does RRF handle score scale differences"
```

---

**Out of scope**

- No write operations to the store
- No evaluation scoring
- No improvement passes
- No LLM calls in navigation path
- No cloud sync
- No multi-user support

---

**Dependencies**

```
rich                          # terminal rendering
click                         # CLI framework
sqlite-vec                    # vector search — optional, semantic mode only
sentence-transformers         # embeddings — optional, semantic mode only
```

Semantic dependencies are optional extras — `pip install ragradar[semantic]`.

---

**Deliverables**

```
ragradar/
  cli.py                      # entrypoint — all ragradar commands, incl. session rename
  store.py                    # read-only SQLite queries (delegates to ragradar-core)
  find/
    query_builder.py          # filter → SQL composer
    bm25.py                   # token scoring
    semantic.py               # embedding + cosine (optional, not yet built)
    fusion.py                 # RRF combiner (optional, not yet built)
  explain/
    loader.py                 # fetch RunRecord from store
    analyzers/
      tokens.py
      duplicates.py
      truncation.py
      history.py
      cache.py
      scores.py
    renderer/
      terminal.py             # rich output
      html.py                 # snapshot
pyproject.toml
README.md
```

(`find/semantic.py` and `find/fusion.py` remain unimplemented — optional
semantic search was scoped here but deferred, as `design-doc.md` notes.
`session rename` shipped as a command group inside `cli.py` rather than a
separate `session.py` module.)

---

**Acceptance criteria**

- `pip install ragradar` works
- `ragradar list` shows sessions in recency order
- `ragradar list s2` shows runs scoped to session 2
- `ragradar find "term"` returns all matching runs
- All date and pipeline filters work correctly
- `ragradar explain` with no arg explains latest run
- `ragradar explain s2r3` explains correct run
- All seven analysis factors render when data present
- Factors skip silently when data absent — no errors
- `ragradar explain --html` writes file to `~/.ragradar/reports/`
- `ragradar diff s2r3 s2r1` produces side-by-side comparison
- `ragradar budget s2r3` renders token waterfall only
- `ragradar session rename s2 "title"` persists rename
- Disambiguation screen fires on multiple search matches
- No write operations to runs.db under any circumstance
- Schema version mismatch produces a clear warning

---
---

## Tool 3 — `ragradar-evaluate`

**Delivered separately, after ragradar-capture + ragradar are stable.**

---

### What it is

Evaluation layer. Takes a captured run. Scores it across two dimensions — input quality (mechanical, deterministic) and output quality (RAGAS or equivalent, LLM-as-judge). Writes scores back to the run record. Accumulates benchmark data over time.

**What it is not**

Not a capture tool. Not a browsing tool. Not an improvement tool.

---

**Delivery scope**

**Two evaluation layers**

Layer 1 — Input quality. Deterministic. No LLM required.

```
Relevance score     — SLM cosine similarity per chunk vs query
Duplicate ratio     — path + window + semantic duplicates as percentage
Truncation severity — were high-score chunks truncated?
Token efficiency    — headroom, low-score chunk ratio
Coherence signal    — source domain count, score variance
```

Layer 2 — Output quality. LLM-as-judge via RAGAS.

```
faithfulness
answer_relevancy
context_precision
context_recall
```

Both layers write to a new `eval_scores` field on the run record.

---

**Benchmark system**

Builds correlation model between input quality factors and output quality scores across accumulated runs.

```bash
ragradar-evaluate benchmark build          # correlate input factors vs output scores
ragradar-evaluate benchmark show           # display discovered thresholds
ragradar-evaluate benchmark check s2r3     # score a run against benchmark
ragradar-evaluate benchmark export         # export as RAGAS-compatible dataset
```

Bootstrap path — no historical runs yet:

```bash
ragradar-evaluate benchmark seed           # generate synthetic known-good
                                      # and known-bad context windows
                                      # as day-zero baseline
```

Synthetic baseline is replaced progressively by real run data.

---

**Policy system**

Human-defined rules encoding known failure modes. Active from day one, before benchmark has data.

```python
@dataclass
class InputQualityPolicy:
    min_chunk_relevance_score:    float = 0.5
    min_top_chunk_score:          float = 0.7
    max_duplicate_ratio:          float = 0.2
    max_low_score_chunk_ratio:    float = 0.3
    min_token_headroom:           float = 0.15
    max_high_score_truncations:   int   = 0
    max_source_domains:           int   = 3
    llm_rewrite_risk_threshold:   float = 0.7
```

Defaults encode known failure modes from the literature. Developer overrides per pipeline.

---

**Risk score**

Single 0.0–1.0 score computed from input state against active policy. Gates Stage 3 improvement (future tool). Stored on the run record.

```python
def compute_risk_score(run, policy) -> float:
    # weighted sum of policy violations
    # truncation and top chunk score weighted highest
```

---

**Schema additions — ragradar-evaluate owns**

```sql
ALTER TABLE runs ADD COLUMN eval_scores JSON;
ALTER TABLE runs ADD COLUMN risk_score REAL;
ALTER TABLE runs ADD COLUMN evaluated_at TEXT;

CREATE TABLE benchmark (
    pipeline        TEXT,
    factor          TEXT,
    threshold       REAL,
    correlation     REAL,
    sample_count    INTEGER,
    updated_at      TEXT,
    PRIMARY KEY (pipeline, factor)
);

CREATE TABLE policies (
    pipeline        TEXT PRIMARY KEY,
    policy_data     JSON,
    updated_at      TEXT
);
```

---

**CLI surface**

```bash
ragradar-evaluate run s2r3                 # evaluate one run — both layers
ragradar-evaluate run s2r3 --input-only    # skip RAGAS, input quality only
ragradar-evaluate run s2r3 --output-only   # skip input, RAGAS only
ragradar-evaluate run --session s2         # evaluate all runs in session
ragradar-evaluate benchmark build
ragradar-evaluate benchmark show
ragradar-evaluate benchmark check s2r3
ragradar-evaluate benchmark seed
ragradar-evaluate benchmark export
ragradar-evaluate policy show              # show active policy
ragradar-evaluate policy set <field> <val> # update a policy value
ragradar-evaluate policy reset             # restore defaults
```

---

**Out of scope**

- No capture
- No browsing
- No improvement passes
- No cloud evaluation services beyond RAGAS API calls
- ragradar-improve integration deferred — risk score computed and stored, consumed later

---

**Dependencies**

```
ragas                         # output quality scoring
sentence-transformers         # input relevance scoring (SLM)
                              # OR: ollama client
scipy                         # correlation computation for benchmark
rich                          # terminal rendering
click                         # CLI framework
```

---

**Deliverables**

```
ragradar_evaluate/
  cli.py                      # entrypoint
  facade.py                   # public task API — check(), evaluate(), available_metrics()
  layers/
    input_quality.py          # deterministic input scoring
    output_quality.py         # RAGAS integration
  benchmark/
    builder.py                # correlation analysis
    seeder.py                 # synthetic baseline generator
    checker.py                # run vs benchmark scoring
    exporter.py                # RAGAS dataset export
  policy/
    schema.py                 # InputQualityPolicy dataclass
    persistence.py             # policy read/write
    risk.py                   # risk score computation
pyproject.toml
README.md
```

(The task-level `check()`/`evaluate()`/`available_metrics()` facade above
was added after this scope was drafted — the CLI now routes through it
instead of scoring directly. `eval_scores`/benchmark persistence itself
lives in the shared `ragradar-core` store, not a package-local `store.py`.)

---

**Acceptance criteria**

- `ragradar-evaluate run s2r3` produces input + output scores
- `ragradar-evaluate run s2r3 --input-only` runs without RAGAS dependency
- Scores written to run record, readable by ragradar explain
- `ragradar-evaluate benchmark build` requires minimum 10 runs
- `ragradar-evaluate benchmark seed` generates usable day-zero baseline
- `ragradar-evaluate benchmark show` displays per-factor thresholds
- `ragradar-evaluate benchmark export` produces RAGAS-compatible dataset
- Risk score between 0.0 and 1.0 stored on run record
- Policy defaults apply without any developer configuration
- Policy overrides persist across sessions
- Schema migration runs cleanly on existing runs.db

---

## Build order

```
Phase 1   ragradar-capture + ragradar          single release, delivered together
Phase 2   ragradar-evaluate               after Phase 1 is stable
Phase 3   ragradar-improve                lowest priority, future
```

## Future — ragradar-improve

Deferred. Acts on risk score and benchmark findings to improve context quality before the LLM call. Three stages — filter (rules + SLM), rerank (SLM), rewrite (LLM, opt-in). Consumes output of ragradar-evaluate. No scope defined until ragradar-evaluate is stable and benchmark has real data.