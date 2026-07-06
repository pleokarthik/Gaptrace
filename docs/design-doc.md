# ragradar — Design Document

**Status:** v0.3 — updated for the ragradar-core extraction, the Capture
rename (capture is the verb, run is the data noun), and the task-level
evaluation facade (`check()` / `evaluate()` / `available_metrics()`).
See `docs/ARCHITECTURE.md` for the current package/dependency picture.  
**Author:** Leo Karthik Paramasivan  
**Date:** 2026-07-03

---

## 1. Problem

A RAG pipeline assembles a context window from retrieved chunks, conversation history, and system instructions, then sends it to an LLM. Every existing evaluation tool — RAGAS, LangSmith, TruLens, DeepEval — fires after the LLM responds. They measure whether the output was faithful, relevant, or grounded. None of them inspect what went into the context window before the expensive call.

```
Query -> [Retrieve] -> [Assemble] -> CONTEXT WINDOW -> [LLM] -> Response
                                          ^                        ^
                                  nothing inspects this     everything measures this
```

This creates a blind spot for an entire class of failures that are mechanical, deterministic, and fixable without touching the model. These failures are invisible because the pipeline still produces a plausible-looking response — the LLM papers over bad input with confident prose.

The seven failure modes that ragradar surfaces:

**Token misallocation.** The token budget is consumed disproportionately by low-value content — system instructions that could be shorter, history turns that add no information, chunks with low relevance scores taking up space that higher-scored chunks needed. The developer never sees that headroom hit 4% on a 4096-token window because the response still came back.

**Duplicate chunks.** The same content enters the context window through multiple retrieval paths. A hybrid retriever using both BM25 and ANN can return the same passage twice — once from each index. Overlapping chunk windows from the same source document produce near-identical text. These duplicates waste token budget and can bias the LLM toward the duplicated topic.

**High-score truncation.** When the assembled context exceeds the token budget, chunks are trimmed or dropped. If the truncation logic is score-unaware, it may cut a chunk with a rerank score of 0.88 while keeping one scored 0.39. The developer never knows this happened because the response still looks reasonable.

**History eviction.** Multi-turn conversations require history in the context window. When history competes with retrieval chunks for token budget, turns are dropped. The question is which turns were dropped and why. If the eviction removed context that grounded the current query, the response degrades silently.

**Cache staleness.** Chunk caching is common in production RAG systems. A cache hit means the chunk was not re-retrieved for the current query — it may reflect an older retrieval state. Whether this matters depends on the workload, but the developer cannot evaluate it without visibility.

**Score distribution problems.** A wide spread between the highest and lowest chunk scores in a single context window suggests the retriever is pulling in low-relevance content to fill budget. A high rerank delta (difference between mean rerank and mean retrieval scores) suggests the reranker is significantly reordering the retriever's output, which may indicate retriever miscalibration.

**Source fragmentation.** When chunks are drawn from many different source documents, the context window lacks coherence. A context assembled from 8 different sources is less likely to contain the depth needed for a good answer than one assembled from 2-3 sources covering the topic thoroughly.

RAGAS and similar tools measure output quality after the fact. ragradar measures input quality before the call. These are complementary — ragradar can feed RAGAS via benchmark export, and RAGAS output scores can be correlated against ragradar input factors to discover which mechanical failures actually predict bad outputs for a specific pipeline. But RAGAS cannot tell you that your best chunk was truncated, or that 30% of your context was duplicated. ragradar can.

---

## 2. Architecture

ragradar is a four-tool system built on one shared kernel. Three tools are implemented. The fourth is deferred.

```
your pipeline
  +-- ragradar_capture (ragradar-capture)  ->  ~/.ragradar/runs.db
                                     ^
                    ragradar (analyst CLI) |
                    ragradar-evaluate      +
                         ^
     all three depend on ragradar-core (schema + store + sNrN parser)
```

**ragradar-core** is the shared kernel (`import ragradar_core`): the `RunRecord` dataclasses, the single SQLite store (location, schema, migrations, all persistence primitives), and the one `sNrN` target parser. Zero third-party dependencies, enforced by test. Users never import it directly — `ragradar_capture` and `ragradar_evaluate` re-export the dataclasses.

**ragradar_capture / ragradar-capture** is the instrumentation SDK. It writes structured run records to the local SQLite store and hands back the run's `sNrN` id. Beyond ragradar-core it is stdlib-only. This is a deliberate constraint: the SDK runs inside the developer's pipeline, so it must never introduce dependency conflicts, slow imports, or failure modes.

**ragradar** is the analyst CLI. It reads from the same SQLite store and renders analysis. It depends on `rich` for terminal rendering and `click` for the CLI framework. Optional semantic search depends on `sentence-transformers` and `sqlite-vec`, gated behind `pip install ragradar[semantic]`. ragradar is read-mostly — it never modifies run data. The single exception is `ragradar session rename`, which writes to the `sessions.title` column (and, like every entry point, opening the store may create/migrate it via ragradar-core).

**ragradar-evaluate** is the evaluation layer. Its public API is two user tasks — `check(target)` ("is this run healthy?", free and deterministic) and `evaluate(target, metrics=...)` (complete or atomic-metric scoring) — plus `available_metrics()` discovery, all built on a per-metric engine. It depends on `ragas`, `scipy`, `rich`, and `click` (the heavy ones imported lazily). It owns the `eval_scores`, `risk_score`, and `evaluated_at` values on the runs table, plus the `benchmark` and `policies` tables.

**ragradar-improve** is deferred. It will act on risk scores and benchmark findings to improve context quality before the LLM call — filtering low-value chunks, reranking via SLM, optionally rewriting via LLM. No scope is defined until ragradar-evaluate's benchmark system has accumulated real pipeline data.

The coupling points between the tools are `~/.ragradar/runs.db` and the ragradar-core kernel that owns it. ragradar, ragradar-capture, and ragradar-evaluate never import each other. Each store function opens and closes its own connection; there is no pooling or shared state.

Ownership boundaries:

| Resource | Owner | Others |
|---|---|---|
| Schema, migrations, connection contract | ragradar-core | everyone connects through it |
| `runs.run_data` JSON | ragradar-capture (write) | ragradar, ragradar-evaluate read — never rewrite |
| `runs.eval_scores`, `runs.risk_score`, `runs.evaluated_at` | ragradar-evaluate | ragradar reads |
| `benchmark` table | ragradar-evaluate | — |
| `policies` table | ragradar-evaluate | — |
| `sessions.title` | ragradar-capture (create) | ragradar writes (rename only) |
| Schema version in `meta` | ragradar-core (single constant, currently "3") | — |

---

## 3. Data model

All data types are defined in `ragradar_core/schema.py` (the shared kernel package; `ragradar_capture` and `ragradar_evaluate` re-export them) as Python dataclasses. A `_flexible` decorator wraps each dataclass's `__init__` to accept and ignore unknown keyword arguments, ensuring forward compatibility — future fields added to a dataclass never cause `TypeError` in code using an older version of the schema.

```python
def _flexible(cls):
    original_init = cls.__init__
    @functools.wraps(original_init)
    def init(self, *args, **kwargs):
        valid = {f.name for f in fields(cls)}
        original_init(self, *args, **{k: v for k, v in kwargs.items() if k in valid})
    cls.__init__ = init
    return cls
```

### RunRecord

The top-level container for a single pipeline execution. Only `query` and `response` are required. Every other field defaults to `None` and is populated only if the corresponding pipeline stage is instrumented.

```python
@dataclass
class RunRecord:
    query:           str                           # required
    response:        str                           # required
    chunks:          Optional[list[ChunkRecord]]   # retrieval stage
    requested_chunk_count: Optional[int]           # retrieval stage (top_k ask)
    final_prompt:    Optional[str]                 # assembly stage
    token_budget:    Optional[TokenBudget]         # assembly stage
    history_pre:     Optional[list[Turn]]          # history management
    history_post:    Optional[list[Turn]]          # history management
    eviction_reason: Optional[str]                 # history management
    cache_events:    Optional[list[CacheEvent]]    # cache layer
    tool_calls:      Optional[list[ToolCallRecord]] # tool-call layer
    model:           Optional[str]                 # LLM call
    token_usage:     Optional[TokenUsage]          # LLM call
    cache:           Optional[CacheRecord]         # semantic-cache stage
    filter:          Optional[FilterRecord]        # metadata-filter stage
```

This optionality contract means a pipeline instrumented with only `ragradar_capture.capture(query, response)` produces a valid RunRecord with two fields. A fully instrumented pipeline populates all fifteen. The analysis tools (ragradar explain, ragradar-evaluate) check for the presence of each field before computing — if `chunks` is None, the duplicates analyzer returns None and the renderer skips that panel silently.

`requested_chunk_count` is a bare `Optional[int]` scalar, the same shape as `eviction_reason` — no wrapper dataclass, since it has no sibling fields. It is also the first capture-side schema addition since `cache`/`filter` (the semantic-cache and metadata-filter stages): every analysis factor added between then and now (`score_degeneracy`, `score_margin`) was a pure derivation over data `RunRecord` already carried, but `candidate_underfill_risk` needed a genuinely new capture surface — the retriever's `top_k` ask isn't otherwise recoverable from `len(chunks)` alone.

### ChunkRecord

Represents a single retrieved chunk in the context window.

```python
@dataclass
class ChunkRecord:
    chunk_id:         str                # unique identifier
    source_doc_id:    str                # parent document
    content:          str                # chunk text
    token_count:      int                # token length
    retrieval_score:  Optional[float]    # raw retrieval score
    rerank_score:     Optional[float]    # cross-encoder score
    retrieval_path:   Optional[str]      # "bm25" | "ann" | "hybrid"
    truncated:        bool = False       # was this chunk trimmed?
    cache_hit:        Optional[bool]     # served from cache?
```

The `retrieval_path` field enables path duplicate detection — when the same `chunk_id` appears twice with different paths (e.g., once via BM25, once via ANN), it indicates the hybrid retriever returned the same content through both indexes.

### TokenBudget

Records how the token budget was allocated during context assembly.

```python
@dataclass
class TokenBudget:
    total_limit:       int    # model context window size
    chunks_allocated:  int    # tokens given to retrieval chunks
    history_allocated: int    # tokens given to conversation history
    system_allocated:  int    # tokens given to system instructions
    headroom:          int    # remaining unused tokens
```

The invariant is `total_limit = chunks_allocated + history_allocated + system_allocated + headroom`. Low headroom (below 15% of total_limit) indicates the pipeline is operating near capacity with little room for longer queries or responses.

### TokenUsage, Turn, CacheEvent

```python
@dataclass
class TokenUsage:
    input_tokens:  int
    output_tokens: int
    total_tokens:  int

@dataclass
class Turn:
    role:    str              # "user" | "assistant"
    content: str
    tokens:  Optional[int]

@dataclass
class CacheEvent:
    chunk_id:     str
    hit:          bool
    cache_source: Optional[str]   # "disk" | "redis" | etc.
```

### Serialization contract

`RunRecord.to_json()` returns a JSON-serializable dict using `dataclasses.asdict()`, which recursively converts nested dataclasses. `RunRecord.from_json(data)` reconstructs the full object tree by explicitly instantiating each nested type:

```python
@classmethod
def from_json(cls, data: dict) -> "RunRecord":
    data = dict(data)
    if data.get("chunks") is not None:
        data["chunks"] = [ChunkRecord(**c) for c in data["chunks"]]
    if data.get("token_budget") is not None:
        data["token_budget"] = TokenBudget(**data["token_budget"])
    # ... same pattern for history_pre, history_post, cache_events, token_usage
    return cls(**data)
```

Because every dataclass uses the `_flexible` decorator, `from_json` is forward-compatible: if the serialized JSON contains fields that were added in a later schema version, they are silently dropped during deserialization rather than raising `TypeError`.

The `run_data` column in the runs table stores the output of `to_json()` as a JSON string. This column is write-once — ragradar-capture writes it on commit, and no other tool ever modifies it. ragradar and ragradar-evaluate deserialize it via `from_json()` at read time.

---

## 4. Store and schema

Owned entirely by `ragradar_core.store` since the ragradar-core extraction: one schema, one version constant (`SCHEMA_VERSION = "3"`), one connection contract. `connect()` guarantees on every call that `~/.ragradar/` exists, `runs.db` exists, and the schema is at the latest version — fresh databases are created directly at v3 (eval columns, benchmark/policies tables, and the FTS5 index included), and databases written by older releases are migrated in place on first connect. The historical version story below is retained because the migration chain still runs against old databases.

### Schema v1 — the original capture schema

The database is created at `~/.ragradar/runs.db` on first capture. The directory `~/.ragradar/` is created if it does not exist. The store uses WAL (Write-Ahead Logging) mode, set via `PRAGMA journal_mode=WAL` in the schema initialization script. WAL was chosen because it allows concurrent readers (ragradar browsing) while a writer (ragradar-capture) is active, which matters when a developer runs `ragradar explain` while their pipeline is still capturing.

```sql
PRAGMA journal_mode=WAL;

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
```

The `meta` table stores `schema_version` as a key-value pair. Schema v1 set this to `"1"`. (The old ragradar-CLI startup version warning is gone: connect-time migration means a supported database is always at the latest version by the time any command reads it, and an unsupported version raises `RuntimeError` from ragradar-core.)

### Session auto-creation

Sessions group runs automatically based on idle time. `get_or_create_session(pipeline, idle_gap_minutes=30)` finds the most recent session for the given pipeline, checks the timestamp of the last run in that session (or the session's own `created_at` if no runs exist), and compares against the current time. If the gap exceeds `idle_gap_minutes`, a new session is created. This means a developer who steps away for lunch gets a new session automatically without any explicit action. Sessions with different `pipeline` values are tracked independently.

### Connection management

Each store function opens and closes its own connection using a context manager pattern (`with sqlite3.connect(...) as conn`). There are no module-level singletons or connection pools. This makes the store safe to call from multiple threads or processes, which matters because ragradar_capture uses thread-local storage for active run tracking.

### Schema v2/v3 — the migration chain (now owned by ragradar-core)

When any ragradar package first connects to an existing v1 database, ragradar-core applies a migration that adds three columns to the `runs` table and creates two new tables (v1 → v2), then adds the FTS5 `runs_fts` index with insert/update/delete sync triggers and drops the redundant `idx_runs_query` (v2 → v3).

```sql
ALTER TABLE runs ADD COLUMN eval_scores  TEXT;
ALTER TABLE runs ADD COLUMN risk_score   REAL;
ALTER TABLE runs ADD COLUMN evaluated_at TEXT;

CREATE TABLE IF NOT EXISTS benchmark (
    pipeline      TEXT NOT NULL,
    factor        TEXT NOT NULL,
    threshold     REAL,
    correlation   REAL,
    sample_count  INTEGER NOT NULL DEFAULT 0,
    updated_at    TEXT NOT NULL,
    PRIMARY KEY (pipeline, factor)
);

CREATE TABLE IF NOT EXISTS policies (
    pipeline     TEXT PRIMARY KEY,
    policy_data  TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
```

The migration must be safe against existing Phase 1 data. SQLite does not support `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`, so the migration uses `PRAGMA table_info(runs)` to check whether each column already exists before issuing the ALTER statement. The `_column_exists` helper iterates the pragma output and returns a boolean. New tables use `CREATE TABLE IF NOT EXISTS`, which is natively idempotent.

Each step updates the meta table's `schema_version` (`"1"` → `"2"` → `"3"`) and commits before the next runs, so a v1 database walks the whole chain in one connect. A version the package doesn't know raises `RuntimeError` — the database is in an unknown state and automated migration is not safe.

Migration runs inside `ragradar_core.store.connect()`, so it happens automatically the first time any package (library call or CLI) touches an old database. Because it is idempotent, connecting repeatedly is safe.

---

## 5. Capture API — ragradar_capture

### Staged capture

The primary capture pattern uses a `Capture` object — the action object for one pipeline run — that accumulates data across pipeline stages, then writes to the store on commit. Committing returns the run's `sNrN` id.

```python
import ragradar_capture

cap = ragradar_capture.start(query="what is RRF", pipeline="my_project")

cap.chunks(chunks)                              # after retrieval
cap.context(final_prompt, token_budget)         # after assembly
cap.history(pre=pre_turns, post=post_turns,
            eviction_reason="token_budget")      # after eviction
cap.cache(cache_events)                         # cache hit/miss data
run_id = cap.response(response, token_usage=usage,
                      model="gpt-4")            # after LLM call

# cap.commit() is called automatically by cap.response(); the returned
# run_id ("sNrN") feeds ragradar explain and ragradar_evaluate.check()/evaluate()
```

`ragradar_capture.start()` creates a `Capture` instance, registers it as this thread's active capture, and returns it. Each stage method (`chunks`, `context`, `history`, `cache`, `tool_call`, `response`) sets fields on the internal `RunRecord`. Calling `cap.response()` automatically calls `cap.commit()`, which writes the accumulated record to the store and returns the run id (also available afterwards as `cap.run_id`; `None` before commit). Calling `commit()` explicitly after `response()` is safe — `commit()` is idempotent and returns the same id without writing again.

Each stage method accepts either typed dataclass instances or plain dicts. For example, `cap.chunks()` accepts a list where each element is either a `ChunkRecord` or a dict that will be unpacked into `ChunkRecord(**d)`. This means the developer does not need to import the dataclass types if they prefer to pass dicts. `cap.chunks(chunks, requested_count=10)` additionally records the retriever's `top_k` ask onto `RunRecord.requested_chunk_count` — a bare optional int, unvalidated against `len(chunks)` — for `ragradar-evaluate`'s `score_score_underfill`.

### Thread-local active capture

When `ragradar_capture.start()` is called, it stores the Capture instance in a `threading.local()` variable. Module-level proxy functions — `ragradar_capture.chunks()`, `ragradar_capture.context()`, `ragradar_capture.history()`, `ragradar_capture.response()`, `ragradar_capture.cache()`, `ragradar_capture.tool_call()`, `ragradar_capture.commit()` — look up the active capture via `get_active_capture()` and delegate to it.

This matters for pipelines where retrieval, assembly, and LLM stages live in different files or modules. The developer calls `ragradar_capture.start()` in the orchestrator, then calls `ragradar_capture.chunks(chunks)` in the retriever module without passing the capture object through the call chain. The thread-local lookup finds the correct capture automatically.

If a proxy function is called with no active capture, it logs an error to `~/.ragradar/errors.log` and returns silently. It never raises.

### Single-line fallback

For pipelines where staged instrumentation is not yet practical, `ragradar_capture.capture()` provides a single-call interface:

```python
run_id = ragradar_capture.capture(query, response, pipeline="my_project")
```

This creates a Capture internally, sets the response, routes the optional keyword-only arguments to the appropriate stage fields (`chunks=`, `requested_chunk_count=`, `final_prompt=`, `token_budget=`, `history_pre=`/`history_post=`, `eviction_reason=`, `cache_events=`, `tool_calls=`, `model=`, `token_usage=`), commits immediately, and returns the run id. The signature is explicit — a misspelled keyword raises `TypeError` at the call site rather than being silently dropped. `requested_chunk_count=` is only recorded when `chunks=` is also given.

### Failure contract

Every public method on the `Capture` class and every module-level function wraps its body in `try/except Exception`. Caught exceptions are logged to `~/.ragradar/errors.log` via a stdlib `logging.FileHandler` with the format:

```
2026-06-26T10:23:11 [ragradar-capture] capture.chunks() failed: <message>
```

No exception from ragradar_capture ever propagates to the caller. The pipeline must never be interrupted by instrumentation failure. This is the core design constraint of the capture layer — it must be invisible on failure. A developer who instruments their pipeline and deploys it cannot have ragradar_capture bring down production because the SQLite disk is full or a chunk dict has an unexpected type. Where a run id was expected, the call returns `None` instead.

For development, strict mode inverts the contract: `ragradar_capture.set_strict(True)` (or `RAGRADAR_CAPTURE_STRICT=1`) makes conversion and commit errors raise so instrumentation bugs surface immediately.

### Scaffold generator

`ragradar-capture init` generates a `ctx_pipeline.py` file in the current directory with capture calls pre-positioned at the correct pipeline stages. The generated file contains commented-out placeholders for each stage, showing where to insert retriever calls, assembler calls, and LLM calls. This is the greenfield onboarding path — a developer who has never used ragradar can generate the scaffold, fill in their pipeline functions, and have full instrumentation from the first run.

The scaffold raises `FileExistsError` if `ctx_pipeline.py` already exists, preventing accidental overwrites.

---

## 6. Search and navigation — ragradar

### Target addressing

Every ragradar command that operates on a specific run uses the same resolution order:

1. **Exact ID** — `s2r3` resolves directly to session 2, run 3 via regex match `^s(\d+)r(\d+)$` (case-insensitive).
2. **No argument** — resolves to the latest run across all sessions, ordered by `created_at DESC`.
3. **Text hint** — passed to `search_runs()`, which performs a SQL query. If exactly one result matches, that run is used. If multiple results match, the caller presents a disambiguation screen.

### SQL-first search

Search is implemented entirely in SQL. The `build_search_query()` function in `find/query_builder.py` composes a SELECT statement by accumulating WHERE clauses from the provided arguments.

The base query joins `runs` and `sessions`:

```sql
SELECT r.session_id, r.run_seq, r.query, r.pipeline,
       r.created_at, s.title as session_title
FROM runs r JOIN sessions s ON r.session_id = s.session_id
```

Filters are appended conditionally:

- **Token match** (default): the hint is split on whitespace into tokens, each generating a `r.query LIKE ?` clause joined with OR. A hint of `"score fusion"` produces `(r.query LIKE '%score%' OR r.query LIKE '%fusion%')`. This is deliberately loose — it finds runs whose query text contains any of the search terms.
- **Exact match** (`--exact`): the full hint is wrapped in a single `r.query LIKE '%hint%'` clause, requiring the exact phrase as a substring.
- **Session filter** (`--session s2`): `r.session_id = ?`
- **Pipeline filter** (`--pipeline name`): `r.pipeline = ?`
- **Date range** (`--from`, `--to`): `r.created_at >= ?` and `r.created_at <= ?` respectively. The `--today` flag is syntactic sugar that sets `from` to today's ISO date and `to` to `{today}T23:59:59.999999Z`.
- **Recent** (`--recent N`): appends `LIMIT ?` to the query.

All filters compose via AND — they narrow the result set, not expand it. The result is always ordered by `r.created_at DESC`.

No LLM is involved anywhere in the navigation path. Search is SQL. Ranking is term frequency. This is a deliberate constraint — the developer should be able to find any run instantly without waiting for a model call.

### BM25 scorer

When `resolve_target()` receives a text hint that matches multiple runs, it sorts the results using a simple term-frequency scorer before presenting them for disambiguation. The `score(hint, query_text)` function splits the hint into tokens, counts how many appear in the query text (case-insensitive), and returns the ratio `matched_tokens / total_tokens`. This is not a full BM25 implementation — it is sufficient for ranking a small disambiguation list.

### Semantic search

Optional, not yet wired into the default search path. When enabled via `pip install ragradar[semantic]`, search would use a BM25 + semantic fusion weighted 0.7/0.3 via Reciprocal Rank Fusion. The system falls back to BM25-only gracefully when no embedding model is available. The `find/semantic.py` and `find/fusion.py` modules are specified in the scope but not yet implemented — the SQL-first path handles all current use cases.

### Disambiguation screen

When a search returns multiple matches, ragradar presents a numbered list:

```
  Multiple matches:

  1   s2 r3   2026-06-08   RRF investigation   -- "does RRF handle score scale differences"
  2   s2 r2   2026-06-08   RRF investigation   -- "why does BM25 score differ from ANN score"

  Pick (number) or press Enter to cancel:
```

The user enters a number to select a run. Invalid input or Enter cancels the operation. The disambiguation logic lives in `_disambiguate()` in `cli.py`, using `click.prompt()` for input capture.

---

## 7. Analysis — ragradar explain

Eleven analysis factors are computed deterministically at read time from captured run data. Each factor is implemented as a standalone analyzer module in `ragradar/explain/analyzers/`. Every analyzer follows the same contract: it takes a `RunRecord`, checks whether the required data is present, and returns either a structured dict or `None`. The renderer skips any factor that returned `None` — there is no error, no placeholder, no "data not available" message.

### tokens.py

Requires `chunks` or `final_prompt`. Computes per-section token breakdown from the stored `TokenBudget` and chunk `token_count` fields. History tokens come from `history_post` (preferred) or `history_pre`. Returns total tokens, per-section allocation, headroom, model limit, and utilization percentage. The per-chunk breakdown lists each chunk's ID and token count.

### scores.py

Requires `chunks` with at least one non-None `retrieval_score` or `rerank_score`. Computes the range (min/max) for both retrieval and rerank scores, the rerank delta (mean rerank minus mean retrieval — a measure of how much the reranker changed the ordering), and the low-score ratio (proportion of chunks with rerank score below 0.5).

### duplicates.py

Requires `chunks`. Detects three tiers of duplication:

**PATH DUP**: Same `chunk_id` appears multiple times with different `retrieval_path` values. This happens when a hybrid retriever returns the same passage via both its BM25 and ANN indexes. Detected by grouping chunks by `chunk_id` and checking for multiple paths.

**WINDOW DUP**: Same `source_doc_id`, overlapping content. This happens when a document is chunked with overlapping windows. Detected by grouping chunks by `source_doc_id`, then checking each pair — if one chunk's content is a substring of the other's, they are window duplicates.

**SEMANTIC DUP**: Deferred in the ragradar analyzer. Requires an embedding model to compute cosine similarity between chunk pairs from different source documents. The ragradar-evaluate input quality layer implements this when an `embedding_fn` is provided.

The duplicate ratio is `(path_dups + window_dups) / total_chunks`.

### truncation.py

Requires `chunks`. Counts chunks where `truncated=True`, then counts how many of those have a retrieval or rerank score above 0.7 (high-score truncations). Severity classification: `"none"` if no chunks truncated, `"high"` if any high-score chunk was truncated, `"low"` otherwise. High-score truncation is the most concerning failure mode — it means the pipeline cut content that the retriever and reranker agreed was relevant.

### history.py

Requires `history_pre` or `history_post`. Computes pre and post turn counts, identifies dropped turns (present in pre but absent from post, matched by `(role, content)` tuple), and reports the eviction reason. Token sums are computed for turns that have `tokens` set, returning `None` if no turn has token counts.

### cache.py

Requires `cache_events`. Counts hits and misses, computes the hit ratio, and lists the chunk IDs for each category.

### degeneracy.py

Requires `chunks`. Computes chunk-score variance — per-chunk score is `rerank_score`, falling back to `retrieval_score` when absent; chunks with neither are excluded. Returns `None` for the variance with fewer than two chunks carrying a usable score (variance is undefined with under two points). Near-zero variance means the retriever isn't discriminating between chunks at all — the same signal `ragradar-evaluate`'s `score_degeneracy` metric computes, under the distinct key `chunk_score_variance` (kept separate from `coherence`'s `score_variance`, which is rerank-only with no fallback).

### semantic_cache.py

Requires `cache` (the query-level semantic-cache check, distinct from the per-chunk `cache_events` above). Not a separate analyzer entry in the generic loop — the renderer calls it directly, since it needs the active pipeline's `InputQualityPolicy` for its thresholds. Flags a `borderline_hit` when a cache hit's similarity score landed within `cache_borderline_margin` of the cache's own threshold, and a `stale_hit` when a hit's `cached_at` is older than `cache_max_age_seconds`. Renders "Not checked" if `record.cache.checked` is `False`.

### metadata_filter.py

Requires `filter`. Also rendered outside the generic loop, directly after semantic cache, to preserve display order. Computes `filtered_exclusion_ratio` — the share of the candidate pool a metadata filter excluded before retrieval/scoring ever saw it — `None` when `candidate_count`/`excluded_count` weren't captured or `candidate_count` isn't positive. Renders "Not applied" if `record.filter.applied` is `False`.

### margin.py

Requires `chunks`. Not a separate analyzer entry in the generic loop either — like `semantic_cache.py`, the renderer calls it directly (rendered last, after the metadata filter) because it needs the active pipeline's `InputQualityPolicy`. Mirrors `ragradar-evaluate`'s `score_score_margin`: `top_second_margin` (top chunk score minus the runner-up's, rerank falling back to retrieval, same convention as `degeneracy.py`) and `threshold_margin` (top chunk score minus `min_top_chunk_score`, diagnostic-only). Returns `None` with fewer than two chunks carrying a usable score.

### underfill.py

Requires `chunks` and a captured `requested_chunk_count` (the retriever's `top_k` ask — see §3's `RunRecord` note on this being the first new capture-side field since `cache`/`filter`). Unlike `margin.py`/`semantic_cache.py`, this one needs no policy and no ordering control, so it's back in the generic `_ANALYZERS` loop rather than special-cased. Mirrors `ragradar-evaluate`'s `score_score_underfill`: `underfill_ratio` (`(requested - returned) / requested`, negative when more chunks came back than asked for) plus `requested_chunk_count`/`returned_chunk_count` as diagnostic context. Returns `None` when `requested_chunk_count` wasn't captured or isn't positive, or when `chunks` is `None`.

### Final prompt

Not a separate analyzer — the renderer checks `record.final_prompt` directly. In compact mode, the first 500 characters are shown. In full mode, the entire prompt is displayed. In HTML mode, the prompt is rendered inside a `<pre>` block.

### Output modes

**Compact** (default): One-screen summary. Each factor renders as a Rich Panel with a title and color-coded border — green for healthy signals, yellow for warnings, red for detected problems. Summary lines only, no per-chunk detail.

**Full** (`--full`): All detail. Per-chunk token counts, individual score values, full list of dropped turns, complete final prompt.

**HTML** (`--html`): Writes a self-contained HTML file to `~/.ragradar/reports/{run_id}.html`. No external dependencies — inline CSS, collapsible `<details>` sections for each factor. The file is a snapshot that can be shared or archived.

---

## 8. Evaluation — ragradar-evaluate

### Two-layer design

The public API is task-shaped: `check(target)` answers "is this run healthy?" (all free input metrics vs the current standards — learned benchmark thresholds once ≥10 evaluated runs exist for the pipeline, policy defaults before that; `CheckResult.thresholds` says which); `evaluate(target, metrics=None, ground_truth=None, policy=None, save=True)` scores everything applicable or exactly the atomic metrics named, returning an `EvalResult` (per-metric results, `skipped` reasons, a single `errors` channel for both missing and failing RAGAS, `risk_score` that is `None` when input metrics weren't computed, and save semantics); `available_metrics()` lists the fourteen metrics with layer/cost/requirements. Targets are `sNrN` strings, committed `Capture` objects, or bare `RunRecord`s (which cannot be saved). `evaluate()` is the only persistence path — the CLI routes through it.

Under the facade, evaluation is split into two layers that run independently. Layer 1 (input quality) is deterministic, requires no LLM, and uses only stdlib math; each metric family (relevance, duplicates, truncation, token_efficiency, coherence) is its own function, so selecting one never computes the others. Layer 2 (output quality) uses RAGAS with an LLM-as-judge, passing exactly the selected metric objects to the judge. The `--input-only` flag maps to the input metric set, and the RAGAS import is deferred — it happens inside the function body, not at module top level — so input-only evaluation runs even if RAGAS is not installed.

### Layer 1 — input_quality.py

Per-family functions plus a `score_input_quality()` dispatcher that takes a `RunRecord` and an `InputQualityPolicy` and returns a structured dict with six signal groups (or `None` if the record has no chunks).

**Relevance scoring.** When an `embedding_fn` callable is provided, each chunk's content is embedded alongside the query, and relevance is computed as cosine similarity between the two vectors. The cosine similarity function is implemented with stdlib `math` only — no numpy:

```python
def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)
```

When `embedding_fn` is None, relevance falls back to the scores already on `ChunkRecord` — `rerank_score` if available, else `retrieval_score`. This means a pipeline that already has a reranker gets relevance scoring for free without running an additional embedding model.

**Duplicate detection.** Path and window duplicates use the same logic as the ragradar analyzer. Semantic duplicates are detected only when `embedding_fn` is provided: chunk pairs from different source documents with cosine similarity above 0.92 are flagged.

**Truncation, token efficiency, coherence.** Same signals as the ragradar analyzers — truncation count and severity, headroom as a percentage of total limit, low-score chunk ratio, source domain count, and rerank score variance.

**Cache risk.** Not part of the `score_input_quality()` dispatcher above — it keys off `record.cache` instead of `record.chunks` and is gated separately in `evaluate()`, returning `None` (not a zero-value result) for a run that never checked a semantic cache. Flags a `borderline_hit` when a cache hit's similarity score landed within `cache_borderline_margin` of the cache's own threshold, and a `stale_hit` when a hit's `cached_at` is older than `cache_max_age_seconds`.

**Filter risk.** Also outside the chunks-based dispatcher — keys off `record.filter`, `None` for a run that never applied a metadata filter or didn't report `candidate_count`/`excluded_count` (a `0.0` ratio would misrepresent unknown data as "nothing excluded"). Computes `filtered_exclusion_ratio`, the share of the candidate pool excluded before retrieval/scoring ever saw it. Advisory-only: surfaced through `_CHECK_FACTORS` and `check_policy_violations()` but deliberately not folded into `compute_risk_score()`'s weighted sum, so it can't silently reweight the existing six signals below.

**Score degeneracy.** A chunk-score variance check — rerank score, falling back to retrieval score, across the run's chunks (chunks with neither score excluded). Near-zero variance means the retriever isn't discriminating between chunks at all, which points at a structural failure (a broken embedding or degraded index) rather than a normal relevance judgment call. Undefined (`None`) with fewer than two chunks carrying a usable score. Tracked under `chunk_score_variance` — deliberately not `score_variance`, which is already used by the coherence signal above (rerank-only, no fallback) and would otherwise collide in the merged output dict.

**Score margin.** Two margin diagnostics around the top chunk's score, bundled into one function, `score_score_margin()`, rather than shipped as two competing factors. Like `score_cache_risk`, it takes the active policy directly — `threshold_margin` needs `min_top_chunk_score` mid-computation, not just a final compare — and is therefore gated separately in `evaluate()` rather than run by the `score_input_quality()` dispatcher above. `top_second_margin` (top chunk score minus the runner-up's, same rerank-falling-back-to-retrieval convention as score degeneracy) is THE CHECKED FACTOR: a thin margin means the retriever isn't decisively ahead on its top pick. `threshold_margin` (top chunk score minus `min_top_chunk_score`) is diagnostic-only — it rides on the boundary the `top_chunk_score` factor already owns, so it deliberately has no policy field and no `_CHECK_FACTORS` entry of its own; it is not itself a checked factor, just context alongside `top_second_margin`. Returns `None` (not a partial dict) with fewer than two chunks carrying a usable score.

**Score underfill.** `score_score_underfill()` is the first input-quality factor built on a genuine new capture surface rather than a pure derivation over existing fields — see §3's note on `RunRecord.requested_chunk_count`. It is policy-free (no policy argument, unlike score margin/cache risk) but, like cache risk/filter risk, keys off data that may never have been captured, so it too is gated separately in `evaluate()` rather than run by the dispatcher. `underfill_ratio` (`(requested_chunk_count - len(chunks)) / requested_chunk_count`) is THE CHECKED FACTOR — negative when retrieval returned more than was asked for (never a violation), positive when it returned fewer. `requested_chunk_count`/`returned_chunk_count` ride alongside as diagnostic-only counts, the same primary-value-plus-counts shape as filter risk's `filtered_exclusion_ratio` plus its candidate/excluded counts. Returns `None` when `requested_chunk_count` was never captured, isn't positive, or `chunks` is `None`.

**Policy violations.** Each signal is checked against the active `InputQualityPolicy`. Violations are collected as a list of field names (e.g., `["max_high_score_truncations", "min_token_headroom", "max_source_domains"]`). The `passes_policy` boolean is `True` only when the violations list is empty.

### Layer 2 — output_quality.py

Takes a `RunRecord` and an optional `ground_truth` string. Returns `None` if chunks or response are missing. Uses RAGAS to compute four metrics: `faithfulness`, `answer_relevancy`, `context_precision`, and `context_recall` (the last requires `ground_truth`).

The RAGAS import is inside the function body:

```python
try:
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision
    from datasets import Dataset
except ImportError:
    raise ImportError("RAGAS not installed. Run: pip install ragas")
```

If individual metrics fail (RAGAS API changes between versions), the function returns partial results with `None` for the failed metrics rather than raising. This ensures forward compatibility across RAGAS releases.

### Policy system

The `InputQualityPolicy` dataclass encodes known failure mode thresholds with defaults derived from retrieval engineering literature and observed failure patterns:

```python
@dataclass
class InputQualityPolicy:
    min_chunk_relevance_score:  float = 0.5   # below this, chunk is noise
    min_top_chunk_score:        float = 0.7   # best chunk should be clearly relevant
    max_duplicate_ratio:        float = 0.2   # >20% duplication wastes budget
    max_low_score_chunk_ratio:  float = 0.3   # >30% low-score chunks dilutes context
    min_token_headroom:         float = 0.15  # <15% headroom risks truncation
    max_high_score_truncations: int   = 0     # any high-score truncation is a problem
    max_source_domains:         int   = 3     # >3 sources fragments coherence
    llm_rewrite_risk_threshold: float = 0.7   # gates future ragradar-improve rewrite stage
    cache_borderline_margin:    float = 0.03  # similarity this close to threshold is a near-miss
    cache_max_age_seconds:      int   = 86400 # cached answer older than this is stale
    max_filtered_exclusion_ratio: float = 0.3 # >30% excluded by a metadata filter is opaque
    min_score_variance:         float = 0.0001 # below this, chunk scores aren't discriminating
    min_top_second_margin:      float = 0.05  # top chunk should lead the runner-up by this much
    max_underfill_ratio:        float = 0.2   # >20% short of the requested chunk count is a problem
```

Policies are stored per-pipeline in the `policies` table as JSON. `load_policy(pipeline)` returns the stored policy or falls back to `InputQualityPolicy.default()` if none is set. `save_policy` writes a policy, `reset_policy` deletes it (reverting to defaults). The CLI exposes `ragradar-evaluate policy show`, `policy set <field> <value>`, and `policy reset`.

### Risk score

A single 0.0-1.0 score computed from six input quality signals against the active policy. Each signal has a fixed weight, and the risk score is the sum of weights for violated signals:

| Signal | Condition (violated when) | Weight |
|---|---|---|
| `high_score_truncations` | > `max_high_score_truncations` | 0.30 |
| `top_chunk_score` | < `min_top_chunk_score` | 0.25 |
| `duplicate_ratio` | > `max_duplicate_ratio` | 0.15 |
| `token_headroom_pct` | < `min_token_headroom` | 0.15 |
| `source_domain_count` | > `max_source_domains` | 0.10 |
| `low_score_chunk_ratio` | > `max_low_score_chunk_ratio` | 0.05 |

Weights sum to 1.0. High-score truncation and top chunk score carry the most weight because these are the clearest indicators of retrievable failure — the pipeline had relevant content and lost it.

If a signal is missing from the input scores (value is `None`), that check is skipped entirely and its weight does not contribute. This means a minimally instrumented pipeline (no token budget, no cache events) still gets a meaningful risk score from whatever data it did capture.

The weights are not configurable in v1 but are structured as an optional parameter (`weights: dict = None`) for future extensibility.

### Benchmark system

The benchmark correlates input quality factors against RAGAS output metrics across accumulated evaluated runs to discover which mechanical failures actually predict bad LLM output for a specific pipeline.

**Build** (`benchmark/builder.py`): Requires a minimum of 10 evaluated runs with both input and output scores. For each of nine input factors (`duplicate_ratio`, `top_chunk_score`, `high_score_truncations`, `token_headroom_pct`, `source_domain_count`, `low_score_chunk_ratio`, `mean_relevance`, `truncated_count`, `score_variance`), computes Pearson correlation against `faithfulness` and `answer_relevancy` using `scipy.stats.pearsonr`. Factors with fewer than 3 data points are skipped.

**Threshold suggestion**: For each factor, a suggested threshold is computed via binary search over the observed value range. The threshold that maximises the difference in mean RAGAS scores between runs above vs. below it is selected. This finds the factor value that best separates good outputs from bad ones, pipeline-specifically.

**Seeder** (`benchmark/seeder.py`): When no evaluated runs exist yet, the seeder generates synthetic run records as a day-zero baseline. Half are known-good profiles (high scores, no truncation, low domain count) and half are known-bad (low scores, high truncation, fragmented sources). Seeded runs are tagged with an internal reserved pipeline suffix to distinguish them from real data. They do not include RAGAS output scores — they provide input quality distribution only.

**Checker** (`benchmark/checker.py`): Loads a run's input quality scores and compares each factor against the benchmark threshold. Returns per-factor status (`ok` or `fail`) and an overall assessment: `ok` if all factors pass, `warn` if 1-2 factors fail, `fail` if 3+ factors fail or the risk score exceeds 0.7.

**Exporter** (`benchmark/exporter.py`): Writes all evaluated runs as a RAGAS-compatible JSONL dataset, one record per line with `question`, `answer`, `contexts`, `ground_truth`, `run_id`, `pipeline`, and `evaluated_at` fields. Seeded runs (carrying the internal reserved suffix) are excluded. This enables ragradar to feed accumulated data back into RAGAS for external analysis or model fine-tuning.

### RAGAS positioning

ragradar does not replace RAGAS — it composes with it. RAGAS measures output quality (was the response faithful to the context?). ragradar measures input quality (was the context worth being faithful to?). The benchmark system connects the two: by correlating ragradar's input factors against RAGAS's output scores, a developer discovers which mechanical failures in their specific pipeline actually predict bad outputs. The export command produces RAGAS-compatible datasets, closing the loop.

---

## 9. Build order and delivery state

### Phase 1 — ragradar_capture + ragradar

Delivered as a single release. ragradar-capture (the instrumentation SDK) and ragradar (the analyst CLI) share a store contract and were developed together. Their shared schema and store were later extracted into the `ragradar-core` kernel, which owns them (and their tests — migration, FTS5 triggers, persistence primitives, the zero-dependency guardrail) today. The full workspace suite currently stands at 290 tests across the four packages; run `uv run pytest` from the repo root for the authoritative count.

Packages: `ragradar-core` v0.1.0, `ragradar-capture` v0.1.0, `ragradar` v0.1.0.

### Phase 2 — ragradar-evaluate

Delivered after Phase 1 stabilized. Its suite covers the task-level facade (metric discovery, atomic selection, unified error channel, save semantics, six end-to-end user stories), input quality scoring (relevance, duplicates, truncation, policy violations, cosine similarity), risk score computation (zero/partial/full violation, missing signals), policy persistence (save/load/reset, unknown key handling), benchmark operations (minimum run requirement, correlation computation, seeder, exporter, checker), and CLI commands (store setup on every command, input-only mode, policy show/set/reset, benchmark seed/build/export).

Package: `ragradar-evaluate` v0.2.0.

### Phase 3 — ragradar-improve

Deferred. No implementation exists. No scope will be defined until ragradar-evaluate's benchmark system has accumulated real pipeline data — the benchmark thresholds need to be grounded in observed correlations, not assumptions.

When scoped, ragradar-improve will act on risk scores and benchmark findings to improve context quality before the LLM call. The planned architecture has three stages:

**Filter** (rules + SLM): Remove chunks that fall below the benchmark threshold for their factor. Rule-based filtering applies immediately (e.g., remove chunks with rerank score below 0.3). SLM-based filtering uses a small language model to evaluate relevance more precisely than score thresholds alone.

**Rerank** (SLM): Re-order the surviving chunks using a small cross-encoder model, independent of the pipeline's own reranker. This addresses cases where the pipeline's reranker is miscalibrated or absent.

**Rewrite** (LLM, opt-in): For runs where the risk score exceeds `llm_rewrite_risk_threshold` (default 0.7), optionally rewrite the context window using a full LLM call before sending it to the primary model. This is the most expensive stage and is explicitly opt-in — it adds an LLM call before the LLM call, which is only justified when the input quality is bad enough that the primary call is likely to fail anyway.

All three stages consume the risk score computed by ragradar-evaluate. The risk score gates the rewrite stage, and the benchmark thresholds gate the filter stage. This creates a direct feedback loop from evaluation to improvement, grounded in pipeline-specific data rather than generic rules.
