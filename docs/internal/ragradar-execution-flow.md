# ragradar — Execution Flow (method-by-method)

> **[PARTLY OUTDATED — written before the ragradar-core extraction, the
> Capture rename, and the task-level evaluation facade. See the README
> files and `docs/ARCHITECTURE.md` for current truth.]**
> What changed since this trace was written:
> - `schema.py` and the entire store (connection, schema, migrations,
>   persistence primitives) moved to the new `ragradar_core` package. Every
>   `ragradar_capture.store` / `ragradar_evaluate.store` file:line reference below
>   now lives in `ragradar_core/store.py`, and `ragradar_core.store.connect()`
>   creates/migrates the database on every connect — fresh DBs are born
>   at the latest schema version (FTS5 included).
> - `class Run` is now `class Capture`; `run.X()` reads as `cap.X()`;
>   `commit()`/`response()`/`capture()` return the run's sNrN id;
>   `capture()` has an explicit keyword signature (no `**kwargs`); strict
>   mode (`set_strict(True)` / `RAGRADAR_CAPTURE_STRICT=1`) makes errors raise.
> - `ragradar_evaluate`'s public API is `check()` / `evaluate()` /
>   `available_metrics()`; the CLI routes through `evaluate()`; benchmark
>   machinery is internal. `ragradar.store.check_schema_version()` no longer
>   exists.
> Sections below are kept as an architectural narrative; their file:line
> references are stale where they touch the moved/renamed code.

This document traces exactly what code runs, in what order, for every entry
point in the user-facing packages. It complements `ragradar-design-doc.md` (rationale,
data model, architecture) and `ragradar-scope.md` (delivery scope) — this one is
pure call-flow, file:line references included so it stays checkable against
the source.

Three independent packages share one SQLite file at `~/.ragradar/runs.db`:

```
ragradar-capture (import ragradar_capture)   → writes runs.db
ragradar         (import ragradar)      → reads runs.db, renders analysis
ragradar-evaluate (ragradar_evaluate)   → reads + augments runs.db with eval columns
        └── ragradar-core (import ragradar_core) → owns schema + store for all three
```

No package imports another's CLI. All three import `ragradar_core` for the
`RunRecord` dataclasses and the store — that kernel is the only real
coupling between them.

---

## 1. Data model (recap)

Everything funnels into one dataclass, `RunRecord`
(`packages/ragradar-capture/src/ragradar_capture/schema.py:69`):

```
RunRecord
├── query, response          (required)
├── chunks: [ChunkRecord]    chunk_id, source_doc_id, content, token_count,
│                            retrieval_score, rerank_score, retrieval_path,
│                            truncated, cache_hit
├── final_prompt: str
├── token_budget: TokenBudget   total_limit, chunks_allocated,
│                                history_allocated, system_allocated, headroom
├── history_pre / history_post: [Turn]   role, content, tokens
├── eviction_reason: str
├── cache_events: [CacheEvent]   chunk_id, hit, cache_source
├── model: str
└── token_usage: TokenUsage      input_tokens, output_tokens, total_tokens
```

All dataclasses are decorated with `_flexible` (`schema.py:6`), which wraps
`__init__` to silently drop unknown kwargs. This is why instrumentation
never crashes a caller's pipeline for passing extra fields — every field
except `query`/`response` is optional, and unknown ones are ignored rather
than raising `TypeError`.

`RunRecord.to_json()` → `dataclasses.asdict()`. `RunRecord.from_json()`
manually reconstructs each nested dataclass list because `asdict`/plain
`dict` round-tripping loses the class information.

---

## 2. ragradar-capture — instrumentation SDK (`import ragradar_capture`)

### 2.1 One-liner path: `ragradar_capture.capture(query, response, **kwargs)`

`api.py:131`

1. Pop `pipeline` out of kwargs.
2. Build a `Run(query, pipeline)` — this immediately creates an empty
   `RunRecord(query=query, response="")` (`api.py:36`).
3. Set `run._record.response = response`.
4. For every optional kwarg present (`chunks`, `final_prompt`,
   `history_pre`/`history_post`, `eviction_reason`, `cache_events`, `model`,
   `token_usage`), call the matching `Run` method or set the field directly.
5. Call `run.commit()`.
6. The **entire body is wrapped in `try/except Exception`** — any failure
   (bad kwarg types, etc.) is swallowed and logged to
   `~/.ragradar/errors.log` via `_get_logger()` (`api.py:15`), never raised to
   the caller's pipeline.

### 2.2 Staged path: `ragradar_capture.start()` → `run.X()` → `run.commit()`

```
run = ragradar_capture.start(query=query, pipeline="my_project")   # api.py:125
  └── Run(query, pipeline)                                # api.py:33
  └── set_active_run(run)                                 # thread_local.py:6
run.chunks(chunks)          # api.py:39  → coerces each item to ChunkRecord
run.context(prompt, budget) # api.py:51  → sets final_prompt + TokenBudget
run.history(pre, post, r)   # api.py:66  → sets history_pre/post + eviction_reason
run.response(text, usage, model)  # api.py:81
  └── sets response, model, token_usage
  └── calls self.commit()          ← auto-commit on response()
run.cache(events)           # api.py:98  → can be called any time before commit
```

Each method:
- coerces raw dicts to the matching dataclass only if not already an
  instance (`c if isinstance(c, ChunkRecord) else ChunkRecord(**c)`),
- is wrapped in its own `try/except`, logging and swallowing failures
  independently — a broken `run.chunks()` call does not prevent
  `run.history()` or `run.response()` from still capturing data.

`run.commit()` (`api.py:110`):
1. No-ops if already committed (`self._committed`).
2. `store.get_or_create_session(pipeline)` — see §2.4.
3. `store.next_run_seq(session_id)` — `MAX(run_seq)+1` for that session.
4. `store.write_run(...)` — INSERT into `runs` with `run_data` as
   `json.dumps(record.to_json())`.
5. Marks `_committed = True` so a second `run.commit()` (or the
   auto-commit inside `response()` plus a manual call) is a silent no-op.

### 2.3 Thread-local proxy functions

`ragradar_capture.chunks()`, `.context()`, `.history()`, `.response()`, `.cache()`,
`.commit()` (module-level, `api.py:168` onward) are free functions that:
1. `get_active_run()` from `thread_local.py` (a `threading.local`).
2. If `None`, log an error ("called with no active run") and return —
   never raises.
3. Otherwise delegate to the corresponding `Run` method.

This lets code deep in a call stack (e.g. a reranker module) call
`ragradar_capture.cache(events)` without threading a `run` object through every
function signature, as long as it executes on the same thread that called
`ragradar_capture.start()`.

### 2.4 Session auto-creation — `store.get_or_create_session()`

`packages/ragradar-capture/src/ragradar_capture/store.py:65`

1. `init_store()` first — idempotently runs `SCHEMA` (CREATE TABLE IF NOT
   EXISTS for `meta`, `sessions`, `runs`) and seeds `schema_version = "1"`
   if `meta` is empty.
2. Look up the most recent session for this `pipeline` (or the most recent
   session with `pipeline IS NULL` if none given).
3. If found, check the last run's (or session's) `created_at` against
   `idle_gap_minutes` (default 30). If the gap is under 30 minutes, **reuse
   that session_id**.
4. Otherwise INSERT a new row into `sessions` and return the new
   `session_id`.

This is the mechanism the example (`examples/rag_pipeline/run_pipeline.py`)
exploits deliberately: it runs 4 queries, rewrites their timestamps 31
minutes into the past, then runs 4 more — forcing a second session to be
created on the next `get_or_create_session()` call.

### 2.5 Scaffold generator — `ragradar-capture init`

`scaffold/cli.py:6` → `scaffold/template.py:40`
1. `generate_scaffold()` refuses to overwrite an existing
   `ctx_pipeline.py` (raises `FileExistsError`).
2. Otherwise writes the hardcoded `TEMPLATE` string — a function skeleton
   with `ragradar_capture.start()` / `run.chunks()` / `run.context()` / `run.history()`
   / `run.response()` calls pre-positioned as comments for the user to
   uncomment and fill in.

---

## 3. ragradar — analyst CLI

Entry point: `ragradar.cli:main`, a `click.Group`. Every subcommand first
triggers `main()`'s body: `store.check_schema_version()`
(`cli.py:68`-`70`), which warns to stderr (not fatal) if `~/.ragradar/runs.db`'s
`meta.schema_version` is older than `EXPECTED_SCHEMA_VERSION = "1"`.

### 3.1 Target resolution — the shared primitive

Almost every command resolves a "target" string to a run row. Two
resolvers exist:

**`store.resolve_target(target)`** (`store.py:166`) — used by `explain`,
`diff`, `budget`:
```
target is None          → get_latest_run()  (MAX created_at across all runs)
target matches s(\d+)r(\d+)  → get_run(session_id, run_seq)   (exact lookup)
else                     → search_runs(hint=target)
                             0 results → None
                             1 result  → get_run(...) for it
                             >1 results → sort by find/bm25.score(target, query)
                                          descending, return the LIST
                                          (caller must disambiguate)
```

**`cli._resolve_and_load(target)`** (`cli.py:53`) wraps this: if
`resolve_target` returns a list, it calls `_disambiguate()` (`cli.py:25`),
which prints a numbered table and prompts interactively via
`click.prompt`; picking an index re-fetches the exact run via
`store.get_run`. Ctrl-C/EOF/empty input cancels cleanly (returns `None`).
Once a single row is settled, `loader.load_run_record(run_row)`
(`explain/loader.py:6`) does `RunRecord.from_json(json.loads(row["run_data"]))`.

### 3.2 `ragradar list [session_id]`

`cli.py:73`
- No arg → `store.list_sessions()`: LEFT JOIN `sessions`/`runs`, grouped,
  `COUNT(run_seq)` per session, newest first. Rendered as a Rich `Table`.
- With `sN` or a bare int → `store.list_runs(sid)`: all runs in that
  session, newest first.

### 3.3 `ragradar find <hint> [--exact] [--from] [--to] [--today] [--session] [--pipeline] [--recent N]`

`cli.py:116` → `store.search_runs(...)` → `find/query_builder.build_search_query()`

`query_builder.py:1` builds one parameterized SQL statement:
- Base: `runs JOIN sessions` (so `session_title` is available).
- Hint clause depends on FTS5 availability
  (`store._has_fts5()` checks for a `runs_fts` virtual table, created by
  the ragradar-evaluate schema-v2→v3 migration — see §4.2):
  - **FTS5 present + `--exact`**: `MATCH '"<hint>"'` (phrase match).
  - **FTS5 present, no `--exact`**: `MATCH '"t1" OR "t2" OR ...'` (any
    token).
  - **FTS5 absent + `--exact`**: `query LIKE '%hint%'`.
  - **FTS5 absent, no `--exact`**: `AND`-ed set of `query LIKE '%token%'`
    clauses (OR'd together) — a plain substring scan, used only when the
    `ragradar-evaluate` migration hasn't run yet on this database.
- Optional `session_id`, `pipeline`, `created_at >= from_dt`,
  `created_at <= to_dt` clauses are appended.
- `--today` (`cli.py:127`) is sugar: sets `from_dt`/`to_dt` to today's date
  bounds before calling `search_runs`.
- `ORDER BY created_at DESC`, optional `LIMIT` for `--recent`.

Results render as a table; no disambiguation step (this command *shows*
multiple matches by design, unlike `resolve_target`).

### 3.4 `ragradar explain [target] [--full] [--html]`

`cli.py:167`
1. `_resolve_and_load(target)` → `(run_row, record)`.
2. If `--html`: `html_renderer.render(record, run_id)` — see §3.6.
3. Else: `terminal_renderer.render(record, full=full, run_row=run_row)` —
   see §3.5.

### 3.5 Terminal renderer — `explain/renderer/terminal.py`

`render()` (`terminal.py:207`) is the orchestrator:
1. Print query, response (truncated to 200 chars unless `--full`), model.
2. Iterate `_ANALYZERS` — a fixed list of `(module, render_fn)` pairs in a
   fixed order: **tokens → scores → duplicates → truncation → history →
   cache** (`terminal.py:152`). For each, call `mod.analyze(record)`; if it
   returns `None` (insufficient data), **skip silently** — nothing is
   printed for that factor. This is the "seven factors, silently skipped"
   behavior the README describes.
3. If `run_row` has `eval_scores` (populated by `ragradar-evaluate run`),
   render an extra "Evaluation Scores" panel (`_render_eval_scores`,
   `terminal.py:162`) — risk score, input-quality violations,
   RAGAS metrics if present.
4. Print the final assembled prompt (truncated to 500 chars unless
   `--full`).

Each analyzer module in `explain/analyzers/` is a pure function
`analyze(record) -> dict | None`:

| Module | Returns `None` when | Computes |
|---|---|---|
| `tokens.py` | no chunks AND no final_prompt | sum of chunk token_counts + history_post tokens + system_allocated; utilisation % against `token_budget.total_limit`; per-chunk token breakdown |
| `scores.py` | no chunks | min/max retrieval & rerank scores; `rerank_delta` = mean(rerank) − mean(retrieval); `low_score_ratio` = fraction of rerank scores < 0.5 |
| `duplicates.py` | no chunks | **path dups**: same `chunk_id` seen via >1 distinct `retrieval_path`; **window dups**: chunks sharing `source_doc_id` where one's `content` is a substring of another's (pairwise, O(n²) within each source group); `semantic_dups` always `[]` (deferred — no embedding model in the free `ragradar` package) |
| `truncation.py` | no chunks | chunks with `truncated=True`; `severity`: `"none"` if none truncated, `"high"` if any truncated chunk has `retrieval_score>0.7` or `rerank_score>0.7`, else `"low"` |
| `history.py` | no history_pre AND no history_post | `dropped` = turns present in `pre` but whose `(role, content)` tuple is absent from `post`'s set; sums pre/post tokens |
| `cache.py` | no `cache_events` | hit/miss counts, `hit_ratio`, lists of hit/miss chunk_ids |

Each `_render_X` function in `terminal.py` picks a Rich `Panel`
border color from thresholds (e.g. token utilisation <80% green, <95%
yellow, else red; duplicate ratio 0 green, ≤20% yellow, else red) and,
when `--full`, appends per-item detail lines.

`render_budget()` (`terminal.py:238`) is just `tokens_mod.analyze()` +
`_render_tokens(..., full=True)` — used by `ragradar budget <target>`.

`render_diff()` (`terminal.py:247`) — used by `ragradar diff`:
- Query side-by-side table.
- Chunk set difference (`chunks_b - chunks_a` = added, vice versa =
  removed) plus counts.
- Score deltas for the **intersection** of chunk_ids (retrieval + rerank,
  per run).
- Token budget attribute-by-attribute table (`total_limit`,
  `chunks_allocated`, `history_allocated`, `system_allocated`,
  `headroom`).
- History pre/post/dropped counts.
- Truncation count + severity via `truncation_mod.analyze()` on both
  records.
- Every section is conditionally printed only if relevant data exists on
  at least one side.

### 3.6 HTML renderer — `explain/renderer/html.py`

`render(record, run_id)` (`html.py:31`) mirrors the terminal renderer's
analyzer loop but emits `<details><summary>...<pre>...</pre></details>`
blocks instead of Rich panels, string-escaping all content (`_esc`,
`html.py:15`). Writes to `~/.ragradar/reports/<run_id>.html` (creating the
`reports/` dir) and returns the `Path`. No eval-scores section here
(HTML report is generated from `record` alone, not `run_row`).

### 3.7 `ragradar diff <target_a> <target_b>`

`cli.py:185` — both targets resolved via `store.resolve_target` directly
(not `_resolve_and_load`, so **no interactive disambiguation**: if either
resolves to a list, the command just prints "Ambiguous target — use exact
run ID" and exits). Otherwise loads both records and calls
`terminal_renderer.render_diff`.

### 3.8 `ragradar budget <target>`

`cli.py:209` — `_resolve_and_load` then `terminal_renderer.render_budget`.

### 3.9 `ragradar session rename <id> <title>`

`cli.py:224` → `store.rename_session()` — a plain `UPDATE sessions SET
title = ? WHERE session_id = ?`.

---

## 4. ragradar-evaluate — evaluation layer (`ragradar_evaluate`)

Entry point: `ragradar_evaluate.cli:main`. Its group callback runs
`store.apply_migration()` (`cli.py:143`-`146`) **on every invocation** —
this is how a `runs.db` created by `ragradar-capture` alone (schema v1) gets
upgraded in place the first time `ragradar-evaluate` touches it.

### 4.1 Migration chain — `store.apply_migration()`

`packages/ragradar-evaluate/src/ragradar_evaluate/store.py:29`

```
version None/absent  → not reachable in practice (ragradar-capture always seeds "1")
version "1" → add eval_scores/risk_score/evaluated_at columns to runs;
              create benchmark table (pipeline, factor, threshold,
              correlation, sample_count, updated_at);
              create policies table (pipeline, policy_data, updated_at);
              meta.schema_version = "2"
version "2" → create FTS5 virtual table runs_fts(query) content-linked to
              runs.rowid; rebuild it; add INSERT/DELETE/UPDATE triggers to
              keep it in sync; drop the now-redundant idx_runs_query;
              meta.schema_version = "3"
version "3" → no-op, already current
anything else → raise RuntimeError("Unsupported schema version")
```

Each step commits before falling through to the next, so a v1 DB walks
v1→v2→v3 in one call. This migration is what makes FTS5 search
(`ragradar find`, §3.3) available — `ragradar` alone never creates `runs_fts`.

### 4.2 `ragradar-evaluate run [target] [--input-only] [--output-only] [--session] [--ground-truth] [--pipeline]`

Both flags together (`--input-only --output-only`) is now a clean error
(exit 1) — it used to silently compute nothing.

Two paths, both routed through the public `ragradar_evaluate.evaluate()`
facade (the CLI no longer has its own scoring sequence):

**Single run**:
1. `_resolve_target(target)` — the `sNrN`/latest resolver (no fuzzy
   search or disambiguation here — evaluation only accepts an exact
   `sNrN` or "latest").
2. `evaluate(run_id, metrics=<flag-mapped set>, save=True)` — loads the
   pipeline's policy, runs the selected input metric families, attempts
   the selected RAGAS metrics (judge failures land in
   `EvalResult.errors` and print as a yellow warning), computes the risk
   score (`None` when no input metrics were computed), and persists
   `eval_scores` (JSON) + `risk_score` + `evaluated_at` on the run row.
3. `_render_eval_result` prints a risk-colored header, an "Input Quality"
   table (only the factors present), policy violations, and a "Output
   Quality (RAGAS)" table if present.

**Session batch** (`--session sN`): loops `store.get_runs_in_session(sid)`,
caches `load_policy()` per distinct pipeline key inside the loop
(`policy_cache` dict) to avoid re-hitting the `policies` table per run,
computes each run's eval via `evaluate(..., save=False)` (no DB write
yet), then writes **all** results in one transaction via
`store.write_eval_scores_batch`, then renders each.

### 4.3 Layer 1 — `layers/input_quality.py::score()` (deterministic, no LLM)

`input_quality.py:72`. Returns `None` if `record.chunks` is empty.
Otherwise, in order:

1. **Relevance**: if an `embedding_fn` is supplied (semantic-search extra
   — not wired into the CLI by default, present for programmatic use),
   compute cosine similarity between query and each chunk's embedding.
   Otherwise fall back to each chunk's `rerank_score`, then
   `retrieval_score`. `mean_relevance` = average of whatever was
   collected. `top_chunk_score` = max `rerank_score` (only from actual
   rerank scores, not the relevance fallback list).
2. **Duplicates**: `_detect_path_dups` (chunk_id repeated across ≥2
   distinct `retrieval_path`s) + `_detect_window_dups` (pairwise within a
   `source_doc_id` group: substring containment **or** token-set Jaccard
   overlap >50% — this is a stricter/different check than `ragradar explain`'s
   analyzer, which only does substring containment) +
   `_detect_semantic_dups` (pairwise cosine similarity > `0.92` across
   *different* `source_doc_id`s, only runs if `embedding_fn` given).
   `duplicate_ratio` = `(path_dup_count + window_dup_count) / total_chunks`
   (semantic dups counted separately, not folded into the ratio).
3. **Truncation**: same logic as `ragradar explain`'s `truncation.py` analyzer
   (severity none/low/high by score>0.7 threshold), duplicated here so
   `ragradar-evaluate` has no runtime dependency on the `ragradar` package.
4. **Token efficiency**: `token_headroom_pct = headroom / total_limit`;
   `low_score_chunk_ratio` = fraction of chunks whose rerank score (or
   retrieval score if no rerank) is < 0.5.
5. **Coherence**: `source_domain_count` = distinct `source_doc_id` count;
   `score_variance` = population variance of rerank scores (needs ≥2).
6. **Policy violations** — compares each computed value against the
   active `InputQualityPolicy` (see §4.6) and appends the field name to
   `violations` for every threshold breached. `passes_policy = not
   violations`.

Returns one flat dict with every intermediate value plus
`policy_violations`/`passes_policy` — this dict is both what's rendered
and what `benchmark/builder.py` later correlates against RAGAS scores.

### 4.4 Layer 2 — `layers/output_quality.py::score()` (RAGAS, LLM-as-judge)

`output_quality.py:4`. Returns `None` if no chunks or no response.
Lazily imports `ragas`/`datasets` inside the function — raises a
descriptive `ImportError` if not installed (caught by the CLI, §4.2).
Builds a single-row HF `Dataset` (`question`, `answer`, `contexts`, and
`ground_truth` if supplied — which also conditionally adds the
`context_recall` metric to the metrics list). Calls `ragas.evaluate()`
and unpacks `faithfulness`, `answer_relevancy`, `context_precision`,
`context_recall` into a flat dict. Any exception from `ragas.evaluate()`
itself (not import) is caught and returned as
`{"...": None, "evaluator": "ragas", "model": "error", "error": str(e)}`
rather than propagating — a broken/unreachable RAGAS backend never kills
the CLI command.

### 4.5 Risk score — `policy/risk.py::compute_risk_score()`

`risk.py:13`. A fixed weighted sum over six factors
(`_DEFAULT_WEIGHTS`, summing to 1.0: duplicate_ratio 0.15, top_chunk_score
0.25, high_score_truncations 0.30, token_headroom_pct 0.15,
source_domain_count 0.10, low_score_chunk_ratio 0.05). For each factor,
if the input-quality value breaches the policy threshold, add that
factor's weight to `risk`. Result is a 0–1 score independent of the
`policy_violations` list (same threshold checks, different output shape —
one is a set of names, the other a weighted magnitude).

### 4.6 Policy system — `policy/schema.py`, `policy/store.py`

`InputQualityPolicy` (`schema.py:4`) is a plain dataclass of eight
thresholds with defaults (e.g. `max_duplicate_ratio=0.2`,
`min_top_chunk_score=0.7`, `max_high_score_truncations=0`). Stored
per-pipeline as JSON in the `policies` table (`pipeline` PK), created in
the v1→v2 migration.

```
load_policy(pipeline)   → store.get_policy(); None → InputQualityPolicy.default()
save_policy(pipeline,p) → store.write_policy(pipeline, p.to_dict())   (INSERT OR REPLACE)
reset_policy(pipeline)  → DELETE FROM policies WHERE pipeline = ?
```

CLI (`policy show|set|reset`, `cli.py:311`):
- `show`: loads the pipeline's policy (or default if unset), prints each
  field bolded if it differs from `InputQualityPolicy.default()`.
- `set <field> <value>`: validates `field` against
  `dataclasses.fields(InputQualityPolicy)`, coerces `value` using
  `typing.get_type_hints()[field]` (so `float` fields parse as float,
  `int` as int), loads-mutates-saves.
- `reset`: deletes the row, causing subsequent `load_policy` calls to fall
  back to `.default()`.

`pipeline` defaults to the literal string `"__default"` everywhere a
per-pipeline lookup key is needed and no `--pipeline`/run-derived pipeline
is available.

### 4.7 Benchmark system

Four independent commands operating on the `benchmark` table.

**`benchmark seed <pipeline> [--count N]`** → `seeder.py:7`
- Generates `count` synthetic `RunRecord`s (half via `_good_record`, half
  `_bad_record`) with hand-tuned scores designed to be clearly
  distinguishable (good: rerank ~0.90-0.98, low duplicate/truncation; bad:
  rerank ~0.25-0.40, half the chunks truncated, 6-8 distinct
  `source_doc_id`s).
- Writes them under an internal seeded-pipeline tag (the pipeline name
  plus a reserved suffix) via `ragradar_core.store.write_runs_batch` (a batch
  INSERT, bypassing the `Capture`/`commit()` API entirely since there's
  no live pipeline to instrument).
- These seeded runs have **no RAGAS scores** — they exist purely to give
  `benchmark build` an input-quality distribution before any real
  evaluated runs exist ("day-zero baseline"). `exporter.py` explicitly
  filters out seeded-tagged pipelines so they never leak into
  a RAGAS training export (`exporter.py:13`).

**`benchmark build [--pipeline]`** → `builder.py:46`
1. `store.get_all_evaluated_runs(pipeline)` — every run with non-null
   `eval_scores`. Raises `ValueError` if fewer than 10.
2. For each of 9 fixed `INPUT_FACTORS` (duplicate_ratio, top_chunk_score,
   high_score_truncations, token_headroom_pct, source_domain_count,
   low_score_chunk_ratio, mean_relevance, truncated_count,
   score_variance): collect `(factor_value, ragas_value)` pairs across
   runs that have **both** the factor and at least one of
   `RAGAS_METRICS = [faithfulness, answer_relevancy]`. Skip the factor if
   fewer than 3 samples.
3. `scipy.stats.pearsonr(factor_values, ragas_values)` per RAGAS metric
   (only if both lists have >1 distinct value, else `None`).
   `primary_corr` = whichever correlation has the largest absolute value.
4. `_suggest_threshold()` (`builder.py:22`): brute-force scan over
   midpoints between consecutive sorted unique factor values; for each
   candidate threshold, split samples into ≤threshold / >threshold, take
   the absolute difference of mean RAGAS score (using
   `RAGAS_METRICS[0]` = faithfulness) between the two groups; keep the
   threshold that maximizes this gap. This is a simple 1D decision-stump
   search, not a formal statistical test.
5. `store.write_benchmark_entries_batch(...)` — `INSERT OR REPLACE` one
   row per factor into `benchmark`, keyed `(pipeline, factor)`.
6. Prints a table of threshold/correlation/sample-count per factor.

**`benchmark show [--pipeline]`** → straight `SELECT * FROM benchmark
WHERE pipeline = ?`, rendered as a table.

**`benchmark check <target> [--pipeline]`** → `checker.py:9`
1. Load the run, its policy, compute fresh `input_quality.score()`.
2. Load the pipeline's `benchmark` rows into a `factor → row` map.
3. For six of the nine factors (`check_factors`, `checker.py:29` — a
   `(factor, direction)` list; `direction` is `"lower_bad"` for
   `top_chunk_score`/`token_headroom_pct`, `"higher_bad"` for the other
   four), compare the run's current value against the benchmark
   threshold: `fail` if it's on the wrong side, else `ok`. If no
   benchmark entry exists for a factor, status is unconditionally `ok`
   (nothing to check against).
4. Overall verdict: `fail` if `risk_score > 0.7` **or** ≥3 factors failed;
   `warn` if 1-2 failed; else `ok`. `risk_score` is read from the
   already-persisted `eval_scores` column (not recomputed) — so
   `benchmark check` requires the run to have been evaluated via
   `ragradar-evaluate run` first, or `risk` silently defaults to `0.0`.

**`benchmark export [--pipeline] [--output]`** → `exporter.py:8`
- Filters `get_all_evaluated_runs()` to drop seeded-tagged pipelines and
  runs missing `chunks`/`response`.
- Writes one JSON object per line (`question`, `answer`, `contexts`,
  `ground_truth: null`, plus `run_id`/`pipeline`/`evaluated_at`
  metadata) to `~/.ragradar/exports/<pipeline>_ragas_<timestamp>.jsonl` (or
  the given `--output` path) — a RAGAS-compatible dataset for offline
  reuse.

---

## 5. End-to-end trace (the shipped example)

`examples/rag_pipeline/run_pipeline.py` → `pipeline.py::run_pipeline()`
walks the full staged-capture API against 8 hardcoded queries:

```
ragradar_capture.start(query, pipeline="rag_example")
  → Run.__init__ creates RunRecord; set_active_run (unused here since the
    example holds `run` directly rather than using the thread-local proxies)

run.chunks(_build_chunks())        # 7 ChunkRecords, deliberately engineered:
                                    #   - rrf_norm_1/rrf_norm_2 share source_doc_id
                                    #     "rrf_paper_2024" and overlapping content
                                    #     → triggers duplicates.py window_dups
                                    #   - bm25_tf_idf: truncated=True, rerank=0.88
                                    #     → triggers truncation.py severity="high"
                                    #   - rrf_norm_1: cache_hit=True
                                    #   - score_calib/ctx_window: rerank 0.41/0.39
                                    #     → pull low_score_ratio up
                                    #   - 6 distinct source_doc_ids
                                    #     → exceeds default max_source_domains=3

run.context(final_prompt, TokenBudget(total=4096, headroom=196))
                                    # headroom/limit = 4.8% → tokens.py flags
                                    # low headroom; utilisation renders red

run.history(pre=[4 turns], post=[2 turns], reason="token_budget")
                                    # 2 turns dropped → history.py flags eviction

run.cache([7 CacheEvents, 1 hit])  # cache.py → hit_ratio = 1/7

run.response(text, token_usage=..., model="gpt-4-turbo")
  → sets response/model/token_usage, then self.commit()
      → store.get_or_create_session("rag_example")
      → store.next_run_seq(session_id)
      → store.write_run(...)  INSERT INTO runs
```

`run_pipeline.py` (the outer driver) calls this 4 times, backdates all
`rag_example` timestamps by 31 minutes directly via SQL
(`_backdate_pipeline_runs`), then calls it 4 more times — because
`get_or_create_session`'s 30-minute idle-gap check now sees a gap,
creating a second session. Result: `ragradar list` shows 2 sessions of 4 runs
each, and every `ragradar explain` on any of these runs lights up all seven
terminal-renderer panels because the fixtures were built specifically to
cross every analyzer's threshold.

---

## 6. Cross-cutting behaviors worth knowing

- **Fail-open instrumentation**: every public `ragradar_capture.api` function
  catches its own exceptions and logs to `~/.ragradar/errors.log`; nothing in
  the capture SDK can raise into a host pipeline. `ragradar` and `ragradar-evaluate`
  are not held to this standard — CLI errors there use
  `SystemExit(1)`/`ValueError` propagation deliberately, since they run
  interactively.
- **No `run.db` = empty results, not errors**: every store `_connect()`
  helper returns `None` if `~/.ragradar/runs.db` doesn't exist yet, and every
  caller treats `None` as "no data" (empty list / `None` row) rather than
  raising. First-run UX is "No runs found." rather than a traceback.
- **Schema versioning is package-local**: `ragradar.store` only ever reads
  `meta.schema_version` and warns if stale — it never writes migrations.
  `ragradar_capture.store` owns v1 (initial create). `ragradar_evaluate.store` owns
  the v1→v2→v3 migrations (eval columns + benchmark/policies tables, then
  FTS5). This means installing `ragradar-evaluate` and running any command
  against an existing capture-only DB is what unlocks FTS5 search for
  plain `ragradar find` too — a real cross-package coupling worth knowing about
  when debugging "why is `ragradar find` doing substring LIKE instead of FTS5."
  It's caused by `ragradar-evaluate` never having been run against that DB.
  See `store.check_schema_version()` (`ragradar/store.py:28`) — only warns,
  never migrates.
  See `ragradar_evaluate.store.apply_migration()` (§4.1) — the only writer.
  See `ragradar.find.query_builder.build_search_query()` (§3.3) — the only
  reader that branches on FTS5 presence.
- **Two independent duplicate-detection implementations**: `ragradar explain`'s
  `duplicates.py` (substring-only window check) and `ragradar-evaluate`'s
  `input_quality._detect_window_dups` (substring **or** >50% token
  Jaccard) can disagree on borderline cases. This is intentional
  package independence (`ragradar-evaluate` doesn't import `ragradar`), not a
  bug, but it means duplicate counts shown by `ragradar explain` and
  `ragradar-evaluate run` for the same run are not guaranteed to match.
- **`__default` pipeline key**: any command accepting `--pipeline` treats
  an omitted pipeline (and any run captured without an explicit
  `pipeline=` at `ragradar_capture.start()`/`capture()` time) as the literal string
  `"__default"` for policy/benchmark table lookups.
