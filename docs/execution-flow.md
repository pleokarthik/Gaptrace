# ragradar — Execution Flow (method-by-method)

> **Snapshot, not a pinned reference.** This traces the source tree as read
> on the date this file was written. Method bodies get refactored and line
> numbers drift; when a citation here looks off, it means the code moved,
> not that the narrative is wrong — re-check the named file before trusting
> a specific line number. It complements `design-doc.md` (rationale,
> data model, architecture) and `scope.md` (delivery scope) — this
> one is pure call-flow.

Four distributions share one SQLite file at `~/.ragradar/runs.db`:

```
ragradar-core     (import ragradar_core)     → schema + store + sNrN parser + coercion (zero deps)
ragradar-capture  (import ragradar_capture)  → writes runs.db (stdlib-only beyond ragradar_core)
ragradar          (import ragradar)          → reads runs.db, renders analysis; ALSO re-exports
                                                 ragradar_capture's and ragradar_evaluate's public
                                                 functions, so `import ragradar` alone is the
                                                 complete public surface (packages/ragradar/src/
                                                 ragradar/__init__.py:1-82)
ragradar-evaluate (import ragradar_evaluate) → reads + augments runs.db with eval columns
```

No package's **CLI** imports another's CLI. But the `ragradar` distribution's
`__init__.py` does import `ragradar_capture` and `ragradar_evaluate` as
*libraries* to re-export their functions — `ragradar.start`, `ragradar.capture`,
`ragradar.check`, `ragradar.evaluate`, `ragradar.available_metrics`, the schema
dataclasses, etc. are all the same objects those packages define, just
reachable through one import. All four packages ultimately depend on
`ragradar_core` for the `RunRecord` dataclasses, the store, the `sNrN` parser
(`targets.py`), and the primitive-to-dataclass coercion helpers (`coerce.py`)
— that kernel is the real coupling point.

---

## 1. Data model (recap)

Everything funnels into one dataclass, `RunRecord`
(`packages/ragradar-core/src/ragradar_core/schema.py:171`):

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
├── tool_calls: [ToolCallRecord]  tool_name, arguments, result, error, latency_ms
├── model: str
├── token_usage: TokenUsage      input_tokens, output_tokens, total_tokens
├── cache: CacheRecord           checked, hit, similarity_score, threshold,
│                                cached_query, cached_at, registered
└── filter: FilterRecord         applied, candidate_count, excluded_count, filters
```

`cache` and `filter` (`schema.py:132` and `schema.py:153`) are the newest
fields — added for the semantic-cache and metadata-filter features, after
`ToolCallRecord` (`schema.py:116`). Every one of the nine `ragradar.explain`
analyzers has now been checked: `cache` is rendered by
`explain/analyzers/semantic_cache.py` and `filter` by
`explain/analyzers/metadata_filter.py`, but `tool_calls` still isn't
referenced by any of the nine — it is captured and persisted (round-trips
through `to_json()`/`from_json()`) but not currently surfaced by `explain`.

All dataclasses are decorated with `_flexible` (`schema.py:15`), which wraps
`__init__` to silently drop unknown kwargs. This is why instrumentation
never crashes a caller's pipeline for passing extra fields — every field
except `query`/`response` is optional, and unknown ones are ignored rather
than raising `TypeError`.

`RunRecord.to_json()` (`schema.py:197`) is `dataclasses.asdict()`.
`RunRecord.from_json()` (`schema.py:202`) manually reconstructs each nested
dataclass list (including `tool_calls` → `ToolCallRecord`) because
`asdict`/plain `dict` round-tripping loses the class information.

`ragradar_core/coerce.py` is the shared primitive-input boundary both
`ragradar_capture` and `ragradar_evaluate` route user input through: bare
dicts, `("role", "content")` tuples, a bare-int token budget, a
`{chunk_id: hit}` cache mapping, all coerce into the dataclasses above.
Token counts default to a deterministic ~4-chars-per-token estimate
(`coerce.py:37`, `estimate_tokens`) when not given explicitly. Every
coercer (`coerce_chunk`/`coerce_chunks` at `coerce.py:88`/`109`,
`coerce_token_budget` at `coerce.py:114`, `coerce_turn`/`coerce_turns` at
`coerce.py:48`/`83`, `coerce_cache_events` at `coerce.py:148`,
`coerce_cache_record`/`coerce_filter_record` at `coerce.py:170`/`183`,
`coerce_token_usage` at `coerce.py:196`, `coerce_tool_call` at
`coerce.py:212`) passes dataclass instances through untouched and only
converts primitives. `coerce_run_record` (`coerce.py:221`) coerces every
nested field of a whole hand-built `RunRecord` at once — this is what lets
`ragradar_evaluate.evaluate()`/`check()` accept a bare `RunRecord` built with
plain dicts, not just an `sNrN` id.

---

## 2. ragradar-capture — instrumentation SDK (`import ragradar_capture`)

### 2.1 One-liner path: `ragradar_capture.capture(query, response, **explicit kwargs)`

`api.py:284`

The signature is explicit keyword arguments (`chunks`, `final_prompt`,
`token_budget`, `history_pre`/`history_post`, `eviction_reason`,
`cache_events`, `tool_calls`, `model`, `token_usage`, `pipeline`) — not
`**kwargs` — so an unrecognized keyword fails immediately with `TypeError`
at the call site rather than being silently swallowed.

1. Build a `Capture(query, pipeline)` — creates an empty
   `RunRecord(query=query, response="")` (`api.py:80`, `Capture.__init__`).
2. Set `cap._record.response = response`.
3. For each optional argument that was passed, either call the matching
   `Capture` method (`cap.chunks(...)`, `cap.history(...)`, `cap.cache(...)`,
   one `cap.tool_call(call)` per entry in `tool_calls`) or set the record
   field directly (`final_prompt`, `token_budget`, `model`, `token_usage`).
   `token_budget` is persisted whether or not `final_prompt` is given.
4. Call `cap.commit()`.
5. The **entire body is wrapped in `try/except Exception`** (`api.py:316`-`341`)
   — any internal failure is swallowed and logged to `~/.ragradar/errors.log`
   via `_handle_error()`/`_get_logger()` (`api.py:61`, `api.py:44`), never
   raised to the caller's pipeline — unless strict mode is on (§2.2).

### 2.2 Strict mode

`set_strict(True)` (`api.py:27`) or the `RAGRADAR_CAPTURE_STRICT=1`
environment variable (checked by `_strict_enabled()`, `api.py:40`) flips
every capture method from "log and swallow" to "re-raise" — meant for
development, to surface instrumentation bugs instead of hiding them in
`errors.log`. The default (`False`) keeps the fail-open production
contract.

### 2.3 Staged path: `ragradar_capture.start()` → `cap.X()` → auto-commit

```
cap = ragradar_capture.start(query=query, pipeline="my_project")   # api.py:272
  └── Capture(query, pipeline)                            # api.py:71, __init__ at api.py:80
  └── set_active_capture(cap)                              # thread_local.py:6
cap.chunks(chunks)          # api.py:92   → coerces each item to ChunkRecord
cap.context(prompt, budget) # api.py:105  → sets final_prompt + TokenBudget
cap.history(pre, post, eviction_reason)  # api.py:122 → sets history_pre/post + eviction_reason
cap.tool_call(call)         # api.py:234  → APPENDS one ToolCallRecord (never replaces)
cap.response(text, usage, model)  # api.py:139
  └── sets response, model, token_usage
  └── calls self.commit()          ← auto-commit on response()
cap.cache(events)           # api.py:161  → can be called any time before commit
cap.semantic_cache(...)     # api.py:173  → sets the query-level CacheRecord
cap.metadata_filter(...)    # api.py:207  → sets the FilterRecord
```

Each method:
- coerces raw dicts to the matching dataclass via `ragradar_core.coerce`
  (only if not already a dataclass instance),
- is wrapped in its own `try/except`, logging and swallowing failures
  independently (unless strict mode is on) — a broken `cap.chunks()` call
  does not prevent `cap.history()` or `cap.response()` from still
  capturing data.

`Capture.commit()` (`api.py:247`):
1. No-ops (returns the existing id) if `self._run_id is not None` — a
   second `cap.commit()` call (or the auto-commit inside `response()` plus
   a later manual call) is a silent no-op, not a second write.
2. `ragradar_core.store.commit_run(pipeline, record)` — resolves/creates
   the session, assigns the next `run_seq`, and inserts the run row all
   inside one `BEGIN IMMEDIATE` transaction (`store.py:430`). This replaced
   an earlier three-separate-connection sequence (`get_or_create_session()`
   + `next_run_seq()` + `write_run()`) that could race under concurrent
   commits to the same session; `commit_run()` is documented as the
   race-free consolidation.
3. Sets `self._run_id = f"s{session_id}r{run_seq}"` and returns it (or
   `None` if the write failed in non-strict mode, logged to
   `~/.ragradar/errors.log`). `cap.run_id` (`api.py:87`, a property) reads the
   same value without re-committing.

### 2.4 Thread-local proxy functions

`ragradar_capture.chunks()`, `.context()`, `.history()`, `.response()`,
`.cache()`, `.semantic_cache()`, `.metadata_filter()`, `.tool_call()`,
`.commit()` (module-level functions, `api.py:350` onward) are free
functions that:
1. `get_active_capture()` from `thread_local.py:11` (a `threading.local`).
2. If `None`, log an error via `_get_logger()` ("`<fn>` called with no
   active capture") and return — never raises.
3. Otherwise delegate to the corresponding `Capture` method.

This lets code deep in a call stack (e.g. a reranker module) call
`ragradar_capture.cache(events)` without threading a `Capture` object
through every function signature, as long as it executes on the same
thread that called `ragradar_capture.start()`.

### 2.5 Session auto-creation — `ragradar_core.store.get_or_create_session()`

`packages/ragradar-core/src/ragradar_core/store.py:347`

`Capture.commit()` (§2.3) no longer calls this public function directly —
it calls `commit_run()`, which runs an equivalent private
`_get_or_create_session_on(conn, ...)` inside its own atomic transaction.
The public `get_or_create_session()` documented below is still live and
used directly by `benchmark/seeder.py` for synthetic run generation
(§4.8), so the algorithm it implements is still worth tracing here.

1. `connect()` (`store.py:268`) is called implicitly on every store access
   — it creates `~/.ragradar/` and `runs.db` if missing and brings the schema
   to `SCHEMA_VERSION` (`"3"`, `store.py:26`) via `_ensure_schema()`
   (`store.py:151`) before any query runs. See §6 for what this means for
   fresh vs. pre-existing databases.
2. Look up the most recent session for this `pipeline` (or the most recent
   session with `pipeline IS NULL` if none given).
3. If found, check the last run's (or session's) `created_at` against
   `idle_gap_minutes` (default 30). If the gap is under 30 minutes, **reuse
   that session_id**.
4. Otherwise `INSERT` a new row into `sessions` and return the new
   `session_id`.

This is the mechanism `examples/rag_pipeline/02_capture_patterns.py`'s
`pattern_multi_session_gap()` exploits deliberately — see §5.2.

### 2.6 Scaffold generator — `ragradar-capture init`

`scaffold/cli.py:6` → `scaffold/template.py:41`
1. `generate_scaffold()` refuses to overwrite an existing file, and that
   file is genuinely still named **`ctx_pipeline.py`**
   (`scaffold/template.py:44`, `target = output_path / "ctx_pipeline.py"`) —
   this is real current behavior, not a leftover doc reference to the
   project's old `ctx` name. It raises `FileExistsError` if the file is
   already there.
2. Otherwise writes the hardcoded `TEMPLATE` string (`template.py:3`-`38`)
   — a function skeleton with `ragradar_capture.start()` / `cap.chunks()` /
   `cap.context()` / `cap.history()` / `cap.response()` calls
   pre-positioned as comments for the user to uncomment and fill in.

---

## 3. ragradar — analyst CLI (and umbrella import)

Entry point: `ragradar.cli:main`, a `click.Group` (`cli.py:68`). Its body is
just a docstring — **no schema-version check runs here**. There is no
`ragradar.store.check_schema_version()` (or any equivalent) anywhere in the
current source; every subcommand's first store access goes through
`ragradar_core.store.connect()`, which unconditionally brings the database
to the latest schema (creating it from scratch if it doesn't exist) as a
side effect of opening it. There is nothing left for `ragradar`'s CLI
group callback to check or warn about.

### 3.1 Target resolution — the shared primitive

Almost every command resolves a "target" string to a run row. Two
resolvers exist:

**`ragradar.store.resolve_target(target)`** (`store.py:89`) — used by
`explain`, `budget` (and, less strictly, `diff`, §3.7):
```
target is None          → get_latest_run()  (MAX created_at across all runs, ragradar_core.store.py:397)
target matches s(\d+)r(\d+)  → parse_target_id() (ragradar_core/targets.py:8) then get_run(...)  (exact lookup)
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
- No arg → `store.list_sessions()` (`store.py:16`): `LEFT JOIN`
  `sessions`/`runs`, grouped, `COUNT(run_seq)` per session, newest first.
  Rendered as a Rich `Table`.
- With `sN` or a bare int → `store.list_runs(sid)` (`store.py:38`): all
  runs in that session, newest first.

### 3.3 `ragradar find <hint> [--exact] [--from] [--to] [--today] [--session] [--pipeline] [--recent N]`

`cli.py:125` → `store.search_runs(...)` (`store.py:55`) →
`find/query_builder.build_search_query()` (`query_builder.py:1`)

`ragradar.store.search_runs()` calls `build_search_query(..., fts5_available=True)`
**unconditionally** (`store.py:82`) — it does not probe the database first.
This is safe because `ragradar_core.store.connect()` guarantees any
database it opens is at the latest schema, which ships the `runs_fts`
FTS5 virtual table and its sync triggers baked directly into the `SCHEMA`
DDL (`ragradar_core/store.py:82`-`103`). Practically: **FTS5 is always
available** to `ragradar find` today. `build_search_query()` still accepts
`fts5_available=False` and implements the plain-`LIKE` fallback
(`query_builder.py:30`-`38`), but nothing in the current CLI path ever
calls it that way — that branch is exercised only by unit tests that call
the function directly, not by any live command.

- Hint clause (when a hint is given):
  - **`--exact`**: `MATCH '"<hint>"'` (phrase match) against `runs_fts`.
  - **no `--exact`**: `MATCH '"t1" OR "t2" OR ...'` (any token) against
    `runs_fts`.
- Optional `session_id`, `pipeline`, `created_at >= from_dt`,
  `created_at <= to_dt` clauses are appended.
- `--today` (`cli.py:127`-`132`) is sugar: sets `from_dt`/`to_dt` to
  today's date bounds before calling `search_runs`.
- `ORDER BY created_at DESC`, optional `LIMIT` for `--recent`.

Results render as a table; no disambiguation step (this command *shows*
multiple matches by design, unlike `resolve_target`).

### 3.4 `ragradar explain [target] [--full] [--html]`

`cli.py:167`/`171`
1. `_resolve_and_load(target)` → `(run_row, record)`.
2. If `--html`: `html_renderer.render(record, run_id)` — see §3.6.
3. Else: `terminal_renderer.render(record, full=full, run_row=run_row)` —
   see §3.5.

### 3.5 Terminal renderer — `explain/renderer/terminal.py`

`render()` (`terminal.py:289`) is the orchestrator:
1. Print query, response (truncated to 200 chars unless `--full`), model.
2. Iterate `_ANALYZERS` (`terminal.py:233`) — a fixed list of
   `(module, render_fn)` pairs in a fixed order: **tokens → scores →
   duplicates → truncation → history → cache → degeneracy**. For each, call
   `mod.analyze(record)`; if it returns `None` (insufficient data), **skip
   silently** — nothing is printed for that factor.
3. If `record.cache is not None`, call `semantic_cache_mod.analyze(record,
   policy)` (loading the run's pipeline policy for its thresholds) and
   print `_render_semantic_cache` if not `None`. Not part of the
   `_ANALYZERS` loop — it needs the policy argument the loop doesn't pass,
   and always renders after it regardless of list position.
4. If `record.filter is not None`, call `metadata_filter_mod.analyze(record)`
   and print `_render_metadata_filter` if not `None` — also outside the
   loop, printed right after semantic cache to keep the two special-cased
   blocks together and preserve README's stated factor order.
5. If `run_row` has `eval_scores` (populated by `ragradar-evaluate run`),
   render an extra "Evaluation Scores" panel (`_render_eval_scores`,
   `terminal.py:244`) — risk score, input-quality violations, RAGAS
   metrics if present.
6. Print the final assembled prompt (truncated to 500 chars unless
   `--full`).

Each analyzer module in `ragradar/explain/analyzers/` is a pure function
`analyze(record) -> dict | None` (nine modules total):

| Module | Returns `None` when | Computes |
|---|---|---|
| `tokens.py` (`analyze` at `tokens.py:4`) | no chunks AND no final_prompt | sum of chunk token_counts + history_post tokens + system_allocated; utilisation % against `token_budget.total_limit`; per-chunk token breakdown |
| `scores.py` (`scores.py:4`) | no chunks (or chunks have neither retrieval nor rerank scores) | min/max retrieval & rerank scores; `rerank_delta` = mean(rerank) − mean(retrieval); `low_score_ratio` = fraction of rerank scores < 0.5 |
| `duplicates.py` (`duplicates.py:4`) | no chunks | **path dups**: same `chunk_id` seen via >1 distinct `retrieval_path`; **window dups**: chunks sharing `source_doc_id` where one's `content` is a substring of another's (pairwise, substring-only); `duplicate_ratio` = count of **distinct chunk_ids** implicated in any dup / total chunks; `semantic_dups` always `[]` (deferred — no embedding model in the free `ragradar` package) |
| `truncation.py` (`truncation.py:4`) | no chunks | chunks with `truncated=True`; `severity`: `"none"` if none truncated, `"high"` if any truncated chunk has `retrieval_score>0.7` or `rerank_score>0.7`, else `"low"` |
| `history.py` (`history.py:4`) | no history_pre AND no history_post | `dropped` = turns present in `pre` but whose `(role, content)` tuple is absent from `post`'s set; sums pre/post tokens |
| `cache.py` (`cache.py:4`) | no `cache_events` | hit/miss counts, `hit_ratio`, lists of hit/miss chunk_ids |
| `degeneracy.py` (`degeneracy.py:4`) | no chunks | per-chunk score = `rerank_score`, falling back to `retrieval_score`; chunks with neither excluded; `chunk_score_variance` = variance of the usable scores, `None` with fewer than two |
| `semantic_cache.py` (`semantic_cache.py:7`, takes `policy` too) | `record.cache is None` | `borderline_hit` (similarity within `policy.cache_borderline_margin` of threshold), `stale_hit` (`cached_at` older than `policy.cache_max_age_seconds`) |
| `metadata_filter.py` (`metadata_filter.py:4`) | `record.filter is None` | `filtered_exclusion_ratio` = `excluded_count / candidate_count`, `None` if either count is missing or `candidate_count` isn't positive |

Each `_render_X` function in `terminal.py` picks a Rich `Panel` border
color from thresholds (e.g. token utilisation <80% green, <95% yellow,
else red; duplicate ratio 0 green, ≤20% yellow, else red — `_render_tokens`
at `terminal.py:38`, `_render_scores` at `terminal.py:62`,
`_render_duplicates` at `terminal.py:86`, `_render_truncation` at
`terminal.py:107`, `_render_history` at `terminal.py:126`, `_render_cache`
at `terminal.py:148`, `_render_semantic_cache` at `terminal.py:163`,
`_render_metadata_filter` at `terminal.py:197`, `_render_degeneracy` at
`terminal.py:220`) and, when `--full`, appends per-item detail lines.

`render_budget()` (`terminal.py:332`) is just `tokens_mod.analyze()` +
`_render_tokens(..., full=True)` — used by `ragradar budget <target>`.

`render_diff()` (`terminal.py:341`) — used by `ragradar diff`:
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

`render(record, run_id)` (`html.py:39`) mirrors the terminal renderer's
analyzer loop but emits `<details><summary>...<pre>...</pre></details>`
blocks instead of Rich panels, string-escaping all content (`_esc`,
`html.py:26`). Writes to `~/.ragradar/reports/<run_id>.html` (creating the
`reports/` dir via `ragradar_core.store._ragradar_dir()`) and returns the
`Path`. No eval-scores section here (the HTML report is generated from
`record` alone, not `run_row`).

### 3.7 `ragradar diff <target_a> <target_b>`

`cli.py:185`/`188` — both targets resolved via `store.resolve_target`
directly (not `_resolve_and_load`, so **no interactive disambiguation**:
if either resolves to a list, the command just prints "Ambiguous target —
use exact run ID" and exits). Otherwise loads both records and calls
`terminal_renderer.render_diff`.

### 3.8 `ragradar budget <target>`

`cli.py:209`/`211` — `_resolve_and_load` then `terminal_renderer.render_budget`.

### 3.9 `ragradar session rename <id> <title>`

`cli.py:224`-`227` → `store.rename_session()` (`store.py:118`) — a plain
`UPDATE sessions SET title = ? WHERE session_id = ?`. This is the only
write `ragradar`'s CLI performs against run data (opening the store to
create/migrate `runs.db` is environment setup, not a "write" in this
sense).

### 3.10 The umbrella re-export (`import ragradar`)

`packages/ragradar/src/ragradar/__init__.py:1`-`82` imports and re-exports,
in one place: every `ragradar_capture` entry point (`Capture`, `start`,
`capture`, `set_strict`, `chunks`, `context`, `history`, `response`,
`cache`, `semantic_cache`, `metadata_filter`, `tool_call`, `commit`),
every `ragradar_evaluate` entry point (`check`, `evaluate`,
`available_metrics`, `CheckResult`, `EvalResult`, `MetricInfo`,
`InputQualityPolicy`), and the `ragradar_core` schema dataclasses
(`ChunkRecord`, `TokenBudget`, `TokenUsage`, `Turn`, `CacheEvent`,
`CacheRecord`, `ToolCallRecord`, `RunRecord`, `FilterRecord`). All three
example scripts
(§5) use only `import ragradar` — never `ragradar_capture`/
`ragradar_evaluate` directly — which is the intended day-to-day usage: a
production pipeline that wants only the capture side without pulling in
`ragas`/`scipy` installs `ragradar-capture` alone and imports
`ragradar_capture` directly instead.

---

## 4. ragradar-evaluate — evaluation layer (`ragradar_evaluate`)

Entry point: `ragradar_evaluate.cli:main` (`cli.py:118`). Its group
callback calls `store.ensure_store()` (`cli.py:121`, which is
`ragradar_core.store.ensure_store()` at `ragradar_core/store.py:295` —
open a connection, which self-migrates, then close it) on every
invocation. There is no `ragradar_evaluate.store` module and no
`apply_migration()` function anymore — both the connection/schema logic
and the full migration chain live in `ragradar_core.store` and run
automatically inside `connect()` (`ragradar_core/store.py:268`) for
**every** package's every store access, not just `ragradar-evaluate`'s.

### 4.1 Migration chain — `ragradar_core.store._ensure_schema()`

`packages/ragradar-core/src/ragradar_core/store.py:151`

```
no 'meta' table (brand-new db) → executescript(SCHEMA) in one shot: the
              FULL latest schema, including eval columns, benchmark/
              policies tables, and the runs_fts FTS5 table + sync
              triggers — all created together; meta.schema_version = "3"
              immediately. No "capture-only, not-yet-evaluated" DB exists
              anymore: the very first connect() from ANY of the four
              packages creates everything.
version "1"  → (a pre-existing db from an older package release) add
              eval_scores/risk_score/evaluated_at columns to runs;
              create benchmark table (pipeline, factor, threshold,
              correlation, sample_count, updated_at); create policies
              table (pipeline, policy_data, updated_at);
              meta.schema_version = "2"; falls through to "2" below
version "2"  → create FTS5 virtual table runs_fts(query) content-linked to
              runs.rowid; rebuild it; add INSERT/DELETE/UPDATE triggers to
              keep it in sync; drop the now-redundant idx_runs_query;
              meta.schema_version = "3"
version "3"  → no-op, already current
anything else → raise RuntimeError("Unsupported schema version")
```

Each step commits before falling through to the next, so a pre-existing
v1 DB walks v1→v2→v3 in one `connect()` call. Because this logic is now
centralized in `ragradar_core` and invoked by every package equally, there
is no cross-package "installing ragradar-evaluate is what unlocks FTS5"
coupling for anything created under the current codebase — that coupling
only ever applies to a `runs.db` file inherited from an older release
that already has `meta.schema_version` stamped `"1"` or `"2"`, and even
then, the very next `connect()` call from *any* of the four packages
(capture, analyst CLI, evaluate) walks it straight to `"3"`.

### 4.2 `ragradar-evaluate run [target] [--input-only] [--output-only] [--session] [--ground-truth] [--pipeline]`

`cli.py:124`/`131`

Both flags together (`--input-only --output-only`) is a clean error
(`cli.py:133`-`138`, exit 1).

Two paths, both routed through the public `ragradar_evaluate.evaluate()`
facade (`facade.py:298`) — the CLI has no scoring logic of its own beyond
flag-to-metric-list mapping (`_metrics_for_flags`, `cli.py:38`):

**Single run**:
1. `cli._resolve_target(target)` (`cli.py:28`) — the `sNrN`/latest
   resolver (no fuzzy search or disambiguation here — evaluation only
   accepts an exact `sNrN` or "latest").
2. `evaluate(run_id, metrics=<flag-mapped list or None>, ground_truth=...,
   pipeline=pipeline, save=True)` — see §4.3 for what this does internally.
3. `_warn_errors(result)` (`cli.py:192`) prints each distinct message in
   `result.errors` once (yellow).
4. `_render_eval_result` (`cli.py:59`) prints a risk-colored header, an
   "Input Quality" table (only the factors present), policy violations,
   and an "Output Quality (RAGAS)" table if present.

**Session batch** (`--session sN`, `cli.py:142`-`175`): loops
`store.get_runs_in_session(sid)`, caches `load_policy()` per distinct
pipeline key inside the loop (`policy_cache` dict, keyed by
`pipeline or run_row["pipeline"] or "__default"`) to avoid re-hitting the
`policies` table per run, computes each run's eval via
`evaluate(..., policy=policy_cache[key], save=False)` (no DB write yet),
then writes **all** results in one transaction via
`store.write_eval_scores_batch`, then renders each.

### 4.3 The `evaluate()`/`check()` facade — `facade.py`

`packages/ragradar-evaluate/src/ragradar_evaluate/facade.py` is the entire
public task-level API: `check()` (`facade.py:476`), `evaluate()`
(`facade.py:343`), `available_metrics()` (`facade.py:216`). Everything
else in the package (layers, policy, benchmark) is implementation detail
the facade calls into.

**Metric registry** (`facade.py:45`-`142`): a `dict[str, MetricInfo]`
(`MetricInfo` dataclass at `facade.py:35`: `name`, `layer` "input"/"output",
`cost` "free"/"llm", `requires` (RunRecord fields or `"ground_truth"`),
`description`). **Twelve entries** — eight input (`relevance`,
`duplicates`, `truncation`, `token_efficiency`, `coherence`, `cache_risk`,
`filter_risk`, `score_degeneracy`) and four output (`faithfulness`,
`answer_relevancy`, `context_precision`, `context_recall`). `cache_risk`
and `filter_risk` key off `record.cache`/`record.filter` instead of
`record.chunks` (`requires=("cache",)`/`("filter",)`); `evaluate()`'s
per-metric gate (below) checks each metric's own `requires` tuple for
`chunks`/`cache`/`filter` individually (`facade.py:407`-`414`) rather than
one hardcoded `chunks` check applied to every metric, specifically so
these two are not skipped for a chunk-less cache-hit run.
`INPUT_METRICS`/`OUTPUT_METRICS`
(`facade.py:144`-`145`) are the name tuples derived from it;
`available_metrics()` returns a fresh copy of the whole dict.

**`_resolve_target(target)`** (`facade.py:301`) accepts, in order: a bare
`RunRecord` (coerced via `ragradar_core.coerce.coerce_run_record`, pipeline
key `"__default"`, no run identity — `facade.py:314`); anything with a
`.run_id` attribute (e.g. a `ragradar_capture.Capture` — raises
`ValueError` if that capture was never committed); or an `sNrN` string
(parsed via `ragradar_core.targets.parse_target_id`, looked up via
`store.get_run`, pipeline defaults to the row's `pipeline` or
`"__default"`).

**`evaluate(target, *, metrics=None, ground_truth=None, pipeline=None,
policy=None, save=True)`** (`facade.py:343`):
1. Validates `metrics` (rejects an empty list and unknown names) if given.
2. Resolves the target and pipeline key.
3. `save=True` with an identity-less target (a bare `RunRecord`) raises
   `ValueError` telling the caller to pass `save=False` or an `sNrN` id.
4. For each requested **input** metric whose family (`relevance` →
   `input_quality.score_relevance`, etc. — the name-to-function map is
   `_INPUT_FN`, `facade.py:149`) is applicable (per its own `requires`
   gate — `chunks` non-empty, or `cache`/`filter` present and
   checked/applied for `cache_risk`/`filter_risk`, see §4.3 above):
   call it, round its output via `input_quality.round_values()`, store in
   `result.metrics[name]`. Everything not requested/applicable lands in
   `result.skipped[name]` with a reason string.
5. If any input metric ran: load the policy (unless one was passed
   explicitly), compute `policy_violations` via
   `input_quality.check_policy_violations()` on the **raw** (unrounded)
   values, and `risk_score` via `policy.risk.compute_risk_score()` on the
   **rounded** values — this raw-vs-rounded asymmetry is deliberate and
   documented in `layers/input_quality.py`'s module docstring (inherited
   from an earlier single-file implementation).
6. For each requested **output** (RAGAS) metric: gate on
   `record.chunks`/`record.response`/`ground_truth` per `MetricInfo.requires`,
   marking ungatable ones `skipped`. The remaining "runnable" list is
   passed as **one** call to `output_quality.score_output_quality()`
   (`facade.py:454`-`461`). If that call raises **for any reason** (missing
   `ragas` install, or a RAGAS runtime failure — `output_quality.py` itself
   only ever raises `ImportError` up front for a missing dependency;
   everything else propagates untouched), the facade catches it once and
   assigns the **same** error string to every requested-and-runnable output
   metric name in `result.errors` — there is no per-metric partial
   success once you're past the applicability gate.
7. `save=True` persists `result.to_eval_scores()` (`facade.py:252`, the
   `{"input": ..., "output": ...}` shape read back by `ragradar explain`)
   plus `risk_score` via `ragradar_core.store.write_eval_scores` — the only
   persistence path.

**`check(target, *, pipeline=None, policy=None)`** (`facade.py:476`):
free, deterministic, no LLM, never writes except a lazy benchmark build.
1. Resolves target/pipeline, runs `evaluate(record, metrics=list(INPUT_METRICS),
   save=False)` internally to get every input factor.
2. No chunks → `CheckResult(verdict="warn", ...)` explaining the missing
   data, rather than an exception.
3. `_learned_thresholds(pipeline_key)` (`facade.py:575`): reads the
   `benchmark` table; if empty and ≥10 evaluated runs exist for the
   pipeline, lazily calls `ragradar_evaluate.benchmark.builder.build()` to
   populate it (a failed build — e.g. no RAGAS scores to correlate against
   — falls back silently to "use policy"). `CheckResult.thresholds` records
   which source ("learned" vs "policy") was actually used.
4. For each of **eight** factors (`_CHECK_FACTORS`, `facade.py:163`-`213`:
   `duplicate_ratio`, `top_chunk_score`, `high_score_truncations`,
   `token_headroom_pct`, `source_domain_count`, `low_score_chunk_ratio`,
   `filtered_exclusion_ratio`, `chunk_score_variance`, each with a
   direction and a human-readable problem template), compare the computed
   value against the learned or policy threshold; `fail` on the wrong
   side, `ok` if no value/threshold exists. `token_headroom_pct`'s policy
   threshold is suppressed (treated as "no threshold") when the record has
   no captured `token_budget`. `filtered_exclusion_ratio` and
   `chunk_score_variance` are advisory-only here and in
   `check_policy_violations()` — neither is folded into
   `compute_risk_score()`'s weighted sum below, which still only weighs
   the original six.
5. Overall verdict: `fail` if `risk_score > 0.7` **or** ≥3 factors failed;
   `warn` if 1–2 failed; else `ok`.

### 4.4 Layer 1 — `layers/input_quality.py` (deterministic, no LLM)

Each metric family is its own pure function taking a `RunRecord`
(assuming non-empty `chunks` — callers gate on that) and returning a flat
dict of **raw** values:

- **`score_relevance`** (`input_quality.py:97`): cosine similarity between
  query and chunk embeddings if `embedding_fn` is supplied (not wired into
  the CLI by default — programmatic use only), else falls back to each
  chunk's `rerank_score` then `retrieval_score`. `mean_relevance` = average
  of whatever was collected; `top_chunk_score` = max **actual** rerank
  score (never from the relevance fallback list, and `None` if no chunk
  has a rerank score).
- **`score_duplicates`** (`input_quality.py:131`): `_detect_path_dups`
  (`input_quality.py:36`, chunk_id repeated across ≥2 distinct
  `retrieval_path`s — counts **qualifying chunk_ids**) +
  `_detect_window_dups` (`input_quality.py:45`, pairwise within a
  `source_doc_id` group: substring containment **or** token-set Jaccard
  overlap >50% — counts **qualifying PAIRS**, deduped via a `seen` set per
  source group) + `_detect_semantic_dups` (`input_quality.py:75`, pairwise
  cosine similarity > `0.92` across *different* `source_doc_id`s, only
  runs if `embedding_fn` given — reported separately, not folded into the
  ratio). `duplicate_ratio` = `(path_dup_count + window_dup_count) /
  total_chunks`.
- **`score_truncation`** (`input_quality.py:153`): same severity logic
  (none/low/high by score>0.7 threshold) as `ragradar explain`'s
  `truncation.py` analyzer, duplicated here so `ragradar-evaluate` has no
  runtime dependency on the `ragradar` package.
- **`score_token_efficiency`** (`input_quality.py:178`):
  `token_headroom_pct = headroom / total_limit` (0.0 when no
  `token_budget` was captured); `low_score_chunk_ratio` = fraction of
  chunks whose rerank score (or retrieval score if no rerank) is < 0.5.
- **`score_coherence`** (`input_quality.py:205`): `source_domain_count` =
  distinct `source_doc_id` count; `score_variance` = population variance
  of rerank scores only, no retrieval-score fallback (`None` with fewer
  than 2).
- **`score_score_degeneracy`** (`input_quality.py:225`): per-chunk score is
  `rerank_score`, falling back to `retrieval_score` when absent (chunks
  with neither excluded); `chunk_score_variance` = population variance of
  the usable scores, `None` with fewer than 2. Deliberately a distinct key
  from `score_coherence`'s `score_variance` above — the two are merged
  into the same flat dict by `score_input_quality()` (and are both
  candidate correlation factors for `benchmark/builder.py`), so sharing a
  key would let one silently clobber the other. No `policy` argument —
  matches `score_duplicates`/`score_token_efficiency`, not
  `score_cache_risk` below.
- **`score_cache_risk(record, policy)`** (`input_quality.py:259`): the one
  family function besides `score_score_degeneracy`'s neighbors that takes
  `policy` directly — it needs `cache_borderline_margin`/
  `cache_max_age_seconds` mid-computation, not just for a final threshold
  compare. Returns `None` (not a zero-value result) when
  `record.cache is None or not record.cache.checked`. Not called by
  `score_input_quality()` at all — gated separately in `evaluate()` on
  `record.cache`, not `record.chunks` (§4.3).
- **`score_filter_risk(record)`** (`input_quality.py:308`): returns `None`
  when `record.filter is None or not record.filter.applied`, or when
  `candidate_count`/`excluded_count` are missing or `candidate_count` isn't
  positive (a `0.0` ratio would misrepresent "unknown" as "nothing
  excluded"). `filtered_exclusion_ratio = excluded_count / candidate_count`.
  Also not called by `score_input_quality()` — gated on `record.filter` in
  `evaluate()`.
- **`check_policy_violations`** (`input_quality.py:366`) compares raw
  values against the active `InputQualityPolicy` and appends the field
  name to `violations` for every threshold breached; only checks a
  threshold whose backing value is present, so a partial (metric-subset)
  evaluation is never flagged for values it didn't compute.
- **`round_values`** (`input_quality.py:351`) rounds display/persistence
  copies to 4 places; **`score_input_quality`** (`input_quality.py:427`) is
  the six-family dispatcher (`relevance`, `duplicates`, `truncation`,
  `token_efficiency`, `coherence`, `score_degeneracy` — **not**
  `cache_risk`/`filter_risk`, which key off other fields) used by
  `benchmark/checker.py`
  (§4.7) — the `evaluate()` facade itself calls the individual family
  functions so that requesting one metric never computes the others.

### 4.5 Layer 2 — `layers/output_quality.py::score_output_quality()` (RAGAS, LLM-as-judge)

`output_quality.py:18`. Returns `None` if no chunks or no response.
Lazily imports `ragas`/`datasets` inside the function — **raises**
`ImportError` with an install hint if not installed; this propagates to
whoever called it (the facade catches it, §4.3 step 6). Builds a
single-row HF `Dataset` (`question`, `answer`, `contexts`, and
`ground_truth` if supplied — which also adds `context_recall` to the
default metric list). Calls `ragas.evaluate()` and unpacks whichever of
`faithfulness`/`answer_relevancy`/`context_precision`/`context_recall`
were requested into a flat dict (`evaluator: "ragas"`, `model: "unknown"`).
Any exception from `ragas.evaluate()` itself is **not** caught here — it
propagates out of `score_output_quality()` to the facade, which is the
only place a RAGAS failure ever turns into a soft error rather than an
exception (see §4.3 step 6 — there is no special `{"model": "error", ...}`
dict built inside this module).

### 4.6 Risk score — `policy/risk.py::compute_risk_score()`

`risk.py:13`. A fixed weighted sum over six factors (`_DEFAULT_WEIGHTS`,
`risk.py:3`-`10`, summing to 1.0: `duplicate_ratio` 0.15, `top_chunk_score`
0.25, `high_score_truncations` 0.30, `token_headroom_pct` 0.15,
`source_domain_count` 0.10, `low_score_chunk_ratio` 0.05). For each
factor, if the (rounded) input-quality value breaches the policy
threshold, add that factor's weight to `risk`. Result is a 0–1 score
independent of the `policy_violations` list (same threshold checks,
different output shape — one is a set of names, the other a weighted
magnitude), rounded to 4 places.

### 4.7 Policy system — `policy/schema.py`, `policy/persistence.py`

`InputQualityPolicy` (`policy/schema.py:5`) is a plain dataclass of
**twelve** thresholds with defaults (`min_chunk_relevance_score=0.5`,
`min_top_chunk_score=0.7`, `max_duplicate_ratio=0.2`,
`max_low_score_chunk_ratio=0.3`, `min_token_headroom=0.15`,
`max_high_score_truncations=0`, `max_source_domains=3`,
`llm_rewrite_risk_threshold=0.7`, `cache_borderline_margin=0.03`,
`cache_max_age_seconds=86400`, `max_filtered_exclusion_ratio=0.3`,
`min_score_variance=0.0001`). Stored per-pipeline as JSON in the
`policies` table (`pipeline` PK), created as part of the v1→v2 migration
step (or baked into a fresh schema directly, §4.1).

```
load_policy(pipeline)   → persistence.py:6  → store.get_policy(); None → InputQualityPolicy.default()
save_policy(pipeline,p) → persistence.py:19 → store.write_policy(pipeline, p.to_dict())   (INSERT OR REPLACE)
reset_policy(pipeline)  → persistence.py:24 → store.delete_policy(pipeline)
```

CLI (`policy show|set|reset`, `cli.py:327`):
- `show` (`cli.py:332`): loads the pipeline's policy (or default if
  unset), prints each field bolded if it differs from
  `InputQualityPolicy.default()`.
- `set <field> <value>` (`cli.py:354`): validates `field` against
  `dataclasses.fields(InputQualityPolicy)`, coerces `value` using
  `typing.get_type_hints()[field]` (so `float` fields parse as float,
  `int` as int), loads-mutates-saves.
- `reset` (`cli.py:384`): deletes the row, causing subsequent
  `load_policy` calls to fall back to `.default()`.

The literal string `"__default"` is the pipeline key used everywhere a
per-pipeline lookup key is needed and no `--pipeline`/run-derived pipeline
is available — confirmed still current in `facade.py:314`
(`_resolve_target` for a bare `RunRecord`) and throughout `cli.py`
(`pipeline or "__default"`, e.g. `cli.py:151`, `240`, `336`, `360`, `388`).

### 4.8 Benchmark system

Four independent commands operating on the `benchmark` table (all
internal machinery driven by the CLI's `benchmark` subgroup, `cli.py:203`
— `check()` also consults it lazily, §4.3).

**`benchmark seed <pipeline> [--count N]`** (`cli.py:302`/`305`) →
`seeder.seed()` (`seeder.py:7`)
- Generates `count` synthetic `RunRecord`s (half via `_good_record`
  (`seeder.py:26`), half `_bad_record` (`seeder.py:63`)) with hand-tuned
  scores designed to be clearly distinguishable (good: rerank
  ~0.90–0.98, no truncation, 3 source docs; bad: rerank ~0.25–0.40, half
  the chunks truncated, 6–8 distinct `source_doc_id`s).
- Writes them under the literal tag `f"{pipeline}__seeded"`
  (`seeder.py:13`) via `ragradar_core.store.write_runs_batch` — a batch
  `INSERT`, bypassing the `Capture`/`commit()` API entirely since there's
  no live pipeline to instrument.
- These seeded runs have **no RAGAS scores** — they exist purely to give
  `benchmark build` an input-quality distribution before any real
  evaluated runs exist ("day-zero baseline"). `exporter.py` explicitly
  filters out any pipeline ending in `"__seeded"` so they never leak into
  a RAGAS training export (`exporter.py:13`).

**`benchmark build [--pipeline]`** (`cli.py:208`/`210`) → `builder.build()`
(`builder.py:45`)
1. `store.get_all_evaluated_runs(pipeline)` — every run with non-null
   `eval_scores`. Raises `ValueError` if fewer than 10 (`builder.py:48`-`49`).
2. For each of 9 fixed `INPUT_FACTORS` (`builder.py:6`-`16`:
   `duplicate_ratio`, `top_chunk_score`, `high_score_truncations`,
   `token_headroom_pct`, `source_domain_count`, `low_score_chunk_ratio`,
   `mean_relevance`, `truncated_count`, `score_variance`): collect
   `(factor_value, ragas_value)` pairs across runs that have **both** the
   factor and at least one of `RAGAS_METRICS = [faithfulness,
   answer_relevancy]` (`builder.py:18`). Skip the factor if fewer than 3
   samples.
3. `scipy.stats.pearsonr(factor_values, ragas_values)` per RAGAS metric
   (only if both lists have >1 distinct value, else `None`). `best_corr`
   in the CLI table (`cli.py:225`-`226`) is whichever correlation has the
   largest absolute value.
4. `_suggest_threshold()` (`builder.py:21`): brute-force scan over
   midpoints between consecutive sorted unique factor values; for each
   candidate threshold, split samples into ≤threshold / >threshold, take
   the absolute difference of mean `faithfulness` (`RAGAS_METRICS[0]`)
   between the two groups; keep the threshold that maximizes this gap. A
   simple 1D decision-stump search, not a formal statistical test.
5. `store.write_benchmark_entries_batch(...)` — `INSERT OR REPLACE` one
   row per factor into `benchmark`, keyed `(pipeline, factor)`.
6. Prints a table of threshold/correlation/sample-count per factor.

**`benchmark show [--pipeline]`** (`cli.py:236`/`238`) → straight
`store.get_benchmark(pipeline)`, rendered as a table.

**`benchmark check <target> [--pipeline]`** (`cli.py:264`/`267`) →
`checker.check(session_id, run_seq, pipeline)` (`checker.py:11`) — note
this function takes the already-parsed `(session_id, run_seq)` pair, not
a raw target string; the CLI does `checker.check(*parse_target_id(target),
pipeline)`.
1. Load the run, its policy, compute fresh `input_quality.score_input_quality()`.
2. Load the pipeline's `benchmark` rows into a `factor → row` map.
3. For all eight `_CHECK_FACTORS` (`checker.py:35`: `check_factors =
   [(factor, direction) for factor, direction, _, _ in _CHECK_FACTORS]` —
   derived directly from `facade.py`'s `_CHECK_FACTORS`, §4.3, rather than
   a second hardcoded list; this was a fixed bug — `checker.py` used to
   maintain its own separate 6-then-7-entry copy that drifted out of sync
   with `facade.py`'s list), compare the run's current value against the
   benchmark threshold: `fail` if it's on the wrong side, else `ok`. If no
   benchmark entry exists for a factor, status is unconditionally `ok`.
   Note `checker.check()`'s thresholds come **only** from the learned
   `benchmark` table, never from policy defaults — unlike `facade.check()`
   (§4.3), it has no policy fallback, so a factor with no benchmark row
   yet is always `ok` regardless of the run's actual value.
4. Overall verdict: `fail` if `risk_score > 0.7` **or** ≥3 factors failed;
   `warn` if 1–2 failed; else `ok`. `risk_score` is read from the
   already-persisted `eval_scores`/`risk_score` columns
   (`store.get_eval_scores`, not recomputed) — so `benchmark check`
   requires the run to have been evaluated via `ragradar-evaluate run`
   first, or `risk` silently reads as `None` (never counts toward the
   verdict).

**`benchmark export [--pipeline] [--output]`** (`cli.py:311`/`314`) →
`exporter.export()` (`exporter.py:8`)
- Filters `get_all_evaluated_runs()` to drop `"__seeded"`-tagged pipelines
  and runs missing `chunks`/`response`.
- Writes one JSON object per line (`question`, `answer`, `contexts`,
  `ground_truth: null`, plus `run_id`/`pipeline`/`evaluated_at` metadata)
  to `~/.ragradar/exports/<pipeline>_ragas_<timestamp>.jsonl` (or the given
  `--output` path) — a RAGAS-compatible dataset for offline reuse.

---

## 5. End-to-end trace (the three shipped examples)

`examples/rag_pipeline/` is now three standalone scripts, each runnable
independently — there is no single `run_pipeline.py`/`pipeline.py`
anymore. All three `import ragradar` (the umbrella package, §3.10), never
`ragradar_capture`/`ragradar_evaluate` directly.

### 5.1 `01_quickstart.py` — the whole capture surface, fast

```
ragradar.capture("what is 2+2?", "4")
  → ragradar_capture.capture(): builds a Capture, sets response="4",
    commits immediately. Returns "sNrN".

cap = ragradar.start(query="what is RRF?", pipeline="quickstart")
cap.chunks([{ "content": "...", "retrieval_score": 0.9, "rerank_score": 0.95 }])
  → one plain dict with only content/retrieval_score/rerank_score given;
    coerce_chunk() (ragradar_core/coerce.py:88) fills chunk_id="chunk_0",
    source_doc_id="unknown", token_count=estimate_tokens(content).
run_id = cap.response("RRF combines rankings from multiple retrievers into one ranked list.")
  → auto-commits (cap.commit() is not called explicitly; response() does it)
```

Two runs land in the store: one in a `pipeline IS NULL` session (the
one-liner), one in a `"quickstart"` session.

### 5.2 `02_capture_patterns.py` — patterns beyond the quickstart

`PIPELINE = "rag_example"`. `_sample_chunks()` (`02_capture_patterns.py:19`)
builds four chunks engineered to trigger specific analyzer behavior:

```
rrf_norm_1   source_doc_id="rrf_paper_2024"  content is an exact PREFIX of rrf_norm_2's
rrf_norm_2   source_doc_id="rrf_paper_2024"  → triggers duplicates.py / _detect_window_dups
                                                 (substring containment) for this pair
bm25_tf_idf  source_doc_id="ir_textbook_ch3"  truncated=True, rerank_score=0.88
                                                 → truncation severity "high" (>0.7 threshold)
ctx_window   source_doc_id="rag_patterns"     retrieval_score=0.48, rerank_score=0.39
                                                 → pulls low_score_ratio/low_score_chunk_ratio up
```
Note the chunk_id `"ctx_window"` is genuine current example data, not a
stale rename artifact. Across these four chunks there are only **3**
distinct `source_doc_id`s (`rrf_paper_2024`, `ir_textbook_ch3`,
`rag_patterns`), which does *not* exceed the default `max_source_domains=3`
policy threshold — unlike the six-source fixture an earlier version of
this example used, this one stays inside the coherence threshold.

**`pattern_full_fields()`** (`02_capture_patterns.py:69`) — one staged
capture touching every field:
```
cap = ragradar.start(query=..., pipeline=PIPELINE)
cap.metadata_filter(applied=True, candidate_count=6, excluded_count=2,
                     filters={"source": "internal"})
  → runs before retrieval, first statement in the function
    (02_capture_patterns.py:74-79) → metadata_filter.py exclusion ratio
    33% (2/6)
cap.chunks(_sample_chunks())
cap.context(prompt, {"total_limit": 4096, "chunks_allocated": 2800,
                      "history_allocated": 600, "system_allocated": 500})
  → headroom omitted; coerce_token_budget() derives it from the given
    allocations: 4096 - (2800+600+500) = 196  (4.8% of the limit — low
    headroom, renders red in tokens.py's utilisation check)
cap.history(pre=[4 turns], post=[2 turns], eviction_reason="token_budget")
  → 2 turns dropped → history.py flags eviction
cap.cache({c["chunk_id"]: c["cache_hit"] for c in chunks})
  → 1 hit out of 4 (only rrf_norm_1) → cache.py hit_ratio = 0.25
cap.tool_call({"tool_name": "rerank", "arguments": {...}, "result": "...", "latency_ms": 42.0})
  → appends one ToolCallRecord; captured but not rendered by any current
    explain analyzer (§1)
run_id = cap.response(text, token_usage={"input_tokens": 1850, "output_tokens": 40}, model="gpt-4-turbo")
  → auto-commits
```
No `cap.semantic_cache(...)` call here — this run never checks a
semantic cache, so `explain/analyzers/semantic_cache.py` returns `None`
and the "Cache behavior" panel doesn't render for it (only "Cache hits",
from the per-chunk `cap.cache(...)` above, does). Verified live via
`ragradar explain s4r3 --full` against a fresh store: this run renders
nine panels total — Token Usage, Chunk Scores, Duplicate Chunks,
Truncation, Dropped History, Cache Hits, Score Degeneracy, Metadata
Filter, Final Prompt — everything except Cache behavior.

**`_backdate_pipeline_runs(db_path, pipeline, minutes)`**
(`02_capture_patterns.py:137`) is explicitly test/demo-only: it rewrites
`sessions.created_at`/`runs.created_at` directly via raw SQL for every row
matching `pipeline`. Real pipelines never touch `runs.db` directly —
session gaps happen naturally over wall-clock time between calls to
`ragradar.start()`.

**`pattern_multi_session_gap()`** (`02_capture_patterns.py:156`):
1. Captures 2 queries under `PIPELINE` (creates session A, 2 runs).
2. Backdates all `"rag_example"` rows 31 minutes into the past.
3. Captures 2 more queries under `PIPELINE` — `get_or_create_session()`
   now sees the (backdated) last activity as >30 minutes stale, so it
   creates a **new** session B (2 runs).

**`pattern_thread_local_proxy()`** (`02_capture_patterns.py:172`): calls
`ragradar.start(query=..., pipeline="proxy_demo")` then uses the
module-level `ragradar.chunks()`/`ragradar.response()` proxies (not a
`cap` object) — demonstrating the thread-local pattern for code that
doesn't want to thread a `Capture` handle through its call stack. Creates
its own session under `"proxy_demo"`.

`__main__` (`02_capture_patterns.py:190`-`197`) runs
`pattern_multi_session_gap()`, then `pattern_thread_local_proxy()`, then
`pattern_full_fields()` last — deliberately, so the full-fields run is
the most recently captured one **within this script's own run**. In
practice, if `03_evaluate.py` is then run afterward (per the top-level
README's quickstart order), *its* demo run becomes the actual latest run
across the whole store — see §5.3's note. Because `pattern_full_fields()`
runs under `PIPELINE` again shortly after session B was created (real
wall-clock time, well under the 30-minute idle gap), it lands as a
**third run in session B**, not a new session. End state: `"rag_example"`
has 2 sessions (A: 2 runs, backdated; B: 3 runs), `"proxy_demo"` has 1
session (1 run).

### 5.3 `03_evaluate.py` — the `check()`/`evaluate()` facade

Runs standalone (captures its own demo run first); `PIPELINE = "rag_example"`.

**`capture_demo_run()`** (`03_evaluate.py:20`): one `ragradar.capture()`
one-liner (not staged) with 4 chunks (`rrf_1`/`rrf_2` overlapping-content
pair, `bm25_1` truncated with rerank 0.88, `win_1` with rerank 0.39) and a
`token_budget` dict (headroom again derives to 196/4096 ≈ 4.8%). No
history/cache/tool_calls/metadata-filter captured here. Returns the run's
`sNrN` id.

**Sequencing note, verified live:** if this script is run *after*
`02_capture_patterns.py` (per the top-level README's quickstart order),
this call's run becomes the actual latest run across the whole store —
superseding `02_capture_patterns.py`'s `pattern_full_fields()` run (§5.2)
as the target of a target-less `ragradar explain --full`. Since this run
has no history/cache/metadata-filter, only **five** panels render for it:
Token Usage, Chunk Scores, Duplicate Chunks, Truncation, Score Degeneracy
— not the nine `pattern_full_fields()`'s run shows. The two demo runs
serve different illustrative purposes but only one of them is actually
"latest" once both scripts have run.

**`show_check(run_id)`** (`03_evaluate.py:84`): `ragradar.check(run_id)` →
prints `verdict`/`risk_score`/`thresholds` plus a factor-by-factor table
(`CheckResult.factors`) and any `problems` strings.

**`show_single_metric(run_id)`** (`03_evaluate.py:113`):
`ragradar.evaluate(run_id, metrics=["duplicates"], save=False)` — computes
**only** the duplicates family (nothing else, no DB write) and prints
`duplicate_ratio`/`path_dup_count`/`window_dup_count`.

**`show_full_evaluate(run_id)`** (`03_evaluate.py:124`):
`ragradar.evaluate(run_id)` — `metrics=None` runs every applicable metric.
For this run: 6 of the 8 input families compute (`relevance`,
`duplicates`, `truncation`, `token_efficiency`, `coherence`,
`score_degeneracy`); `cache_risk`/`filter_risk` land in `result.skipped`
("not applicable: run never checked a semantic cache" /
"...never applied a metadata filter") since this run has neither. All 4
RAGAS output metrics are attempted if `ragas` is installed. `save=True`
(default) persists the result. Prints `result.saved`, `risk_score`,
`policy_violations`, a metrics table, and (if `ragas` isn't installed or
fails) the shared error string from `result.errors` for each requested
output metric — this is the fail-soft path traced in §4.3 step 6,
demonstrated live.

**`show_available_metrics()`** (`03_evaluate.py:155`):
`ragradar.available_metrics()` → prints all **twelve** registered
`MetricInfo` entries (name, layer, cost, requires) — see §4.3.

`__main__` (`03_evaluate.py:166`-`177`) runs all four functions in order,
then prints a pointer to `ragradar explain <run_id>` (to see the same
scores rendered alongside the run analysis) and
`ragradar-evaluate benchmark export`.

---

## 6. Cross-cutting behaviors worth knowing

- **Fail-open instrumentation**: every public `ragradar_capture.api`
  function/method catches its own exceptions and logs to
  `~/.ragradar/errors.log`; nothing in the capture SDK can raise into a host
  pipeline **unless strict mode is on** (`set_strict(True)` or
  `RAGRADAR_CAPTURE_STRICT=1`, §2.2). `ragradar` and `ragradar-evaluate` are
  not held to this standard — CLI errors there use
  `SystemExit(1)`/`ValueError` propagation deliberately, since they run
  interactively.

- **The store always exists after any access — there is no "missing
  runs.db" state to special-case.** `ragradar_core.store.connect()`
  (`store.py:268`) unconditionally creates `~/.ragradar/` and `runs.db` (if
  missing) and brings the schema to the latest version before returning a
  connection. There is no `_connect()` helper anywhere that returns `None`
  for a missing file — that was true of an earlier per-package store
  design and is no longer how this works. "No runs found." UX comes from
  querying a real, freshly-created, empty database, not from a
  guard-clause on a missing file.

- **Schema versioning is centralized in `ragradar_core`, not
  package-local.** `ragradar_core.store` owns `SCHEMA_VERSION` and the
  entire migration chain (§4.1) in one file, invoked by `connect()` on
  every open, from every package. `ragradar.cli`'s `main()` group callback
  performs no schema check of any kind (§3) — there is no
  `ragradar.store.check_schema_version()`. `ragradar_evaluate.cli`'s
  `main()` calls `store.ensure_store()` (§4), but this is no longer a
  special "upgrade" step unique to that package — every command in every
  package gets the same guarantee automatically via `connect()`. The
  practical result: a brand-new `runs.db` is created with FTS5 already
  wired up regardless of which package touches it first, and
  `ragradar.store.search_runs()` hardcodes `fts5_available=True`
  (`ragradar/store.py:82`) rather than probing for it — the plain-`LIKE`
  fallback branches in `find/query_builder.py` are exercised only by unit
  tests today, not by any live command.

- **Two independent duplicate-detection implementations, and they don't
  even count the same unit.** `ragradar explain`'s `duplicates.py`
  (`explain/analyzers/duplicates.py:4`) does substring-containment-only
  window-dup detection and reports `duplicate_ratio` as **distinct
  chunk_ids** implicated in any dup, divided by total chunks.
  `ragradar-evaluate`'s `input_quality._detect_window_dups`
  (`layers/input_quality.py:45`) does substring containment **or** >50%
  token-set Jaccard overlap, and reports `window_dup_count` as the number
  of qualifying **pairs** (deduped per source group), which feeds into
  `duplicate_ratio = (path_dup_count + window_dup_count) / total_chunks` —
  a pair-count ratio, not a chunk-count ratio. The two can disagree even
  on the exact same run for two independent reasons (the broader
  Jaccard check, and the pairs-vs-chunks unit mismatch), not just
  borderline-case substring disagreement. This is intentional package
  independence (`ragradar-evaluate` doesn't import `ragradar`), not a bug,
  but `ragradar explain`'s and `ragradar-evaluate run`'s duplicate numbers
  for the same run are not guaranteed to match, and the two implementations
  aren't even measuring quite the same thing.

- **`"__default"` pipeline key**: the literal string `"__default"` is used
  everywhere a per-pipeline lookup key is needed and no `--pipeline`/
  run-derived pipeline is available — confirmed in `facade.py:314` and
  throughout `ragradar_evaluate/cli.py`'s command bodies. (Seeded benchmark
  pipelines use a different literal, `f"{pipeline}__seeded"` —
  `benchmark/seeder.py:13` — not the default-pipeline convention; don't
  confuse the two suffixes/keys.)
