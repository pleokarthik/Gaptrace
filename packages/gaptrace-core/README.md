# gaptrace-core

Shared kernel for the gaptrace observability system: the run-record schema, the
single SQLite store, and the sNrN target parser. `gaptrace-capture`, `gaptrace`, and
`gaptrace-evaluate` all depend on it — it depends on nothing.

**You normally do not import this directly.** Instrument pipelines with
`gaptrace_capture`, evaluate with `gaptrace_evaluate` — both re-export the schema
dataclasses. `gaptrace_core` exists so those packages share one store contract
instead of three copies of it.

## Zero-dependency guarantee

`gaptrace_core` imports only the Python standard library (`sqlite3`,
`dataclasses`, `json`, `re`, `pathlib`, `datetime`). This is enforced by a
test (`tests/test_zero_deps.py`) that imports the package in a subprocess
and asserts nothing outside the stdlib was loaded.

## What lives here

| Module | Contents |
|---|---|
| `gaptrace_core.schema` | `RunRecord` and its child dataclasses (`ChunkRecord`, `TokenBudget`, `TokenUsage`, `Turn`, `CacheEvent`, `ToolCallRecord`), all tolerant of unknown kwargs |
| `gaptrace_core.store` | store location, schema + migrations, and every persistence primitive (runs, eval scores, benchmark, policies) |
| `gaptrace_core.targets` | `parse_target_id("s4r3") -> (4, 3)` — the one sNrN parser |

## Environment setup contract

`gaptrace_core.store.connect()` guarantees the environment before returning a
connection:

1. `~/.gaptrace/` exists (created if missing),
2. `~/.gaptrace/runs.db` exists (created if missing),
3. the schema is at the latest version — fresh databases are created
   directly at the latest version; databases written by older package
   versions are migrated in place.

Any entry point — a library call, a CLI command, an example script — works
on a fresh machine with no prior CLI invocation.

## Schema version + migration story

One constant, `gaptrace_core.store.SCHEMA_VERSION` (currently `"3"`), recorded
in the `meta` table. The migration chain walks old databases forward on
first connect:

- **v1 → v2**: adds `eval_scores` / `risk_score` / `evaluated_at` columns
  to `runs`; creates the `benchmark` and `policies` tables.
- **v2 → v3**: creates the `runs_fts` FTS5 index over run queries (with
  insert/update/delete sync triggers, backfilled from existing rows) and
  drops the now-redundant `idx_runs_query` index.

A database reporting a version this package doesn't know raises
`RuntimeError` rather than guessing.

## DB location and layout

The store lives at `~/.gaptrace/runs.db` (SQLite, WAL mode).

| Table | Columns |
|---|---|
| `meta` | `key`, `value` — holds `schema_version` |
| `sessions` | `session_id`, `title`, `pipeline`, `created_at` |
| `runs` | `session_id`, `run_seq`, `query`, `pipeline`, `created_at`, `run_data` (JSON `RunRecord`), `eval_scores` (JSON), `risk_score`, `evaluated_at` |
| `benchmark` | `pipeline`, `factor`, `threshold`, `correlation`, `sample_count`, `updated_at` |
| `policies` | `pipeline`, `policy_data` (JSON), `updated_at` |
| `runs_fts` | FTS5 index over `runs.query` |

Runs are addressed as `s{session_id}r{run_seq}` (e.g. `s2r3`) everywhere —
"run" is the data noun; capturing is the verb, and belongs to
`gaptrace-capture`.
