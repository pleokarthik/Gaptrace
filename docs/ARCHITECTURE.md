# Architecture

## The naming law: capture vs run

**"Capture" is the verb — the API surface. "Run" is the noun — the
data.** You *capture* a pipeline execution; what lands in the store is a
*run*.

- `ragradar_capture.capture()`, `ragradar_capture.start()` → `Capture` (the action
  object), `cap.chunks()`, `cap.response()`, strict mode — all verb-side.
- `RunRecord`, the `runs` table, `run_id`, `sNrN` ids (`s2r3` = session
  2, run 3) — all noun-side.

Anything that reads data talks about runs; anything that instruments a
pipeline talks about capturing.

## The dependency star

```
        ragradar-capture     ragradar      ragradar-evaluate
              \          |          /
               \         |         /
                └──  ragradar-core  ──┘
```

`ragradar-core` is the shared kernel: the `RunRecord` schema, the single
SQLite store, and the one `sNrN` parser. The three user-facing packages
depend on it and **not on each other** — `ragradar` and `ragradar-evaluate` never
import `ragradar_capture`. Users don't import `ragradar_core` directly;
`ragradar_capture` and `ragradar_evaluate` re-export the dataclasses.

## Zero-dependency guarantee

`ragradar_core` (and therefore `ragradar-capture`, which adds nothing beyond it)
imports only the Python standard library. A subprocess test
(`packages/ragradar-core/tests/test_zero_deps.py`) enforces this: importing
every `ragradar_core` module must pull in nothing outside the stdlib.
Similarly, `import ragradar_evaluate` must not import scipy or ragas — those
load lazily inside the functions that need them.

## Store and schema ownership

One store: `~/.ragradar/runs.db` (SQLite, WAL). One owner: `ragradar_core.store`.
One version constant: `ragradar_core.store.SCHEMA_VERSION` (currently `"3"`).

`ragradar_core.store.connect()` is the environment-setup contract — every
call guarantees the directory exists, the database exists, and the
schema is at the latest version. Fresh databases are created directly at
the latest version (including the FTS5 search index and evaluation
columns); databases written by older releases are migrated in place
(v1 → v2 adds eval columns + benchmark/policies tables; v2 → v3 adds
FTS5). Any entry point — library call, CLI command, example script —
therefore works on a fresh machine with no prior setup step.

Tables: `meta` (schema version), `sessions`, `runs` (with
`eval_scores`/`risk_score`/`evaluated_at` written by ragradar-evaluate),
`benchmark`, `policies`, `runs_fts` (FTS5 index over run queries).

## Where errors go

Capture is fail-open: nothing in `ragradar_capture` can raise into a host
pipeline. Failures are logged to `~/.ragradar/errors.log` and the call
returns `None` where a run id was expected. Development flips this with
`ragradar_capture.set_strict(True)` or `RAGRADAR_CAPTURE_STRICT=1`.

`ragradar` and `ragradar-evaluate` run interactively and are allowed to exit
non-zero with a clear message. In the evaluation API, LLM-judge (RAGAS)
failures never raise — both "not installed" and runtime failures land in
`EvalResult.errors`, one channel, same shape.

## Evaluation surface

`ragradar_evaluate` exposes user tasks, not machinery:

- `check(target)` — "is this run healthy?" Free, deterministic, no LLM.
  Compares all input metrics against the current standards: learned
  benchmark thresholds once ≥10 evaluated runs exist for the pipeline,
  policy defaults before that (`CheckResult.thresholds` says which).
- `evaluate(target, metrics=None, ...)` — complete scoring, or exactly
  the atomic metrics named.
- `available_metrics()` — discovery.

Benchmark seeding/building/exporting is internal, reachable through the
`ragradar-evaluate benchmark` CLI commands for inspection.
