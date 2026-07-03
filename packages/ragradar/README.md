# ragradar

Analyst CLI for the ragradar observability system. Reads the local store at
`~/.ragradar/runs.db` that `ragradar-capture` writes; browses sessions, searches
runs, and explains exactly what went into a run's context window.

```
pip install ragradar
```

Runs are addressed as `sNrN` (session 2, run 3 → `s2r3`) — the id every
capture call returns. Commands that take a `<target>` accept an exact
id, nothing (= latest run), or a quoted text hint (searched; multiple
matches show a pick list).

## ragradar list — sessions and runs

```bash
ragradar list          # sessions, newest first
ragradar list s2       # runs inside session 2
```

```
                  Sessions
| ID | Runs | Pipeline    | Created    | Title |
|----+------+-------------+------------+-------|
| s2 |    3 | rag_example | 2026-07-02 |       |
| s1 |    1 | quickstart  | 2026-07-02 |       |
```

## ragradar find — search runs by query text

```bash
ragradar find "reranking"              # token match (FTS5)
ragradar find "score scale" --exact    # phrase match
ragradar find "RRF" --session s2       # scope to a session
ragradar find "RRF" --pipeline rag_example
ragradar find "RRF" --from 2026-07-01 --to 2026-07-02
ragradar find "RRF" --today
ragradar find --recent 5               # latest N runs, no hint
```

```
        Search results (2)
| Run   | Date       | Session | Query                            |
|-------+------------+---------+----------------------------------|
| s2 r3 | 2026-07-02 |         | what is RRF and how does it ...  |
```

## ragradar explain — the seven analysis factors

```bash
ragradar explain            # latest run
ragradar explain s2r3       # specific run
ragradar explain s2r3 --full
ragradar explain s2r3 --html    # snapshot to ~/.ragradar/reports/s2r3.html
```

Renders every factor the captured data supports, silently skipping the
rest: token usage, chunk scores, duplicates, truncation, dropped
history, cache hits, and the final prompt. Runs scored by
`ragradar-evaluate` also show an Evaluation Scores panel (risk score, policy
violations, RAGAS metrics).

```
+------------- Token Usage --------------+
| Total: 1138/4096 (27.8%)               |
|   Chunks:   625                        |
|   Headroom: 196                        |
+----------------------------------------+
+------------- Duplicates ---------------+
| 1 duplicate (25%): 1 window            |
+----------------------------------------+
```

## ragradar diff — compare two runs

```bash
ragradar diff s2r1 s2r3
```

Side-by-side query delta, chunks added/removed, per-chunk score deltas,
token budget deltas, history and truncation changes. Ambiguous targets
are rejected — use exact ids here.

## ragradar budget — token waterfall only

```bash
ragradar budget s2r3
```

```
+------------- Token Usage --------------+
| Total: 1138/4096 (27.8%)               |
|   Chunks:   625   History: 13         |
|   System:   500   Headroom: 196       |
+----------------------------------------+
```

## ragradar session rename

```bash
ragradar session rename s2 "RRF investigation"
```

```
Session 2 renamed to "RRF investigation".
```

## Notes

- Read-mostly by design: the only data this CLI writes is a session
  title. (Opening the store may create/migrate `runs.db` via `ragradar-core`
  — that's environment setup, not run data.)
- No LLM anywhere in the navigation path; search is SQLite FTS5.
- Optional semantic search: `pip install ragradar[semantic]`.
