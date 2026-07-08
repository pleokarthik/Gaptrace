# gaptrace

Analyst CLI for the gaptrace observability system. Reads the local store at
`~/.gaptrace/runs.db` that `gaptrace-capture` writes; browses sessions, searches
runs, and explains exactly what went into a run's context window.

```
pip install gaptrace
```

Runs are addressed as `sNrN` (session 2, run 3 → `s2r3`) — the id every
capture call returns. Commands that take a `<target>` accept an exact
id, nothing (= latest run), or a quoted text hint (searched; multiple
matches show a pick list).

## gaptrace list — sessions and runs

```bash
gaptrace list          # sessions, newest first
gaptrace list s2       # runs inside session 2
```

```
                  Sessions
| ID | Runs | Pipeline    | Created    | Title |
|----+------+-------------+------------+-------|
| s2 |    3 | rag_example | 2026-07-02 |       |
| s1 |    1 | quickstart  | 2026-07-02 |       |
```

## gaptrace find — search runs by query text

```bash
gaptrace find "reranking"              # token match (FTS5)
gaptrace find "score scale" --exact    # phrase match
gaptrace find "RRF" --session s2       # scope to a session
gaptrace find "RRF" --pipeline rag_example
gaptrace find "RRF" --from 2026-07-01 --to 2026-07-02
gaptrace find "RRF" --today
gaptrace find --recent 5               # latest N runs, no hint
```

```
        Search results (2)
| Run   | Date       | Session | Query                            |
|-------+------------+---------+----------------------------------|
| s2 r3 | 2026-07-02 |         | what is RRF and how does it ...  |
```

## gaptrace explain — the seven analysis factors

```bash
gaptrace explain            # latest run
gaptrace explain s2r3       # specific run
gaptrace explain s2r3 --full
gaptrace explain s2r3 --html    # snapshot to ~/.gaptrace/reports/s2r3.html
```

Renders every factor the captured data supports, silently skipping the
rest: token usage, chunk scores, duplicates, truncation, dropped
history, cache hits, and the final prompt. Runs scored by
`gaptrace-evaluate` also show an Evaluation Scores panel (risk score, policy
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

## gaptrace diff — compare two runs

```bash
gaptrace diff s2r1 s2r3
```

Side-by-side query delta, chunks added/removed, per-chunk score deltas,
token budget deltas, history and truncation changes. Ambiguous targets
are rejected — use exact ids here.

## gaptrace budget — token waterfall only

```bash
gaptrace budget s2r3
```

```
+------------- Token Usage --------------+
| Total: 1138/4096 (27.8%)               |
|   Chunks:   625   History: 13         |
|   System:   500   Headroom: 196       |
+----------------------------------------+
```

## gaptrace session rename

```bash
gaptrace session rename s2 "RRF investigation"
```

```
Session 2 renamed to "RRF investigation".
```

## Notes

- Read-mostly by design: the only data this CLI writes is a session
  title. (Opening the store may create/migrate `runs.db` via `gaptrace-core`
  — that's environment setup, not run data.)
- No LLM anywhere in the navigation path; search is SQLite FTS5.
- Optional semantic search: `pip install gaptrace[semantic]`.
