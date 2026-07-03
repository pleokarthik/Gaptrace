# ragradar-evaluate

Scores captured runs. Two tasks, one discovery helper:

| Task | Call | Cost |
|---|---|---|
| "Is this run healthy?" | `check(run_id)` | free — deterministic, no LLM, instant |
| "Score it fully" | `evaluate(run_id)` | free input metrics + LLM-judged output metrics |
| "What can be scored?" | `available_metrics()` | free |

```
pip install ragradar-evaluate
```

## Is this run healthy? — check()

Call before paying for an LLM; put it in CI.

```python
import ragradar_capture
from ragradar_evaluate import check

run_id = ragradar_capture.capture(
    "what is RRF?", "RRF fuses rankings.",
    chunks=[{"chunk_id": "c1", "source_doc_id": "d1",
             "content": "RRF combines rankings.", "token_count": 10,
             "rerank_score": 0.9}],
)
result = check(run_id)

print(result.verdict)      # "ok" | "warn" | "fail"
print(result.problems)     # ["duplicate chunks: ratio 0.50 exceeds 0.20", ...]
print(result.risk_score)   # 0.0-1.0, None if input quality couldn't be assessed
print(result.factors)      # per-factor {value, threshold, status}
print(result.thresholds)   # "learned" | "policy" — which standards were applied
```

`check()` compares all free input metrics against the **current
standards**: once at least 10 evaluated runs exist for the pipeline it
uses thresholds learned from your own history (and says so via
`thresholds == "learned"`); before that it falls back to the policy
defaults. A run captured without chunks gets a `warn` verdict explaining
the missing data — never an exception.

## Score it fully — evaluate()

```python
import ragradar_capture
from ragradar_evaluate import evaluate

run_id = ragradar_capture.capture(
    "what is RRF?", "RRF fuses rankings.",
    chunks=[{"chunk_id": "c1", "source_doc_id": "d1",
             "content": "RRF combines rankings.", "token_count": 10,
             "rerank_score": 0.9}],
)

# Complete eval: every metric applicable to the record.
result = evaluate(run_id)

# One atomic metric — nothing else is computed:
result = evaluate(run_id, metrics=["duplicates"], save=False)

# A chosen subset:
result = evaluate(run_id, metrics=["relevance", "faithfulness"])
```

`target` can be an sNrN string (what `ragradar_capture.capture()` returns), a
committed `Capture` object, or a bare `RunRecord` (then pass
`save=False` — there's no run row to write to).

### EvalResult

| Field | Meaning |
|---|---|
| `metrics` | per-metric results: a dict of values for input families, a float for RAGAS metrics |
| `skipped` | metric → reason: `"not requested"`, `"missing data: ..."`, or `"requires ground_truth"` |
| `errors` | metric → error string; RAGAS-not-installed and RAGAS runtime failures land here identically — `evaluate()` never raises for judge failures |
| `policy_violations` | policy thresholds breached by the computed values |
| `risk_score` | `None` when input metrics weren't computed; `0.0` only ever means "computed, no risk" |
| `run_id` / `saved` | identity and whether scores were persisted |

`save=True` (default) persists via the one store path; `ragradar explain
<run_id>` then shows the scores alongside its analysis.

## available_metrics()

| Metric | Layer | Cost | Requires |
|---|---|---|---|
| `relevance` | input | free | chunks |
| `duplicates` | input | free | chunks |
| `truncation` | input | free | chunks |
| `token_efficiency` | input | free | chunks |
| `coherence` | input | free | chunks |
| `faithfulness` | output | llm | chunks, response |
| `answer_relevancy` | output | llm | chunks, response |
| `context_precision` | output | llm | chunks, response |
| `context_recall` | output | llm | chunks, response, ground_truth |

Output metrics are RAGAS LLM-as-judge calls — they cost money and need a
configured judge. **To stay free-only**, use `check()`, or select input
metrics explicitly: `evaluate(run_id, metrics=["relevance",
"duplicates", "truncation", "token_efficiency", "coherence"])`.

## Policy system

Human-set thresholds encoding known failure modes; active from day one
and the fallback standard for `check()`. Stored per pipeline.

```bash
ragradar-evaluate policy show
ragradar-evaluate policy set max_duplicate_ratio 0.1
ragradar-evaluate policy reset
```

Programmatic override: `evaluate(run_id, policy=InputQualityPolicy(...))`.

## Benchmark lifecycle (CLI)

Learned thresholds accumulate as you evaluate real runs — `check()`
picks them up automatically at 10+ evaluated runs. The CLI exposes the
machinery for inspection:

```bash
ragradar-evaluate run s2r3                 # evaluate one run (both layers)
ragradar-evaluate run s2r3 --input-only    # free metrics only
ragradar-evaluate run --session s2         # evaluate a whole session
ragradar-evaluate benchmark show           # current learned thresholds
ragradar-evaluate benchmark build          # rebuild from evaluated history
ragradar-evaluate benchmark check s2r3     # factor-by-factor threshold check
ragradar-evaluate benchmark export         # RAGAS-compatible JSONL dataset
```

`--input-only --output-only` together is an error (it would compute
nothing).
