"""Task-level public API of gaptrace-evaluate.

Two user tasks, one discovery helper:

- check(target)     — "is this run healthy?" Free, deterministic, no LLM.
- evaluate(target)  — "score it fully" (or any atomic subset of metrics).
- available_metrics() — what can be scored, at what cost.

Underneath sits the atomic-metric engine: each input metric family is its
own function in layers.input_quality; output metrics are individually
selectable RAGAS metrics in layers.output_quality. Benchmark machinery
(seeding, building, exporting) is internal — check() consults learned
thresholds automatically when enough evaluated runs exist.
"""

import json
from dataclasses import dataclass, field

from gaptrace_core import store
from gaptrace_core.coerce import coerce_run_record
from gaptrace_core.schema import RunRecord
from gaptrace_core.targets import parse_target_id

from gaptrace_evaluate.layers import input_quality, output_quality
from gaptrace_evaluate.policy.persistence import load_policy
from gaptrace_evaluate.policy.risk import compute_risk_score
from gaptrace_evaluate.policy.schema import InputQualityPolicy

# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricInfo:
    """Describes one selectable metric. Pure data."""

    name: str
    layer: str  # "input" | "output"
    cost: str  # "free" | "llm"
    requires: tuple  # RunRecord fields (or "ground_truth") the metric needs
    description: str


_METRICS: dict[str, MetricInfo] = {
    m.name: m
    for m in [
        MetricInfo(
            "relevance",
            "input",
            "free",
            ("chunks",),
            "Per-chunk relevance vs the query (rerank/retrieval scores, or "
            "embeddings when configured): mean_relevance, top_chunk_score.",
        ),
        MetricInfo(
            "duplicates",
            "input",
            "free",
            ("chunks",),
            "Path and window duplicate detection: duplicate_ratio and per-kind counts.",
        ),
        MetricInfo(
            "truncation",
            "input",
            "free",
            ("chunks",),
            "Which chunks were trimmed and severity (high when a high-score chunk was truncated).",
        ),
        MetricInfo(
            "token_efficiency",
            "input",
            "free",
            ("chunks",),
            "Token headroom percentage (0.0 without a captured budget) and low-score chunk ratio.",
        ),
        MetricInfo(
            "coherence",
            "input",
            "free",
            ("chunks",),
            "Source-domain spread and rerank score variance.",
        ),
        MetricInfo(
            "cache_risk",
            "input",
            "free",
            ("cache",),
            "Semantic-cache check quality: flags borderline-similarity hits and "
            "stale cached answers. Not applicable to runs that never checked a "
            "semantic cache.",
        ),
        MetricInfo(
            "filter_risk",
            "input",
            "free",
            ("filter",),
            "Metadata-filter exclusion ratio: how much of the candidate pool "
            "was excluded before scoring ever saw it. Not applicable to runs "
            "that never applied a metadata filter (or didn't report "
            "candidate/excluded counts).",
        ),
        MetricInfo(
            "score_degeneracy",
            "input",
            "free",
            ("chunks",),
            "Chunk-score variance (rerank falling back to retrieval): "
            "near-zero variance means scores aren't discriminating between "
            "chunks, a sign of a structural retrieval failure rather than "
            "a normal quality judgment call.",
        ),
        MetricInfo(
            "score_margin",
            "input",
            "free",
            ("chunks",),
            "Top-vs-second chunk score margin (checked factor) bundled with "
            "top-vs-threshold margin (diagnostic-only context on the "
            "existing min_top_chunk_score boundary): a thin top_second_margin "
            "means the retriever isn't decisively ahead on its top pick.",
        ),
        MetricInfo(
            "score_underfill",
            "input",
            "free",
            ("chunks",),
            "Requested-vs-returned chunk count: underfill_ratio, how far "
            "short retrieval landed of the captured requested_chunk_count "
            "(top_k ask). Not applicable to runs that never captured a "
            "requested_chunk_count.",
        ),
        MetricInfo(
            "faithfulness",
            "output",
            "llm",
            ("chunks", "response"),
            "RAGAS: is the answer grounded in the retrieved context?",
        ),
        MetricInfo(
            "answer_relevancy",
            "output",
            "llm",
            ("chunks", "response"),
            "RAGAS: does the answer address the question?",
        ),
        MetricInfo(
            "context_precision",
            "output",
            "llm",
            ("chunks", "response"),
            "RAGAS: are the relevant chunks ranked ahead of irrelevant ones?",
        ),
        MetricInfo(
            "context_recall",
            "output",
            "llm",
            ("chunks", "response", "ground_truth"),
            "RAGAS: does the context cover the ground-truth answer? Needs ground_truth.",
        ),
    ]
}

INPUT_METRICS: tuple = tuple(m for m, i in _METRICS.items() if i.layer == "input")
OUTPUT_METRICS: tuple = tuple(m for m, i in _METRICS.items() if i.layer == "output")

# Metric family name -> input_quality function name; resolved via getattr
# at call time so tests can spy on individual families.
_INPUT_FN = {
    "relevance": "score_relevance",
    "duplicates": "score_duplicates",
    "truncation": "score_truncation",
    "token_efficiency": "score_token_efficiency",
    "coherence": "score_coherence",
    "cache_risk": "score_cache_risk",
    "filter_risk": "score_filter_risk",
    "score_degeneracy": "score_score_degeneracy",
    "score_margin": "score_score_margin",
    "score_underfill": "score_score_underfill",
}

# Metric families whose function needs the active policy passed in during
# computation (not just a final compare against a computed value) --
# score_cache_risk needs it for borderline/stale thresholds, score_score_margin
# needs it for threshold_margin. Everything else's family function is
# policy-free; check_policy_violations()/_CHECK_FACTORS compare its raw
# output against the policy afterward instead.
_POLICY_ARG_METRICS = {"cache_risk", "score_margin"}

# The ten factors check() compares against thresholds, with the direction
# in which a value is bad, the policy field naming the default threshold,
# and a human-readable problem template.
_CHECK_FACTORS = [
    (
        "duplicate_ratio",
        "higher_bad",
        "max_duplicate_ratio",
        "duplicate chunks: ratio {value:.2f} exceeds {threshold:.2f}",
    ),
    (
        "top_chunk_score",
        "lower_bad",
        "min_top_chunk_score",
        "top chunk score {value:.2f} below {threshold:.2f} minimum",
    ),
    (
        "high_score_truncations",
        "higher_bad",
        "max_high_score_truncations",
        "{value:.0f} high-score chunk(s) truncated (allowed: {threshold:.0f})",
    ),
    (
        "token_headroom_pct",
        "lower_bad",
        "min_token_headroom",
        "token headroom {value:.0%} below {threshold:.0%} minimum",
    ),
    (
        "source_domain_count",
        "higher_bad",
        "max_source_domains",
        "chunks span {value:.0f} sources (max {threshold:.0f})",
    ),
    (
        "low_score_chunk_ratio",
        "higher_bad",
        "max_low_score_chunk_ratio",
        "{value:.0%} of chunks scored below 0.5 (max {threshold:.0%})",
    ),
    (
        "filtered_exclusion_ratio",
        "higher_bad",
        "max_filtered_exclusion_ratio",
        "{value:.0%} of candidates excluded by metadata filter (max {threshold:.0%})",
    ),
    (
        "chunk_score_variance",
        "lower_bad",
        "min_score_variance",
        "chunk score variance {value:.4f} below {threshold:.4f} minimum "
        "(scores aren't discriminating between chunks)",
    ),
    (
        "top_second_margin",
        "lower_bad",
        "min_top_second_margin",
        "top-second margin {value:.2f} below {threshold:.2f} minimum "
        "(top chunk isn't decisively ahead of the runner-up)",
    ),
    (
        "underfill_ratio",
        "higher_bad",
        "max_underfill_ratio",
        "{value:.0%} short of the requested chunk count (max {threshold:.0%})",
    ),
]


def available_metrics() -> dict[str, MetricInfo]:
    """Every selectable metric, keyed by name. Pure.

    Returns a fresh dict of MetricInfo entries (name, layer input/output,
    cost free/llm, required RunRecord fields, description). Every key is
    a valid entry for evaluate()'s metrics argument.
    """
    return dict(_METRICS)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    """Outcome of evaluate(). Pure data.

    metrics maps each computed metric name to its result — a dict of
    values for input families, a float (or None if RAGAS returned
    nothing) for output metrics. skipped maps metric -> reason (not
    requested / missing data / requires ground_truth). errors maps
    metric -> error string; RAGAS-not-installed and RAGAS runtime
    failures both land here, same shape. risk_score is None when input
    metrics were not computed — 0.0 only ever means "computed, no risk".
    """

    run_id: str | None
    metrics: dict = field(default_factory=dict)
    skipped: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    policy_violations: list = field(default_factory=list)
    risk_score: float | None = None
    saved: bool = False

    def to_eval_scores(self) -> dict:
        """The {"input": ..., "output": ...} dict persisted on the run row
        (and read back by `gaptrace explain`). Pure."""
        input_values: dict = {}
        for name in INPUT_METRICS:
            if name in self.metrics:
                input_values.update(self.metrics[name])
        if input_values:
            input_values["policy_violations"] = self.policy_violations
            input_values["passes_policy"] = len(self.policy_violations) == 0

        output_values: dict = {}
        for name in OUTPUT_METRICS:
            if name in self.metrics:
                output_values[name] = self.metrics[name]
        if output_values:
            output_values["evaluator"] = "ragas"

        return {
            "input": input_values or None,
            "output": output_values or None,
        }


@dataclass
class CheckResult:
    """Outcome of check(). Pure data.

    verdict is "ok" | "warn" | "fail". problems is a list of
    human-readable strings (empty when ok). thresholds says which
    standards were applied: "learned" (benchmark built from >=10
    evaluated runs) or "policy" (defaults / configured policy).
    risk_score is None when input quality could not be assessed.
    factors maps factor name -> {value, threshold, status}.
    """

    verdict: str
    problems: list = field(default_factory=list)
    risk_score: float | None = None
    factors: dict = field(default_factory=dict)
    thresholds: str = "policy"
    run_id: str | None = None


# ---------------------------------------------------------------------------
# Target resolution
# ---------------------------------------------------------------------------


def _resolve_target(target) -> tuple[RunRecord, str | None, tuple | None, str]:
    """Resolve evaluate()/check() targets to (record, run_id, (sid, seq), pipeline).

    Read-only store access. Accepts an sNrN string, a RunRecord (no
    identity -> run_id/key are None, pipeline "__default"), or any
    object with a .run_id attribute (e.g. a gaptrace_capture.Capture).
    Hand-built RunRecords are normalized through gaptrace_core.coerce, so
    primitive chunks/turns (the same shapes gaptrace_capture.capture()
    accepts) score fine. Raises ValueError for unresolvable targets,
    naming the id.
    """
    if isinstance(target, RunRecord):
        try:
            return coerce_run_record(target), None, None, "__default"
        except (TypeError, KeyError) as e:
            raise ValueError(
                f"Target RunRecord contains data that cannot be coerced into the schema: {e}"
            ) from e

    if not isinstance(target, str) and hasattr(target, "run_id"):
        run_id = target.run_id
        if run_id is None:
            raise ValueError(
                "This Capture has no run id — it was never committed (or the "
                "commit failed). Call cap.commit() / cap.response() first."
            )
        target = run_id

    session_id, run_seq = parse_target_id(target)  # ValueError on bad format
    row = store.get_run(session_id, run_seq)
    if row is None:
        raise ValueError(f"Run {target} not found in the store.")
    record = RunRecord.from_json(json.loads(row["run_data"]))
    pipeline = row["pipeline"] or "__default"
    return record, f"s{session_id}r{run_seq}", (session_id, run_seq), pipeline


# ---------------------------------------------------------------------------
# evaluate()
# ---------------------------------------------------------------------------


def evaluate(
    target,
    *,
    metrics: list[str] | None = None,
    ground_truth: str | None = None,
    pipeline: str | None = None,
    policy: InputQualityPolicy | None = None,
    save: bool = True,
) -> EvalResult:
    """Score a run: everything applicable, or exactly the metrics named.

    Inputs: target is an sNrN string, a RunRecord, or a committed
    gaptrace_capture.Capture. metrics=None runs every metric applicable to
    the record (the complete eval); a list like ["duplicates"] computes
    ONLY those metrics — unselected input families are never computed
    and unselected RAGAS metrics are never sent to the judge.
    pipeline= scores against that pipeline's configured policy instead
    of the run's own (mirroring check()); passing an explicit policy=
    overrides both.

    Output: an EvalResult (see its docstring for the skipped/errors/
    risk_score semantics).

    Side effects: save=True (default) persists the scores on the run row
    via gaptrace_core.store.write_eval_scores — the only persistence path.
    save=True requires a resolvable identity; a bare RunRecord raises
    ValueError telling you to pass save=False or an sNrN id.

    Errors: ValueError for an unknown metric name (listing valid names),
    an empty metrics list, an unresolvable target, or an identity-less
    save. RAGAS failures never raise — they land in EvalResult.errors.
    """
    if metrics is not None:
        if len(metrics) == 0:
            raise ValueError(
                "metrics=[] computes nothing. Pass metrics=None for a "
                "complete eval or name at least one metric."
            )
        unknown = [m for m in metrics if m not in _METRICS]
        if unknown:
            raise ValueError(f"Unknown metric(s): {unknown}. Valid metrics: {sorted(_METRICS)}")

    record, run_id, run_key, derived_pipeline = _resolve_target(target)
    pipeline_key = pipeline or derived_pipeline

    if save and run_key is None:
        raise ValueError(
            "save=True needs a run identity to write to, but a bare "
            "RunRecord has none. Pass save=False, or pass the run's sNrN "
            "id (e.g. the string gaptrace_capture.capture() returned)."
        )

    requested = list(_METRICS) if metrics is None else list(metrics)
    result = EvalResult(run_id=run_id)

    for name in _METRICS:
        if name not in requested:
            result.skipped[name] = "not requested"

    # --- Input metric families (free, deterministic) ---
    requested_input = [m for m in requested if m in INPUT_METRICS]
    raw_input_values: dict = {}
    for name in requested_input:
        info = _METRICS[name]
        if "chunks" in info.requires and not record.chunks:
            result.skipped[name] = "missing data: record has no chunks"
            continue
        if "cache" in info.requires and (record.cache is None or not record.cache.checked):
            result.skipped[name] = "not applicable: run never checked a semantic cache"
            continue
        if "filter" in info.requires and (record.filter is None or not record.filter.applied):
            result.skipped[name] = "not applicable: run never applied a metadata filter"
            continue
        family_fn = getattr(input_quality, _INPUT_FN[name])
        if name in _POLICY_ARG_METRICS:
            if policy is None:
                policy = load_policy(pipeline_key)
            values = family_fn(record, policy)
        else:
            values = family_fn(record)
        if values is None:
            result.skipped[name] = "not applicable"
            continue
        result.metrics[name] = input_quality.round_values(values)
        raw_input_values.update(values)

    if raw_input_values:
        if policy is None:
            policy = load_policy(pipeline_key)
        # Policy checks run on raw values; risk (like the persisted
        # metrics) uses the rounded copy — both inherited from the
        # original monolith.
        result.policy_violations = input_quality.check_policy_violations(
            raw_input_values, policy, record
        )
        result.risk_score = compute_risk_score(input_quality.round_values(raw_input_values), policy)

    # --- Output metrics (RAGAS, LLM cost) ---
    requested_output = [m for m in requested if m in OUTPUT_METRICS]
    runnable: list[str] = []
    for name in requested_output:
        info = _METRICS[name]
        if "chunks" in info.requires and not record.chunks:
            result.skipped[name] = "missing data: record has no chunks"
        elif "response" in info.requires and not record.response:
            result.skipped[name] = "missing data: record has no response"
        elif "ground_truth" in info.requires and not ground_truth:
            result.skipped[name] = "requires ground_truth"
        else:
            runnable.append(name)

    if runnable:
        try:
            scores = output_quality.score_output_quality(record, ground_truth, metrics=runnable)
            for name in runnable:
                result.metrics[name] = (scores or {}).get(name)
        except Exception as e:
            for name in runnable:
                result.errors[name] = str(e)

    if save:
        session_id, run_seq = run_key
        store.write_eval_scores(session_id, run_seq, result.to_eval_scores(), result.risk_score)
        result.saved = True

    return result


# ---------------------------------------------------------------------------
# check()
# ---------------------------------------------------------------------------


def check(
    target,
    *,
    pipeline: str | None = None,
    policy: InputQualityPolicy | None = None,
) -> CheckResult:
    """Is this run healthy? Free, deterministic, no LLM, instant.

    Runs all input metric families on the target (sNrN string, RunRecord,
    or committed Capture) and compares them against the current
    standards: learned benchmark thresholds when they exist (built
    automatically once >=10 evaluated runs exist for the pipeline),
    the pipeline's policy otherwise — CheckResult.thresholds says which.

    Read-only except for the lazy benchmark build (which writes benchmark
    rows the first time enough evaluated history exists). Never calls an
    LLM. Raises ValueError only for an unresolvable target; a chunk-less
    record gets a "warn" verdict explaining the missing data instead of
    an exception.
    """
    record, run_id, _run_key, derived_pipeline = _resolve_target(target)
    pipeline_key = pipeline or derived_pipeline

    if policy is None:
        policy = load_policy(pipeline_key)

    eval_result = evaluate(record, metrics=list(INPUT_METRICS), policy=policy, save=False)

    input_values: dict = {}
    for name in INPUT_METRICS:
        if name in eval_result.metrics:
            input_values.update(eval_result.metrics[name])

    if not input_values:
        return CheckResult(
            verdict="warn",
            problems=[
                "No chunks captured — input quality cannot be assessed. "
                "Capture retrieval chunks (cap.chunks(...)) to enable checks."
            ],
            risk_score=None,
            factors={},
            thresholds="policy",
            run_id=run_id,
        )

    learned = _learned_thresholds(pipeline_key)
    thresholds_source = "learned" if learned else "policy"

    factors: dict = {}
    problems: list[str] = []
    fail_count = 0

    for factor, direction, policy_field, template in _CHECK_FACTORS:
        value = input_values.get(factor)
        if learned:
            row = learned.get(factor)
            threshold = row["threshold"] if row else None
        else:
            threshold = getattr(policy, policy_field)
            # Policy headroom check only applies when a budget was captured.
            if factor == "token_headroom_pct" and not record.token_budget:
                threshold = None

        if value is None or threshold is None:
            status = "ok"
        elif direction == "lower_bad":
            status = "fail" if value < threshold else "ok"
        else:
            status = "fail" if value > threshold else "ok"

        if status == "fail":
            fail_count += 1
            problems.append(template.format(value=float(value), threshold=float(threshold)))

        factors[factor] = {
            "value": value,
            "threshold": threshold,
            "status": status,
        }

    risk = eval_result.risk_score
    if (risk is not None and risk > 0.7) or fail_count >= 3:
        verdict = "fail"
    elif fail_count >= 1:
        verdict = "warn"
    else:
        verdict = "ok"

    return CheckResult(
        verdict=verdict,
        problems=problems,
        risk_score=risk,
        factors=factors,
        thresholds=thresholds_source,
        run_id=run_id,
    )


def _learned_thresholds(pipeline: str) -> dict:
    """Benchmark rows for ``pipeline`` keyed by factor, building them
    lazily when >=10 evaluated runs exist and none are stored yet.

    Writes to store only in the lazy-build case; a failed build (e.g.
    the evaluated runs have no RAGAS scores to correlate against) falls
    back silently to an empty dict, meaning "use policy".
    """
    rows = {r["factor"]: r for r in store.get_benchmark(pipeline)}
    if rows:
        return rows

    if len(store.get_all_evaluated_runs(pipeline)) >= 10:
        try:
            from gaptrace_evaluate.benchmark import builder

            builder.build(pipeline)
            rows = {r["factor"]: r for r in store.get_benchmark(pipeline)}
        except Exception:
            rows = {}
    return rows
