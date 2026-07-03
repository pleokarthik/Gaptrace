import json

from ragradar_core import store
from scipy import stats

INPUT_FACTORS = [
    "duplicate_ratio",
    "top_chunk_score",
    "high_score_truncations",
    "token_headroom_pct",
    "source_domain_count",
    "low_score_chunk_ratio",
    "mean_relevance",
    "truncated_count",
    "score_variance",
]

RAGAS_METRICS = ["faithfulness", "answer_relevancy"]


def _suggest_threshold(values: list[float], ragas_scores: list[float]) -> float:
    sorted_vals = sorted(set(values))
    if len(sorted_vals) < 2:
        return sorted_vals[0] if sorted_vals else 0.0

    best_threshold = sorted_vals[0]
    best_diff = 0.0

    for i in range(len(sorted_vals) - 1):
        threshold = (sorted_vals[i] + sorted_vals[i + 1]) / 2
        below = [r for v, r in zip(values, ragas_scores) if v <= threshold]
        above = [r for v, r in zip(values, ragas_scores) if v > threshold]

        if not below or not above:
            continue

        diff = abs(sum(above) / len(above) - sum(below) / len(below))
        if diff > best_diff:
            best_diff = diff
            best_threshold = threshold

    return round(best_threshold, 4)


def build(pipeline: str | None = None) -> dict:
    runs = store.get_all_evaluated_runs(pipeline)

    if len(runs) < 10:
        raise ValueError(f"Need at least 10 evaluated runs to build benchmark, found {len(runs)}.")

    parsed = []
    for r in runs:
        parsed.append(json.loads(r["eval_scores"]))

    pipeline_key = pipeline or "__default"
    factors_result = {}
    batch_entries: list[tuple] = []

    for factor in INPUT_FACTORS:
        factor_values: list[float] = []
        ragas_values: dict[str, list[float]] = {m: [] for m in RAGAS_METRICS}

        for eval_data in parsed:
            input_data = eval_data.get("input") or {}
            output_data = eval_data.get("output") or {}

            fval = input_data.get(factor)
            if fval is None:
                continue

            has_ragas = any(output_data.get(m) is not None for m in RAGAS_METRICS)
            if not has_ragas:
                continue

            factor_values.append(float(fval))
            for m in RAGAS_METRICS:
                ragas_values[m].append(float(output_data.get(m) or 0.0))

        if len(factor_values) < 3:
            continue

        correlations: dict[str, float | None] = {}
        for m in RAGAS_METRICS:
            vals = ragas_values[m]
            if (
                len(vals) == len(factor_values)
                and len(set(factor_values)) > 1
                and len(set(vals)) > 1
            ):
                corr, _ = stats.pearsonr(factor_values, vals)
                # float(): scipy stubs expose corr as numpy.float64 (_T_co), not float;
                # the cast is the narrowing workaround — no type: ignore needed.
                correlations[f"{m}_correlation"] = round(float(corr), 4)
            else:
                correlations[f"{m}_correlation"] = None

        valid_corrs = [v for v in correlations.values() if v is not None]
        primary_corr = max(valid_corrs, key=abs) if valid_corrs else 0.0

        primary_ragas = ragas_values[RAGAS_METRICS[0]]
        suggested = _suggest_threshold(factor_values, primary_ragas)

        batch_entries.append((pipeline_key, factor, suggested, primary_corr, len(factor_values)))

        factors_result[factor] = {
            **correlations,
            "suggested_threshold": suggested,
            "sample_count": len(factor_values),
        }

    store.write_benchmark_entries_batch(batch_entries)

    return {
        "run_count": len(runs),
        "pipeline": pipeline,
        "factors": factors_result,
    }
