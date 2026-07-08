import json

from gaptrace_core import store
from gaptrace_core.schema import RunRecord

from gaptrace_evaluate.facade import _CHECK_FACTORS
from gaptrace_evaluate.layers import input_quality
from gaptrace_evaluate.policy.persistence import load_policy


def check(
    session_id: int,
    run_seq: int,
    pipeline: str | None = None,
) -> dict:
    run_row = store.get_run(session_id, run_seq)
    if run_row is None:
        raise ValueError(f"Run s{session_id}r{run_seq} not found.")

    record = RunRecord.from_json(json.loads(run_row["run_data"]))
    pipeline = pipeline or run_row["pipeline"] or "__default"
    policy = load_policy(pipeline)

    input_scores = input_quality.score_input_quality(record, policy)
    benchmark = store.get_benchmark(pipeline)
    benchmark_map = {b["factor"]: b for b in benchmark}

    factors = {}
    fail_count = 0

    # Factor/direction pairs mirror facade.py's _CHECK_FACTORS (the single
    # source of truth for which factors check() evaluates); this check()
    # only needs the factor name and direction, not the policy field or
    # human-readable template used against learned benchmarks.
    check_factors = [(factor, direction) for factor, direction, _, _ in _CHECK_FACTORS]

    for factor, direction in check_factors:
        value = input_scores.get(factor) if input_scores else None
        bench = benchmark_map.get(factor)
        threshold = bench["threshold"] if bench else None

        if value is None or threshold is None:
            status = "ok"
        elif direction == "lower_bad":
            status = "fail" if value < threshold else "ok"
        else:
            status = "fail" if value > threshold else "ok"

        if status == "fail":
            fail_count += 1

        factors[factor] = {
            "value": value,
            "benchmark_threshold": threshold,
            "status": status,
        }

    # risk_score is None when the run was never evaluated or its input
    # metrics could not be computed (0.0 strictly means "computed, no
    # risk") — unknown risk never counts toward the verdict.
    eval_data = store.get_eval_scores(session_id, run_seq)
    risk = eval_data.get("risk_score") if eval_data else None

    if (risk is not None and risk > 0.7) or fail_count >= 3:
        overall = "fail"
    elif fail_count >= 1:
        overall = "warn"
    else:
        overall = "ok"

    return {
        "run_id": f"s{session_id}r{run_seq}",
        "risk_score": risk,
        "benchmark_available": len(benchmark) > 0,
        "factors": factors,
        "overall": overall,
    }
