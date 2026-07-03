"""LLM-as-judge output-quality scoring via RAGAS.

ragas/datasets are imported lazily inside the scoring function so that
importing this module (and ragradar_evaluate) stays cheap and works without
the optional heavy dependencies installed.
"""

from ragradar_core.schema import RunRecord

OUTPUT_METRICS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


def score_output_quality(
    record: RunRecord,
    ground_truth: str | None = None,
    metrics: list[str] | None = None,
) -> dict | None:
    """Score a run's output with RAGAS. Pure computation, costs LLM calls.

    Inputs: the record (needs chunks + response — returns None if either
    is missing), optional ground_truth, and an optional list of metric
    names from OUTPUT_METRICS. metrics=None means the default set:
    faithfulness, answer_relevancy, context_precision, plus
    context_recall when ground_truth is given. When metrics IS given,
    exactly those RAGAS metric objects are passed to ragas.evaluate —
    nothing else is computed.

    Errors: raises ImportError when ragas/datasets are not installed;
    ragas runtime failures propagate to the caller (the evaluate() facade
    turns both into EvalResult.errors entries).
    """
    if not record.chunks or not record.response:
        return None

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas import metrics as ragas_metrics
    except ImportError:
        raise ImportError("RAGAS not installed. Run: pip install ragas")

    if metrics is None:
        metrics = ["faithfulness", "answer_relevancy", "context_precision"]
        if ground_truth:
            metrics = metrics + ["context_recall"]

    unknown = [m for m in metrics if m not in OUTPUT_METRICS]
    if unknown:
        raise ValueError(f"Unknown output metric(s): {unknown}. Valid: {list(OUTPUT_METRICS)}")

    data = {
        "question": [record.query],
        "answer": [record.response],
        "contexts": [[c.content for c in record.chunks]],
    }
    if ground_truth:
        data["ground_truth"] = [ground_truth]

    metric_objects = [getattr(ragas_metrics, name) for name in metrics]

    dataset = Dataset.from_dict(data)
    result = evaluate(dataset, metrics=metric_objects)

    scores: dict = {name: None for name in OUTPUT_METRICS}
    scores["evaluator"] = "ragas"
    scores["model"] = "unknown"

    for key in metrics:
        try:
            val = result[key]
            if isinstance(val, (list, tuple)) and len(val) > 0:
                scores[key] = float(val[0])
            elif isinstance(val, (int, float)):
                scores[key] = float(val)
        except (KeyError, TypeError, IndexError):
            pass

    return scores
