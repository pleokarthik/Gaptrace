"""
Evaluate captured runs with the two ragradar user tasks:

  check(run_id)     -- "is this run healthy?"  free, deterministic, instant
  evaluate(run_id)  -- "score it fully" (or any subset of metrics)

Runs standalone: it captures one demonstration run first (run
02_capture_patterns.py beforehand if you want more runs to browse with
`ragradar list`).
"""

import ragradar
from rich.console import Console
from rich.table import Table

console = Console()
PIPELINE = "rag_example"


def capture_demo_run() -> str:
    """Capture one run with deliberately mixed-quality retrieval.

    Plain dicts throughout — no schema types needed; headroom is derived
    from the budget allocations.
    """
    run_id = ragradar.capture(
        "what is RRF and how does it normalize scores?",
        "RRF replaces raw scores with rank-based reciprocal values.",
        pipeline=PIPELINE,
        # requested_chunk_count=4 matches the 4 chunks below exactly --
        # a clean score_underfill reading (underfill_ratio 0.0), contrast
        # with 02_capture_patterns.py's under-filled demo run.
        requested_chunk_count=4,
        chunks=[
            {
                "chunk_id": "rrf_1",
                "source_doc_id": "rrf_paper",
                "content": "Reciprocal Rank Fusion normalizes scores from different retrieval systems",
                "token_count": 180,
                "retrieval_score": 0.85,
                "rerank_score": 0.92,
                "retrieval_path": "hybrid",
            },
            {
                "chunk_id": "rrf_2",
                "source_doc_id": "rrf_paper",
                "content": (
                    "Reciprocal Rank Fusion normalizes scores from different "
                    "retrieval systems and ranks documents accordingly."
                ),
                "token_count": 160,
                "retrieval_score": 0.71,
                "rerank_score": 0.78,
                "retrieval_path": "bm25",
            },
            {
                "chunk_id": "bm25_1",
                "source_doc_id": "ir_textbook",
                "content": "BM25 computes relevance using term frequency and inverse document frequency.",
                "token_count": 140,
                "retrieval_score": 0.82,
                "rerank_score": 0.88,
                "retrieval_path": "bm25",
                "truncated": True,
            },
            {
                "chunk_id": "win_1",
                "source_doc_id": "rag_patterns",
                "content": "Context window management determines which chunks survive token budgets.",
                "token_count": 145,
                "retrieval_score": 0.48,
                "rerank_score": 0.39,
                "retrieval_path": "bm25",
            },
        ],
        token_budget={
            "total_limit": 4096,
            "chunks_allocated": 2800,
            "history_allocated": 600,
            "system_allocated": 500,
        },
        model="gpt-4-turbo",
    )
    console.print(f"Captured [cyan]{run_id}[/cyan]")
    return run_id


def show_check(run_id: str) -> None:
    """Task 1: is this run healthy? Call before paying for an LLM."""
    result = ragradar.check(run_id)

    style = {"ok": "green", "warn": "yellow", "fail": "red"}[result.verdict]
    risk = "-" if result.risk_score is None else f"{result.risk_score:.2f}"
    console.print(
        f"\n[bold]check({run_id})[/bold] -> [{style}]{result.verdict}[/{style}]  "
        f"risk: {risk}  (standards: {result.thresholds})"
    )
    for problem in result.problems:
        console.print(f"  [red]-[/red] {problem}")

    tbl = Table(title="Current standards")
    tbl.add_column("Factor")
    tbl.add_column("Value", justify="right")
    tbl.add_column("Threshold", justify="right")
    tbl.add_column("Status")
    for factor, data in result.factors.items():
        fstyle = "green" if data["status"] == "ok" else "red"
        tbl.add_row(
            factor,
            "-" if data["value"] is None else f"{data['value']:.4f}",
            "-" if data["threshold"] is None else f"{data['threshold']:.4f}",
            f"[{fstyle}]{data['status']}[/{fstyle}]",
        )
    console.print(tbl)


def show_single_metric(run_id: str) -> None:
    """Atomic selection: compute exactly one cheap metric, nothing else."""
    result = ragradar.evaluate(run_id, metrics=["duplicates"], save=False)
    dup = result.metrics["duplicates"]
    console.print(
        f'\n[bold]evaluate({run_id}, metrics=["duplicates"])[/bold] -> '
        f"ratio {dup['duplicate_ratio']:.2f} "
        f"({dup['path_dup_count']} path, {dup['window_dup_count']} window)"
    )


def show_full_evaluate(run_id: str) -> None:
    """Complete eval: every applicable metric; scores persist on the run."""
    result = ragradar.evaluate(run_id)

    console.print(f"\n[bold]evaluate({run_id})[/bold] -- saved: {result.saved}")
    risk = "-" if result.risk_score is None else f"{result.risk_score:.2f}"
    console.print(f"Risk score: {risk}")
    if result.policy_violations:
        console.print(f"Policy violations: {', '.join(sorted(result.policy_violations))}")

    tbl = Table(title="Metric results")
    tbl.add_column("Metric")
    tbl.add_column("Result")
    for name, value in result.metrics.items():
        if isinstance(value, dict):
            summary = ", ".join(f"{k}={v}" for k, v in list(value.items())[:3])
        else:
            summary = "-" if value is None else f"{value:.4f}"
        tbl.add_row(name, summary)
    console.print(tbl)

    # RAGAS metrics cost LLM calls; without a configured judge they land
    # in result.errors (never an exception) and the free metrics above
    # are unaffected.
    for name, err in result.errors.items():
        console.print(f"[yellow]{name}: {err[:80]}[/yellow]")
    for name, reason in result.skipped.items():
        if reason != "not requested":
            console.print(f"[dim]{name} skipped: {reason}[/dim]")


def show_available_metrics() -> None:
    tbl = Table(title="ragradar.available_metrics()")
    tbl.add_column("Metric")
    tbl.add_column("Layer")
    tbl.add_column("Cost")
    tbl.add_column("Requires")
    for name, info in ragradar.available_metrics().items():
        tbl.add_row(name, info.layer, info.cost, ", ".join(info.requires))
    console.print(tbl)


if __name__ == "__main__":
    run_id = capture_demo_run()
    show_check(run_id)
    show_single_metric(run_id)
    show_full_evaluate(run_id)
    show_available_metrics()
    console.print(
        "\nNext: [cyan]ragradar explain "
        f"{run_id}[/cyan] shows these scores alongside the run analysis; "
        "[cyan]ragradar-evaluate benchmark export[/cyan] writes a RAGAS-compatible "
        "dataset of everything evaluated."
    )
