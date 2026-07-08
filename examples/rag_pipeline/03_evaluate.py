"""
Evaluate captured runs with the two gaptrace user tasks:

  check(run_id)     -- "is this run healthy?"  free, deterministic, instant
  evaluate(run_id)  -- "score it fully" (or any subset of metrics)

Runs standalone: it captures one demonstration run first (run
02_capture_patterns.py beforehand if you want more runs to browse with
`gaptrace list`).
"""

import gaptrace
from rich.console import Console
from rich.table import Table

console = Console()
PIPELINE = "rag_example"


def capture_demo_run() -> str:
    """Capture one run with deliberately mixed-quality retrieval, staged
    end-to-end so every gaptrace explain panel that *can* fire on this
    record shape has data to render.

    Matches the field breadth of 02_capture_patterns.py's
    pattern_full_fields() (metadata filter, chunks, context, history,
    cache, tool call) so this run — the reader's actual latest run,
    since 03 runs after 02 in the documented order — lights up the same
    panels rather than the thinner set the one-liner capture() used to
    produce. Literal reuse of pattern_full_fields()/_sample_chunks() isn't
    possible since "02_capture_patterns" starts with a digit and can't be
    imported as a plain Python module, so the same field coverage is
    mirrored here via the staged Capture API instead of the one-liner.

    requested_chunk_count=4 exactly matches the 4 chunks below (a clean
    score_underfill reading, underfill_ratio 0.0) — deliberately
    contrasting with 02_capture_patterns.py's under-filled demo run.
    Cache behavior (semantic_cache) is the one panel that structurally
    can't fire here: no example script calls cap.semantic_cache().
    """
    cap = gaptrace.start("what is RRF and how does it normalize scores?", pipeline=PIPELINE)

    # Metadata filter runs before retrieval; excluded candidates never reach scoring.
    cap.metadata_filter(
        applied=True,
        candidate_count=6,
        excluded_count=2,
        filters={"source": "internal"},
    )

    chunks = [
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
    ]
    # requested_count=4 matches len(chunks) exactly -- see docstring.
    cap.chunks(chunks, requested_count=4)

    prompt = (
        "System: answer using context.\n\nContext:\n"
        + "\n".join(f"[{i}] {c['content']}" for i, c in enumerate(chunks, 1))
        + "\n\nQuery: what is RRF and how does it normalize scores?"
    )
    cap.context(
        prompt,
        {
            "total_limit": 4096,
            "chunks_allocated": 2800,
            "history_allocated": 600,
            "system_allocated": 500,
        },
    )

    cap.history(
        pre=[
            {"user": "Can you summarize RRF for me first?"},
            {"assistant": "Sure -- RRF combines rankings from multiple systems."},
            {"user": "Now explain how it normalizes scores."},
        ],
        post=[
            {"user": "Now explain how it normalizes scores."},
        ],
        eviction_reason="token_budget",
    )

    cap.cache({"rrf_1": True, "rrf_2": False, "bm25_1": False, "win_1": False})

    cap.tool_call(
        {
            "tool_name": "rerank",
            "arguments": {"chunk_ids": [c["chunk_id"] for c in chunks]},
            "result": f"reranked {len(chunks)} chunks",
            "latency_ms": 38.0,
        }
    )

    run_id = cap.response(
        "RRF replaces raw scores with rank-based reciprocal values.",
        token_usage={"input_tokens": 1850, "output_tokens": 40},
        model="gpt-4-turbo",
    )
    console.print(f"Captured [cyan]{run_id}[/cyan]")
    return run_id


def show_check(run_id: str) -> None:
    """Task 1: is this run healthy? Call before paying for an LLM."""
    result = gaptrace.check(run_id)

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
    result = gaptrace.evaluate(run_id, metrics=["duplicates"], save=False)
    dup = result.metrics["duplicates"]
    console.print(
        f'\n[bold]evaluate({run_id}, metrics=["duplicates"])[/bold] -> '
        f"ratio {dup['duplicate_ratio']:.2f} "
        f"({dup['path_dup_count']} path, {dup['window_dup_count']} window)"
    )


def show_full_evaluate(run_id: str) -> None:
    """Complete eval: every applicable metric; scores persist on the run."""
    result = gaptrace.evaluate(run_id)

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
    tbl = Table(title="gaptrace.available_metrics()")
    tbl.add_column("Metric")
    tbl.add_column("Layer")
    tbl.add_column("Cost")
    tbl.add_column("Requires")
    for name, info in gaptrace.available_metrics().items():
        tbl.add_row(name, info.layer, info.cost, ", ".join(info.requires))
    console.print(tbl)


if __name__ == "__main__":
    run_id = capture_demo_run()
    show_check(run_id)
    show_single_metric(run_id)
    show_full_evaluate(run_id)
    show_available_metrics()
    console.print(
        "\nNext: [cyan]gaptrace explain "
        f"{run_id}[/cyan] shows these scores alongside the run analysis; "
        "[cyan]gaptrace-evaluate benchmark export[/cyan] writes a RAGAS-compatible "
        "dataset of everything evaluated."
    )
