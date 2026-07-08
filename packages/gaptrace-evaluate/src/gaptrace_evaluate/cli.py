import re
from dataclasses import fields
from typing import get_type_hints

import click
from gaptrace_core import store
from gaptrace_core.targets import parse_target_id
from rich.console import Console
from rich.table import Table

from gaptrace_evaluate import evaluate
from gaptrace_evaluate.benchmark import builder, checker, exporter, seeder
from gaptrace_evaluate.facade import INPUT_METRICS, OUTPUT_METRICS
from gaptrace_evaluate.policy.persistence import load_policy, reset_policy, save_policy
from gaptrace_evaluate.policy.schema import InputQualityPolicy

console = Console()
_SESSION_RE = re.compile(r"^s(\d+)$", re.IGNORECASE)


def _parse_session_id(value: str) -> int:
    m = _SESSION_RE.match(value)
    if m:
        return int(m.group(1))
    return int(value)


def _resolve_target(target: str | None = None) -> dict | None:
    if target is None:
        return store.get_latest_run()
    try:
        parsed = parse_target_id(target)
    except ValueError:
        return None
    return store.get_run(*parsed)


def _metrics_for_flags(input_only: bool, output_only: bool) -> list | None:
    """Map the CLI flags onto the facade's metric selection. Pure.

    Both flags together is a caller error (the old code silently computed
    nothing); the command errors out before calling this with both set.
    """
    if input_only:
        return list(INPUT_METRICS)
    if output_only:
        return list(OUTPUT_METRICS)
    return None


def _fmt(val) -> str:
    if val is None:
        return "-"
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def _render_eval_result(run_row, eval_scores: dict, risk_score: float | None):
    run_id = f"s{run_row['session_id']}r{run_row['run_seq']}"

    console.print(f"\n[bold]{run_id}[/bold] -- {run_row['query'][:60]}")
    if risk_score is None:
        console.print("Risk score: -")
    else:
        risk_style = "green" if risk_score < 0.3 else "yellow" if risk_score < 0.7 else "red"
        console.print(f"Risk score: [{risk_style}]{risk_score:.2f}[/{risk_style}]")

    inp = eval_scores.get("input")
    if inp:
        passes = inp.get("passes_policy", True)
        header_style = "green" if passes else "red"

        tbl = Table(title=f"[{header_style}]Input Quality[/{header_style}]")
        tbl.add_column("Factor")
        tbl.add_column("Value", justify="right")

        for key in [
            "mean_relevance",
            "top_chunk_score",
            "duplicate_ratio",
            "low_score_chunk_ratio",
            "token_headroom_pct",
            "source_domain_count",
            "truncation_severity",
            "high_score_truncations",
        ]:
            val = inp.get(key)
            if val is not None:
                tbl.add_row(key, _fmt(val))
        console.print(tbl)

        violations = inp.get("policy_violations", [])
        if violations:
            console.print(f"[red]Policy violations: {', '.join(sorted(violations))}[/red]")
        else:
            console.print("[green]All policy checks passed.[/green]")

    out = eval_scores.get("output")
    if out and out.get("error") is None:
        tbl = Table(title="Output Quality (RAGAS)")
        tbl.add_column("Metric")
        tbl.add_column("Score", justify="right")
        for key in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            val = out.get(key)
            if val is None:
                continue
            style = "green" if val > 0.7 else "yellow" if val > 0.5 else "red"
            tbl.add_row(key, f"[{style}]{val:.4f}[/{style}]")
        console.print(tbl)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@click.group()
def main():
    """gaptrace-evaluate -- evaluation layer for gaptrace observability system."""
    store.ensure_store()


@main.command("run")
@click.argument("target", required=False)
@click.option("--input-only", is_flag=True)
@click.option("--output-only", is_flag=True)
@click.option("--session", "session_filter", default=None)
@click.option("--ground-truth", default=None)
@click.option("--pipeline", default=None)
def run_cmd(target, input_only, output_only, session_filter, ground_truth, pipeline):
    """Evaluate a run or all runs in a session."""
    if input_only and output_only:
        console.print(
            "[red]--input-only and --output-only exclude each other; "
            "drop both flags for a complete evaluation.[/red]"
        )
        raise SystemExit(1)

    metrics = _metrics_for_flags(input_only, output_only)

    if session_filter:
        sid = _parse_session_id(session_filter)
        runs = store.get_runs_in_session(sid)
        if not runs:
            console.print(f"No runs found in session {sid}.")
            return
        policy_cache: dict = {}
        computed = []
        for run_row in runs:
            pipeline_key = pipeline or run_row["pipeline"] or "__default"
            if pipeline_key not in policy_cache:
                policy_cache[pipeline_key] = load_policy(pipeline_key)
            result = evaluate(
                f"s{run_row['session_id']}r{run_row['run_seq']}",
                metrics=metrics,
                ground_truth=ground_truth,
                policy=policy_cache[pipeline_key],
                save=False,
            )
            _warn_errors(result)
            computed.append((run_row, result))
        store.write_eval_scores_batch(
            [
                (
                    run_row["session_id"],
                    run_row["run_seq"],
                    result.to_eval_scores(),
                    result.risk_score,
                )
                for run_row, result in computed
            ]
        )
        for run_row, result in computed:
            _render_eval_result(run_row, result.to_eval_scores(), result.risk_score)
    else:
        run_row = _resolve_target(target)
        if run_row is None:
            console.print("No runs found.")
            return
        result = evaluate(
            f"s{run_row['session_id']}r{run_row['run_seq']}",
            metrics=metrics,
            ground_truth=ground_truth,
            pipeline=pipeline,
            save=True,
        )
        _warn_errors(result)
        _render_eval_result(run_row, result.to_eval_scores(), result.risk_score)


def _warn_errors(result) -> None:
    """Print each metric error once (RAGAS missing/runtime failures)."""
    for msg in dict.fromkeys(result.errors.values()):
        console.print(f"[yellow]{msg}[/yellow]")


# ---------------------------------------------------------------------------
# Benchmark subgroup
# ---------------------------------------------------------------------------


@main.group()
def benchmark():
    """Benchmark commands."""


@benchmark.command("build")
@click.option("--pipeline", default=None)
def benchmark_build(pipeline):
    """Build correlation model from evaluated runs."""
    try:
        result = builder.build(pipeline)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise SystemExit(1)

    tbl = Table(title=f"Benchmark ({result['run_count']} runs)")
    tbl.add_column("Factor")
    tbl.add_column("Threshold", justify="right")
    tbl.add_column("Correlation", justify="right")
    tbl.add_column("Samples", justify="right")

    for factor, data in result["factors"].items():
        corr_vals = [v for k, v in data.items() if "correlation" in k and v is not None]
        best_corr = max(corr_vals, key=abs) if corr_vals else None
        tbl.add_row(
            factor,
            _fmt(data["suggested_threshold"]),
            _fmt(best_corr),
            str(data["sample_count"]),
        )
    console.print(tbl)


@benchmark.command("show")
@click.option("--pipeline", default=None)
def benchmark_show(pipeline):
    """Show benchmark thresholds and correlations."""
    pipeline_key = pipeline or "__default"
    entries = store.get_benchmark(pipeline_key)
    if not entries:
        console.print("No benchmark data found.")
        return

    tbl = Table(title=f"Benchmark: {pipeline_key}")
    tbl.add_column("Factor")
    tbl.add_column("Threshold", justify="right")
    tbl.add_column("Correlation", justify="right")
    tbl.add_column("Samples", justify="right")
    tbl.add_column("Updated")

    for e in entries:
        tbl.add_row(
            e["factor"],
            _fmt(e["threshold"]),
            _fmt(e["correlation"]),
            str(e["sample_count"]),
            e["updated_at"][:10],
        )
    console.print(tbl)


@benchmark.command("check")
@click.argument("target")
@click.option("--pipeline", default=None)
def benchmark_check(target, pipeline):
    """Check a run against benchmark thresholds."""
    try:
        parsed = parse_target_id(target)
        result = checker.check(*parsed, pipeline)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)

    overall_style = {"ok": "green", "warn": "yellow", "fail": "red"}[result["overall"]]
    console.print(
        f"\n[bold]{result['run_id']}[/bold] -- overall: [{overall_style}]{result['overall']}[/{overall_style}]"
    )
    if result["risk_score"] is None:
        console.print("Risk score: -")
    else:
        console.print(f"Risk score: {result['risk_score']:.2f}")

    tbl = Table(title="Factor Check")
    tbl.add_column("Factor")
    tbl.add_column("Value", justify="right")
    tbl.add_column("Threshold", justify="right")
    tbl.add_column("Status")

    for factor, data in result["factors"].items():
        style = {"ok": "green", "warn": "yellow", "fail": "red"}[data["status"]]
        tbl.add_row(
            factor,
            _fmt(data["value"]),
            _fmt(data["benchmark_threshold"]),
            f"[{style}]{data['status']}[/{style}]",
        )
    console.print(tbl)


@benchmark.command("seed")
@click.argument("pipeline")
@click.option("--count", default=20, type=int)
def benchmark_seed(pipeline, count):
    """Generate synthetic runs as day-zero baseline."""
    n = seeder.seed(pipeline, count)
    console.print(f"Seeded {n} synthetic runs for pipeline '{pipeline}'.")


@benchmark.command("export")
@click.option("--pipeline", default=None)
@click.option("--output", default=None)
def benchmark_export(pipeline, output):
    """Export evaluated runs as RAGAS-compatible JSONL."""
    from pathlib import Path

    path = exporter.export(pipeline, Path(output) if output else None)
    console.print(f"Exported to {path}")


# ---------------------------------------------------------------------------
# Policy subgroup
# ---------------------------------------------------------------------------


@main.group()
def policy():
    """Policy management commands."""


@policy.command("show")
@click.option("--pipeline", default=None)
def policy_show(pipeline):
    """Show current policy values."""
    pipeline_key = pipeline or "__default"
    pol = load_policy(pipeline_key)
    default = InputQualityPolicy.default()

    tbl = Table(title=f"Policy: {pipeline_key}")
    tbl.add_column("Field")
    tbl.add_column("Value", justify="right")
    tbl.add_column("Default", justify="right")

    for f in fields(InputQualityPolicy):
        val = getattr(pol, f.name)
        dval = getattr(default, f.name)
        style = "" if val == dval else "[bold]"
        end = "" if val == dval else "[/bold]"
        tbl.add_row(f.name, f"{style}{val}{end}", str(dval))
    console.print(tbl)


@policy.command("set")
@click.argument("field")
@click.argument("value")
@click.option("--pipeline", default=None)
def policy_set(field, value, pipeline):
    """Set a policy field value."""
    pipeline_key = pipeline or "__default"

    valid_fields = {f.name: f for f in fields(InputQualityPolicy)}
    if field not in valid_fields:
        console.print(f"[red]Unknown field: {field}[/red]")
        console.print(f"Valid fields: {', '.join(sorted(valid_fields))}")
        raise SystemExit(1)

    f = valid_fields[field]
    field_type = get_type_hints(InputQualityPolicy)[f.name]
    try:
        typed_value = field_type(value)
    except (ValueError, TypeError):
        console.print(
            f"[red]Invalid value for {field} (expected {field_type.__name__}): {value}[/red]"
        )
        raise SystemExit(1)

    pol = load_policy(pipeline_key)
    setattr(pol, field, typed_value)
    save_policy(pipeline_key, pol)
    console.print(f"Set {field} = {typed_value} for pipeline '{pipeline_key}'.")


@policy.command("reset")
@click.option("--pipeline", default=None)
def policy_reset(pipeline):
    """Reset policy to defaults."""
    pipeline_key = pipeline or "__default"
    reset_policy(pipeline_key)
    console.print(f"Policy reset to defaults for pipeline '{pipeline_key}'.")
