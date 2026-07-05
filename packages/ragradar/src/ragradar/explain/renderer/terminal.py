from ragradar_core.schema import RunRecord
from ragradar_evaluate.policy.persistence import load_policy
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ragradar.explain.analyzers import (
    cache as cache_mod,
)
from ragradar.explain.analyzers import (
    duplicates as duplicates_mod,
)
from ragradar.explain.analyzers import (
    history as history_mod,
)
from ragradar.explain.analyzers import (
    scores as scores_mod,
)
from ragradar.explain.analyzers import (
    semantic_cache as semantic_cache_mod,
)
from ragradar.explain.analyzers import (
    tokens as tokens_mod,
)
from ragradar.explain.analyzers import (
    truncation as truncation_mod,
)

console = Console()


def _render_tokens(result: dict, full: bool) -> Panel:
    pct = result["utilisation_pct"]
    style = "green" if pct < 80 else "yellow" if pct < 95 else "red"

    lines = []
    if result["model_limit"]:
        lines.append(
            f"[{style}]Total: {result['total_tokens']}/{result['model_limit']} ({pct}%)[/{style}]"
        )
    else:
        lines.append(f"Total: {result['total_tokens']} tokens")
    lines.append(f"  Chunks:   {result['chunks_tokens']}")
    lines.append(f"  History:  {result['history_tokens']}")
    lines.append(f"  System:   {result['system_tokens']}")
    lines.append(f"  Headroom: {result['headroom']}")

    if full and result["per_chunk"]:
        lines.append("")
        for pc in result["per_chunk"]:
            lines.append(f"  {pc['chunk_id']}: {pc['token_count']} tokens")

    return Panel("\n".join(lines), title="Token Usage", border_style=style)


def _render_scores(result: dict, full: bool) -> Panel:
    low = result["low_score_ratio"]
    style = "green" if low < 0.1 else "yellow" if low < 0.3 else "red"

    lines = []
    if result["retrieval_scores"]:
        lines.append(f"Retrieval: {result['bottom_retrieval']:.2f}-{result['top_retrieval']:.2f}")
    if result["rerank_scores"]:
        lines.append(f"Rerank:    {result['bottom_rerank']:.2f}-{result['top_rerank']:.2f}")
    if result["rerank_delta"] is not None:
        lines.append(f"Rerank delta:  {result['rerank_delta']:+.4f}")
    lines.append(f"Low-score: {low:.0%}")

    if full:
        if result["retrieval_scores"]:
            vals = ", ".join(f"{s:.2f}" for s in result["retrieval_scores"])
            lines.append(f"\nRetrieval: {vals}")
        if result["rerank_scores"]:
            vals = ", ".join(f"{s:.2f}" for s in result["rerank_scores"])
            lines.append(f"Rerank:    {vals}")

    return Panel("\n".join(lines), title="Chunk Scores", border_style=style)


def _render_duplicates(result: dict, full: bool) -> Panel:
    ratio = result["duplicate_ratio"]
    style = "green" if ratio == 0 else "yellow" if ratio <= 0.2 else "red"

    lines = [
        f"Path dups:     {len(result['path_dups'])}",
        f"Window dups:   {len(result['window_dups'])}",
        "Semantic dups: (deferred)",
        f"Ratio:         {ratio:.0%}",
    ]

    if full:
        for d in result["path_dups"]:
            lines.append(f"  [PATH DUP] {d['chunk_id']} via {', '.join(d['paths'])}")
        for d in result["window_dups"]:
            ids = ", ".join(d["chunk_ids"])
            lines.append(f"  [WINDOW DUP] {ids} (source: {d['source_doc_id']})")

    return Panel("\n".join(lines), title="Duplicate Chunks", border_style=style)


def _render_truncation(result: dict, full: bool) -> Panel:
    sev = result["severity"]
    style = "green" if sev == "none" else "yellow" if sev == "low" else "red"

    lines = [
        f"Truncated: {result['truncated_count']} chunks",
        f"High-score truncations: {result['high_score_truncations']}",
        f"Severity: [{style}]{sev}[/{style}]",
    ]

    if full:
        for tc in result["truncated_chunks"]:
            lines.append(
                f"  {tc['chunk_id']}: retrieval={tc['score']}, rerank={tc['rerank_score']}"
            )

    return Panel("\n".join(lines), title="Truncation", border_style=style)


def _render_history(result: dict, full: bool) -> Panel:
    dropped = result["dropped_turn_count"]
    style = "green" if dropped == 0 else "yellow" if dropped <= 2 else "red"

    lines = [
        f"Turns: {result['pre_turn_count']} -> {result['post_turn_count']} ({dropped} dropped)",
    ]
    if result["eviction_reason"]:
        lines.append(f"Reason: {result['eviction_reason']}")
    if result["pre_tokens"] is not None:
        lines.append(f"Pre tokens:  {result['pre_tokens']}")
    if result["post_tokens"] is not None:
        lines.append(f"Post tokens: {result['post_tokens']}")

    if full and result["dropped_turns"]:
        lines.append("\nDropped turns:")
        for t in result["dropped_turns"]:
            lines.append(f"  [{t.role}] {t.content[:100]}")

    return Panel("\n".join(lines), title="Dropped History", border_style=style)


def _render_cache(result: dict, full: bool) -> Panel:
    ratio = result["hit_ratio"]
    style = "green" if ratio > 0.7 else "yellow" if ratio > 0.3 else "red"

    lines = [f"Hits: {result['hits']}/{result['total_events']} ({ratio:.0%})"]

    if full:
        if result["hit_chunks"]:
            lines.append(f"Hit:  {', '.join(result['hit_chunks'])}")
        if result["miss_chunks"]:
            lines.append(f"Miss: {', '.join(result['miss_chunks'])}")

    return Panel("\n".join(lines), title="Cache Hits", border_style=style)


def _render_semantic_cache(result: dict, full: bool) -> Panel:
    if not result["checked"]:
        return Panel("Not checked", title="Cache behavior", border_style="white")

    flags = []
    if result["hit"] and result["borderline_hit"]:
        flags.append(
            "Borderline hit: similarity landed within the borderline margin of the "
            "threshold — this may have reused a near-miss cached answer."
        )
    if result["hit"] and result["stale_hit"]:
        flags.append(
            f"Stale hit: cached answer is {result['age_seconds']:.0f}s old, past the "
            "configured max age — it may no longer be accurate."
        )

    style = "green" if result["hit"] and not flags else "yellow" if flags else "white"

    lines = [f"Checked: yes -> {'hit' if result['hit'] else 'miss'}"]
    if result["similarity_score"] is not None and result["threshold"] is not None:
        lines.append(
            f"Similarity: {result['similarity_score']:.4f} (threshold {result['threshold']:.4f})"
        )
    if result["hit"] and not result["registered"]:
        lines.append("Not registered in the cache index.")
    for flag in flags:
        lines.append(f"[yellow]{flag}[/yellow]")

    if full and result["cached_query"]:
        lines.append(f"\nCached query: {result['cached_query']}")

    return Panel("\n".join(lines), title="Cache behavior", border_style=style)


_ANALYZERS = [
    (tokens_mod, _render_tokens),
    (scores_mod, _render_scores),
    (duplicates_mod, _render_duplicates),
    (truncation_mod, _render_truncation),
    (history_mod, _render_history),
    (cache_mod, _render_cache),
]


def _render_eval_scores(run_row: dict) -> Panel | None:
    import json as _json

    raw = run_row.get("eval_scores")
    if not raw:
        return None
    scores = _json.loads(raw) if isinstance(raw, str) else raw
    risk = run_row.get("risk_score")
    evaluated_at = run_row.get("evaluated_at")

    lines = []

    if risk is not None:
        style = "green" if risk < 0.3 else "yellow" if risk <= 0.7 else "red"
        lines.append(f"Risk score:         [{style}]{risk:.4f}[/{style}]")

    inp = scores.get("input")
    if inp:
        violations = inp.get("policy_violations", [])
        if violations:
            lines.append(f"Input quality:      [red]{', '.join(sorted(violations))}[/red]")
        else:
            lines.append("Input quality:      [green]all checks passed[/green]")

    out = scores.get("output")
    if out and out.get("error") is None:
        for key, label in [
            ("faithfulness", "Faithfulness"),
            ("answer_relevancy", "Answer relevancy"),
            ("context_precision", "Context precision"),
            ("context_recall", "Context recall"),
        ]:
            val = out.get(key)
            if val is not None:
                lines.append(f"{label + ':':<20}{val:.4f}")

    if evaluated_at:
        lines.append(f"Evaluated at:       {evaluated_at}")

    if not lines:
        return None

    return Panel("\n".join(lines), title="Evaluation Scores", border_style="cyan")


def render(record: RunRecord, full: bool = False, run_row: dict | None = None) -> None:
    console.print()
    console.print(f"[bold]Query:[/bold] {record.query}")
    resp = record.response
    if len(resp) > 200 and not full:
        resp = resp[:200] + "..."
    console.print(f"[bold]Response:[/bold] {resp}")
    if record.model:
        console.print(f"[bold]Model:[/bold] {record.model}")
    console.print()

    for mod, renderer in _ANALYZERS:
        result = mod.analyze(record)
        if result is not None:
            console.print(renderer(result, full))

    if record.cache is not None:
        pipeline_key = (run_row or {}).get("pipeline") or "__default"
        policy = load_policy(pipeline_key)
        cache_result = semantic_cache_mod.analyze(record, policy)
        if cache_result is not None:
            console.print(_render_semantic_cache(cache_result, full))

    if run_row is not None:
        eval_panel = _render_eval_scores(run_row)
        if eval_panel is not None:
            console.print(eval_panel)

    if record.final_prompt:
        if full:
            text = record.final_prompt
        else:
            text = record.final_prompt[:500]
            if len(record.final_prompt) > 500:
                text += f"\n... ({len(record.final_prompt) - 500} chars truncated)"
        console.print(Panel(text, title="Final Prompt", border_style="blue"))


def render_budget(record: RunRecord) -> None:
    result = tokens_mod.analyze(record)
    if result is None:
        console.print("No token budget data available for this run.")
        return
    console.print()
    console.print(_render_tokens(result, full=True))


def render_diff(rec_a: RunRecord, rec_b: RunRecord, id_a: str, id_b: str) -> None:
    console.print()
    console.print(f"[bold]Comparing {id_a} vs {id_b}[/bold]\n")

    # Query delta
    tbl = Table(title="Query")
    tbl.add_column(id_a)
    tbl.add_column(id_b)
    tbl.add_row(rec_a.query, rec_b.query)
    console.print(tbl)

    # Chunks delta
    chunks_a = {c.chunk_id for c in (rec_a.chunks or [])}
    chunks_b = {c.chunk_id for c in (rec_b.chunks or [])}
    added = chunks_b - chunks_a
    removed = chunks_a - chunks_b

    if chunks_a or chunks_b:
        tbl = Table(title="Chunks")
        tbl.add_column("Metric")
        tbl.add_column(id_a, justify="right")
        tbl.add_column(id_b, justify="right")
        tbl.add_row("Count", str(len(chunks_a)), str(len(chunks_b)))
        if added:
            tbl.add_row("Added", "", ", ".join(sorted(added)))
        if removed:
            tbl.add_row("Removed", ", ".join(sorted(removed)), "")
        console.print(tbl)

    # Score delta for shared chunks
    shared = chunks_a & chunks_b
    if shared:
        map_a = {c.chunk_id: c for c in rec_a.chunks}
        map_b = {c.chunk_id: c for c in rec_b.chunks}
        tbl = Table(title="Score Delta (shared chunks)")
        tbl.add_column("Chunk")
        tbl.add_column(f"Ret ({id_a})", justify="right")
        tbl.add_column(f"Ret ({id_b})", justify="right")
        tbl.add_column(f"Rer ({id_a})", justify="right")
        tbl.add_column(f"Rer ({id_b})", justify="right")
        for cid in sorted(shared):
            ca, cb = map_a[cid], map_b[cid]
            tbl.add_row(
                cid,
                f"{ca.retrieval_score:.2f}" if ca.retrieval_score is not None else "-",
                f"{cb.retrieval_score:.2f}" if cb.retrieval_score is not None else "-",
                f"{ca.rerank_score:.2f}" if ca.rerank_score is not None else "-",
                f"{cb.rerank_score:.2f}" if cb.rerank_score is not None else "-",
            )
        console.print(tbl)

    # Token budget delta
    ba, bb = rec_a.token_budget, rec_b.token_budget
    if ba or bb:
        tbl = Table(title="Token Budget")
        tbl.add_column("Metric")
        tbl.add_column(id_a, justify="right")
        tbl.add_column(id_b, justify="right")
        for attr in [
            "total_limit",
            "chunks_allocated",
            "history_allocated",
            "system_allocated",
            "headroom",
        ]:
            va = str(getattr(ba, attr)) if ba else "-"
            vb = str(getattr(bb, attr)) if bb else "-"
            tbl.add_row(attr, va, vb)
        console.print(tbl)

    # History delta
    ha_pre = len(rec_a.history_pre or [])
    hb_pre = len(rec_b.history_pre or [])
    ha_post = len(rec_a.history_post or [])
    hb_post = len(rec_b.history_post or [])
    if ha_pre or hb_pre or ha_post or hb_post:
        tbl = Table(title="History")
        tbl.add_column("Metric")
        tbl.add_column(id_a, justify="right")
        tbl.add_column(id_b, justify="right")
        tbl.add_row("Pre turns", str(ha_pre), str(hb_pre))
        tbl.add_row("Post turns", str(ha_post), str(hb_post))
        tbl.add_row("Dropped", str(ha_pre - ha_post), str(hb_pre - hb_post))
        console.print(tbl)

    # Truncation delta
    ta = truncation_mod.analyze(rec_a)
    tb = truncation_mod.analyze(rec_b)
    if ta or tb:
        tbl = Table(title="Truncation")
        tbl.add_column("Metric")
        tbl.add_column(id_a, justify="right")
        tbl.add_column(id_b, justify="right")
        tbl.add_row(
            "Truncated",
            str(ta["truncated_count"]) if ta else "0",
            str(tb["truncated_count"]) if tb else "0",
        )
        tbl.add_row(
            "Severity",
            ta["severity"] if ta else "none",
            tb["severity"] if tb else "none",
        )
        console.print(tbl)
