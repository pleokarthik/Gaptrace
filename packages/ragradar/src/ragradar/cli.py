import re
from datetime import date

import click
from rich.console import Console
from rich.table import Table

from ragradar import store
from ragradar.explain import loader
from ragradar.explain.renderer import html as html_renderer
from ragradar.explain.renderer import terminal as terminal_renderer

console = Console()

_SESSION_RE = re.compile(r"^s(\d+)$", re.IGNORECASE)


def _parse_session_id(value: str) -> int:
    m = _SESSION_RE.match(value)
    if m:
        return int(m.group(1))
    return int(value)


def _disambiguate(results: list[dict]) -> dict | None:
    console.print("\n  [bold]Multiple matches:[/bold]\n")
    for i, r in enumerate(results, 1):
        title = r.get("session_title") or ""
        query_preview = r["query"][:60]
        console.print(
            f"  {i}   s{r['session_id']} r{r['run_seq']}   "
            f"{r['created_at'][:10]}   {title}   "
            f'— "{query_preview}"'
        )
    console.print()
    try:
        choice = click.prompt(
            "  Pick (number) or press Enter to cancel",
            default="",
            show_default=False,
        )
        if not choice:
            return None
        idx = int(choice) - 1
        if 0 <= idx < len(results):
            r = results[idx]
            return store.get_run(r["session_id"], r["run_seq"])
    except (ValueError, KeyboardInterrupt, EOFError):
        pass
    return None


def _resolve_and_load(target: str | None = None):
    result = store.resolve_target(target)
    if result is None:
        console.print("No runs found.")
        return None, None
    if isinstance(result, list):
        run_row = _disambiguate(result)
        if run_row is None:
            return None, None
    else:
        run_row = result
    record = loader.load_run_record(run_row)
    return run_row, record


@click.group()
def main():
    """ragradar — analyst CLI for the ragradar observability system."""


@main.command("list")
@click.argument("session_id", required=False)
def list_cmd(session_id):
    """List sessions, or runs within a session."""
    if session_id is not None:
        sid = _parse_session_id(session_id)
        runs = store.list_runs(sid)
        if not runs:
            console.print(f"No runs found in session {sid}.")
            return
        tbl = Table(title=f"Session {sid} — Runs")
        tbl.add_column("Run", style="cyan")
        tbl.add_column("Date")
        tbl.add_column("Query")
        for r in runs:
            tbl.add_row(
                f"s{r['session_id']} r{r['run_seq']}",
                r["created_at"][:10],
                r["query"][:80],
            )
        console.print(tbl)
    else:
        sessions = store.list_sessions()
        if not sessions:
            console.print("No sessions found.")
            return
        tbl = Table(title="Sessions")
        tbl.add_column("ID", style="cyan")
        tbl.add_column("Runs", justify="right")
        tbl.add_column("Pipeline")
        tbl.add_column("Created")
        tbl.add_column("Title")
        for s in sessions:
            tbl.add_row(
                f"s{s['session_id']}",
                str(s["run_count"]),
                s["pipeline"] or "",
                s["created_at"][:10],
                s["title"] or "",
            )
        console.print(tbl)


@main.command()
@click.argument("hint", required=False)
@click.option("--exact", is_flag=True)
@click.option("--from", "from_dt", default=None)
@click.option("--to", "to_dt", default=None)
@click.option("--today", is_flag=True)
@click.option("--session", "session_filter", default=None)
@click.option("--pipeline", default=None)
@click.option("--recent", default=None, type=int)
def find(hint, exact, from_dt, to_dt, today, session_filter, pipeline, recent):
    """Search runs by query text."""
    if today:
        today_str = date.today().isoformat()
        if from_dt is None:
            from_dt = today_str
        if to_dt is None:
            to_dt = today_str + "T23:59:59.999999Z"

    sid = None
    if session_filter is not None:
        sid = _parse_session_id(session_filter)

    results = store.search_runs(
        hint=hint,
        exact=exact,
        session_id=sid,
        pipeline=pipeline,
        from_dt=from_dt,
        to_dt=to_dt,
        recent_n=recent,
    )

    if not results:
        console.print("No matching runs found.")
        return

    tbl = Table(title=f"Search results ({len(results)})")
    tbl.add_column("Run", style="cyan")
    tbl.add_column("Date")
    tbl.add_column("Session")
    tbl.add_column("Query")
    for r in results:
        tbl.add_row(
            f"s{r['session_id']} r{r['run_seq']}",
            r["created_at"][:10],
            r.get("session_title") or "",
            r["query"][:80],
        )
    console.print(tbl)


@main.command()
@click.argument("target", required=False)
@click.option("--full", is_flag=True)
@click.option("--html", "to_html", is_flag=True)
def explain(target, full, to_html):
    """Explain a run — all analysis factors."""
    run_row, record = _resolve_and_load(target)
    if record is None:
        return

    if to_html:
        run_id = f"s{run_row['session_id']}r{run_row['run_seq']}"
        path = html_renderer.render(record, run_id)
        console.print(f"Report written to {path}")
    else:
        terminal_renderer.render(record, full=full, run_row=run_row)


@main.command()
@click.argument("target_a")
@click.argument("target_b")
def diff(target_a, target_b):
    """Compare two runs side by side."""
    row_a = store.resolve_target(target_a)
    row_b = store.resolve_target(target_b)

    if row_a is None or row_b is None:
        console.print("Could not resolve both targets.")
        return
    if isinstance(row_a, list) or isinstance(row_b, list):
        console.print("Ambiguous target — use exact run ID (e.g. s2r3).")
        return

    rec_a = loader.load_run_record(row_a)
    rec_b = loader.load_run_record(row_b)

    id_a = f"s{row_a['session_id']}r{row_a['run_seq']}"
    id_b = f"s{row_b['session_id']}r{row_b['run_seq']}"

    terminal_renderer.render_diff(rec_a, rec_b, id_a, id_b)


@main.command()
@click.argument("target")
def budget(target):
    """Token waterfall only."""
    run_row, record = _resolve_and_load(target)
    if record is None:
        return
    terminal_renderer.render_budget(record)


@main.group()
def session():
    """Session management commands."""


@session.command()
@click.argument("session_id")
@click.argument("title")
def rename(session_id, title):
    """Rename a session."""
    sid = _parse_session_id(session_id)
    store.rename_session(sid, title)
    console.print(f'Session {sid} renamed to "{title}".')
