"""Read-side store queries for the gaptrace analyst CLI.

Connection handling, schema, and the shared run getters live in
gaptrace_core.store — this module only adds the CLI-specific queries
(session listing, search, target resolution, session rename).
"""

from gaptrace_core.store import (  # noqa: F401 (get_run/get_latest_run re-exported for callers)
    connect,
    get_latest_run,
    get_run,
)
from gaptrace_core.targets import parse_target_id


def list_sessions(pipeline: str | None = None) -> list[dict]:
    """List sessions (with per-session run counts), newest first.

    Read-only query (though connecting may create/migrate the store).
    """
    conn = connect()
    try:
        sql = (
            "SELECT s.session_id, s.title, s.pipeline, s.created_at, "
            "COUNT(r.run_seq) as run_count "
            "FROM sessions s LEFT JOIN runs r ON s.session_id = r.session_id "
        )
        params: list = []
        if pipeline is not None:
            sql += "WHERE s.pipeline = ? "
            params.append(pipeline)
        sql += "GROUP BY s.session_id ORDER BY s.created_at DESC"
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def list_runs(session_id: int) -> list[dict]:
    """List all runs in a session, newest first (empty list if none).

    Read-only query (though connecting may create/migrate the store).
    """
    conn = connect()
    try:
        rows = conn.execute(
            "SELECT session_id, run_seq, query, pipeline, created_at "
            "FROM runs WHERE session_id = ? ORDER BY created_at DESC",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def search_runs(
    hint: str | None = None,
    exact: bool = False,
    session_id: int | None = None,
    pipeline: str | None = None,
    from_dt: str | None = None,
    to_dt: str | None = None,
    recent_n: int | None = None,
) -> list[dict]:
    """Search runs by query text and/or filters, newest first.

    Read-only query (though connecting may create/migrate the store).
    FTS5 is always available: gaptrace_core guarantees the schema is at the
    latest version, which ships the runs_fts index.
    """
    from gaptrace.find.query_builder import build_search_query

    conn = connect()
    try:
        sql, params = build_search_query(
            hint=hint,
            exact=exact,
            session_id=session_id,
            pipeline=pipeline,
            from_dt=from_dt,
            to_dt=to_dt,
            recent_n=recent_n,
            fts5_available=True,
        )
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def resolve_target(target: str | None = None) -> dict | list[dict] | None:
    """Resolve a target string to a run row.

    Read-only (though connecting may create/migrate the store).
    None → latest run; sNrN → exact lookup; anything else → text search
    (single match → that run, multiple → ranked list for the caller to
    disambiguate, none → None).
    """
    if target is None:
        return get_latest_run()

    try:
        parsed = parse_target_id(target)
    except ValueError:
        parsed = None
    if parsed:
        return get_run(*parsed)

    results = search_runs(hint=target)
    if len(results) == 1:
        return get_run(results[0]["session_id"], results[0]["run_seq"])
    if len(results) > 1:
        from gaptrace.find.bm25 import score

        results.sort(key=lambda r: score(target, r["query"]), reverse=True)
        return results
    return None


def rename_session(session_id: int, title: str) -> None:
    """Set a session's title. Writes to store (no-op if session missing)."""
    conn = connect()
    try:
        conn.execute(
            "UPDATE sessions SET title = ? WHERE session_id = ?",
            (title, session_id),
        )
        conn.commit()
    finally:
        conn.close()
