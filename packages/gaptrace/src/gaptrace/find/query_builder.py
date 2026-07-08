def build_search_query(
    hint: str | None = None,
    exact: bool = False,
    session_id: int | None = None,
    pipeline: str | None = None,
    from_dt: str | None = None,
    to_dt: str | None = None,
    recent_n: int | None = None,
    fts5_available: bool = False,
) -> tuple[str, list]:
    base = (
        "SELECT r.session_id, r.run_seq, r.query, r.pipeline, "
        "r.created_at, s.title as session_title "
        "FROM runs r JOIN sessions s ON r.session_id = s.session_id"
    )

    clauses = []
    params: list = []

    if hint is not None:
        if fts5_available:
            if exact:
                fts_query = f'"{hint}"'
            else:
                tokens = hint.split()
                fts_query = " OR ".join(f'"{t}"' for t in tokens) if tokens else None
            if fts_query:
                clauses.append("r.rowid IN (SELECT rowid FROM runs_fts WHERE runs_fts MATCH ?)")
                params.append(fts_query)
        elif exact:
            clauses.append("r.query LIKE ?")
            params.append(f"%{hint}%")
        else:
            tokens = hint.split()
            if tokens:
                token_clauses = ["r.query LIKE ?" for _ in tokens]
                params.extend(f"%{t}%" for t in tokens)
                clauses.append(f"({' OR '.join(token_clauses)})")

    if session_id is not None:
        clauses.append("r.session_id = ?")
        params.append(session_id)

    if pipeline is not None:
        clauses.append("r.pipeline = ?")
        params.append(pipeline)

    if from_dt is not None:
        clauses.append("r.created_at >= ?")
        params.append(from_dt)

    if to_dt is not None:
        clauses.append("r.created_at <= ?")
        params.append(to_dt)

    sql = base
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY r.created_at DESC"

    if recent_n is not None:
        sql += " LIMIT ?"
        params.append(recent_n)

    return sql, params
