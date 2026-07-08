"""Single source of truth for the gaptrace SQLite store.

Owns the store location (``~/.gaptrace/runs.db``), the schema (always created
at the LATEST version), the migration chain for databases created by
older versions, and every run/eval/benchmark/policy persistence
primitive. All other packages (gaptrace_capture, gaptrace, gaptrace_evaluate) import
their store access from here — none of them define their own connection
helper, schema, or version constant.

Environment-setup contract: :func:`connect` guarantees that the ``~/.gaptrace``
directory exists, the database file exists, and its schema is at
``SCHEMA_VERSION`` — creating fresh databases directly at the latest
version and migrating old ones in place. Any entry point (library call,
CLI, example script) therefore works on a fresh machine with no prior
CLI invocation.
"""

import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

from gaptrace_core.schema import RunRecord

SCHEMA_VERSION = "3"

# How long to keep retrying the one-time WAL-mode switch on a brand-new
# database file (see _set_wal_mode below) before giving up.
_WAL_SWITCH_RETRY_SECONDS = 5.0

# Latest schema, created as-is for fresh databases. Databases written by
# older package versions carry meta.schema_version "1" or "2" and are
# walked to "3" by _ensure_schema()'s migration chain.
#
# One statement per tuple entry (not a single multi-statement string run
# via executescript()): executescript() implicitly commits any pending
# transaction before it runs, which would silently release the
# BEGIN IMMEDIATE lock _ensure_schema() holds while bootstrapping a
# fresh database — reopening the exact concurrent-bootstrap race this
# structure exists to close. conn.execute() on one statement at a time
# respects the ambient transaction instead.
SCHEMA_STATEMENTS: tuple[str, ...] = (
    """CREATE TABLE IF NOT EXISTS meta (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS sessions (
        session_id INTEGER PRIMARY KEY AUTOINCREMENT,
        title      TEXT,
        pipeline   TEXT,
        created_at TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS runs (
        session_id   INTEGER NOT NULL REFERENCES sessions(session_id),
        run_seq      INTEGER NOT NULL,
        query        TEXT NOT NULL,
        pipeline     TEXT,
        created_at   TEXT NOT NULL,
        run_data     TEXT NOT NULL,
        eval_scores  TEXT,
        risk_score   REAL,
        evaluated_at TEXT,
        PRIMARY KEY (session_id, run_seq)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_runs_pipeline   ON runs(pipeline)",
    """CREATE TABLE IF NOT EXISTS benchmark (
        pipeline      TEXT NOT NULL,
        factor        TEXT NOT NULL,
        threshold     REAL,
        correlation   REAL,
        sample_count  INTEGER NOT NULL DEFAULT 0,
        updated_at    TEXT NOT NULL,
        PRIMARY KEY (pipeline, factor)
    )""",
    """CREATE TABLE IF NOT EXISTS policies (
        pipeline     TEXT PRIMARY KEY,
        policy_data  TEXT NOT NULL,
        updated_at   TEXT NOT NULL
    )""",
    """CREATE VIRTUAL TABLE IF NOT EXISTS runs_fts
    USING fts5(
        query,
        content=runs,
        content_rowid=rowid,
        tokenize='unicode61 remove_diacritics 1'
    )""",
    """CREATE TRIGGER IF NOT EXISTS runs_fts_ins
    AFTER INSERT ON runs BEGIN
        INSERT INTO runs_fts(rowid, query) VALUES (new.rowid, new.query);
    END""",
    """CREATE TRIGGER IF NOT EXISTS runs_fts_del
    AFTER DELETE ON runs BEGIN
        INSERT INTO runs_fts(runs_fts, rowid, query)
        VALUES ('delete', old.rowid, old.query);
    END""",
    """CREATE TRIGGER IF NOT EXISTS runs_fts_upd
    AFTER UPDATE OF query ON runs BEGIN
        INSERT INTO runs_fts(runs_fts, rowid, query)
        VALUES ('delete', old.rowid, old.query);
        INSERT INTO runs_fts(rowid, query) VALUES (new.rowid, new.query);
    END""",
)


def _gaptrace_dir() -> Path:
    """Return the gaptrace home directory (``~/.gaptrace``). Pure — does not create it.

    Tests monkeypatch this one function to isolate the store.
    """
    return Path.home() / ".gaptrace"


def db_path() -> Path:
    """Return the store's database path. Pure — does not create it."""
    return _gaptrace_dir() / "runs.db"


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(row[1] == column for row in rows)


def _set_wal_mode(conn: sqlite3.Connection) -> None:
    """Switch ``conn``'s database to WAL journal mode, retrying briefly
    under concurrent first-time connects. Writes to store (once ever).

    Changing journal mode requires exclusive access to the database file
    and, unlike ordinary lock contention on a normal statement, does not
    reliably back off via sqlite3's own ``timeout``/busy-handler retry —
    confirmed empirically: many threads opening a brand-new (non-WAL)
    file at once and each issuing this PRAGMA can raise
    ``sqlite3.OperationalError: database is locked`` even with a 5s
    connection timeout. Once the file is already in WAL mode (the
    common case after the very first connect ever), re-issuing this
    PRAGMA is a fast no-op read that never contends, so the retry loop
    below only ever matters for that one-time switch.
    """
    deadline = time.monotonic() + _WAL_SWITCH_RETRY_SECONDS
    while True:
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            return
        except sqlite3.OperationalError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Bring ``conn``'s database to SCHEMA_VERSION.

    Fresh (or meta-less) databases get the full latest schema in one
    shot; version "1"/"2" databases are migrated in place with data
    intact; anything else raises RuntimeError. Writes to store.

    Runs inside one ``BEGIN IMMEDIATE`` transaction — the same pattern
    :func:`commit_run` uses for run inserts — so concurrent first-time
    ``connect()`` calls against a brand-new database can't interleave
    "check whether meta exists" with "create it and stamp
    schema_version". Without this, one connection could observe the
    ``meta`` table (created by another connection's in-flight bootstrap,
    since plain DDL auto-commits per statement outside an explicit
    transaction) before that connection had committed the
    ``schema_version`` row, and raise a bogus "Unsupported schema
    version: None" error — a real, previously-uncovered race distinct
    from the run_seq race :func:`commit_run` fixes.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        has_meta = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='meta'"
        ).fetchone()
        if has_meta is None:
            for stmt in SCHEMA_STATEMENTS:
                conn.execute(stmt)
            conn.execute(
                "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)",
                (SCHEMA_VERSION,),
            )
            conn.commit()
            return

        row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        version = row[0] if row else None

        if version == SCHEMA_VERSION:
            conn.commit()
            return
        if version not in ("1", "2"):
            raise RuntimeError(
                f"Unsupported schema version: {version!r}. "
                f"Expected '1', '2', or '{SCHEMA_VERSION}'. Cannot migrate."
            )

        if version == "1":
            for col, col_type in [
                ("eval_scores", "TEXT"),
                ("risk_score", "REAL"),
                ("evaluated_at", "TEXT"),
            ]:
                if not _column_exists(conn, "runs", col):
                    conn.execute(f"ALTER TABLE runs ADD COLUMN {col} {col_type}")

            conn.execute(
                """CREATE TABLE IF NOT EXISTS benchmark (
                    pipeline      TEXT NOT NULL,
                    factor        TEXT NOT NULL,
                    threshold     REAL,
                    correlation   REAL,
                    sample_count  INTEGER NOT NULL DEFAULT 0,
                    updated_at    TEXT NOT NULL,
                    PRIMARY KEY (pipeline, factor)
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS policies (
                    pipeline     TEXT PRIMARY KEY,
                    policy_data  TEXT NOT NULL,
                    updated_at   TEXT NOT NULL
                )"""
            )
            conn.execute("UPDATE meta SET value = '2' WHERE key = 'schema_version'")
            version = "2"

        if version == "2":
            conn.execute(
                """CREATE VIRTUAL TABLE IF NOT EXISTS runs_fts
                USING fts5(
                    query,
                    content=runs,
                    content_rowid=rowid,
                    tokenize='unicode61 remove_diacritics 1'
                )"""
            )
            conn.execute("INSERT INTO runs_fts(runs_fts) VALUES('rebuild')")
            conn.execute(
                """CREATE TRIGGER IF NOT EXISTS runs_fts_ins
                AFTER INSERT ON runs BEGIN
                    INSERT INTO runs_fts(rowid, query) VALUES (new.rowid, new.query);
                END"""
            )
            conn.execute(
                """CREATE TRIGGER IF NOT EXISTS runs_fts_del
                AFTER DELETE ON runs BEGIN
                    INSERT INTO runs_fts(runs_fts, rowid, query)
                    VALUES ('delete', old.rowid, old.query);
                END"""
            )
            conn.execute(
                """CREATE TRIGGER IF NOT EXISTS runs_fts_upd
                AFTER UPDATE OF query ON runs BEGIN
                    INSERT INTO runs_fts(runs_fts, rowid, query)
                    VALUES ('delete', old.rowid, old.query);
                    INSERT INTO runs_fts(rowid, query) VALUES (new.rowid, new.query);
                END"""
            )
            conn.execute("DROP INDEX IF EXISTS idx_runs_query")
            conn.execute("UPDATE meta SET value = '3' WHERE key = 'schema_version'")

        conn.commit()
    except BaseException:
        conn.rollback()
        raise


def connect() -> sqlite3.Connection:
    """Open a Row-factory connection to the store, setting up the environment.

    Side effects (writes to store): creates ``~/.gaptrace`` and ``runs.db`` if
    missing, creates the schema at SCHEMA_VERSION for fresh databases,
    and migrates version "1"/"2" databases in place.

    Returns an open ``sqlite3.Connection`` with ``sqlite3.Row`` row
    factory — the caller must close it. Raises RuntimeError for a
    database whose schema version is unsupported (newer than this
    package understands).
    """
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        # PRAGMA journal_mode can't be changed from inside a transaction,
        # so this runs before _ensure_schema()'s BEGIN IMMEDIATE.
        _set_wal_mode(conn)
        _ensure_schema(conn)
    except BaseException:
        conn.close()
        raise
    return conn


def ensure_store() -> Path:
    """Create/migrate the store without keeping a connection open.

    Writes to store (via :func:`connect`). Returns the database path.
    """
    connect().close()
    return db_path()


# ---------------------------------------------------------------------------
# Session / run persistence
# ---------------------------------------------------------------------------


def _get_or_create_session_on(
    conn: sqlite3.Connection, pipeline: str | None, idle_gap_minutes: int = 30
) -> int:
    """Same contract as :func:`get_or_create_session`, on a caller-owned
    connection/transaction instead of opening its own. Writes to store."""
    if pipeline is not None:
        row = conn.execute(
            "SELECT session_id, created_at FROM sessions "
            "WHERE pipeline = ? ORDER BY created_at DESC LIMIT 1",
            (pipeline,),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT session_id, created_at FROM sessions "
            "WHERE pipeline IS NULL ORDER BY created_at DESC LIMIT 1",
        ).fetchone()

    if row is not None:
        session_id, session_created = row
        last_run = conn.execute(
            "SELECT created_at FROM runs WHERE session_id = ? ORDER BY created_at DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        last_time = datetime.fromisoformat(last_run[0] if last_run else session_created)
        if last_time.tzinfo is None:
            last_time = last_time.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        if (now - last_time).total_seconds() < idle_gap_minutes * 60:
            return session_id

    now_iso = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        "INSERT INTO sessions (pipeline, created_at) VALUES (?, ?)",
        (pipeline, now_iso),
    )
    return cursor.lastrowid


def get_or_create_session(pipeline: str | None, idle_gap_minutes: int = 30) -> int:
    """Return the current session id for ``pipeline``, creating one if needed.

    Writes to store: reuses the most recent session for ``pipeline`` when
    its last activity is within ``idle_gap_minutes``, otherwise inserts a
    new session row. Returns the session id.

    Opens and commits its own transaction, so calling this back-to-back
    with :func:`next_run_seq`/:func:`write_run` as three separate calls is
    not race-free under concurrent writers — see :func:`commit_run` for
    the atomic path used by ``Capture.commit()``.
    """
    conn = connect()
    try:
        session_id = _get_or_create_session_on(conn, pipeline, idle_gap_minutes)
        conn.commit()
        return session_id
    finally:
        conn.close()


def _next_run_seq_on(conn: sqlite3.Connection, session_id: int) -> int:
    """Same contract as :func:`next_run_seq`, on a caller-owned connection."""
    row = conn.execute(
        "SELECT MAX(run_seq) FROM runs WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    return (row[0] or 0) + 1


def next_run_seq(session_id: int) -> int:
    """Return the next run_seq for ``session_id`` (1 for an empty session).

    Read-only query (though connecting may create/migrate the store).
    Calling this and then :func:`write_run` as two separate calls has a
    TOCTOU race under concurrent writers to the same session — see
    :func:`commit_run` for the atomic path.
    """
    conn = connect()
    try:
        return _next_run_seq_on(conn, session_id)
    finally:
        conn.close()


def _write_run_on(
    conn: sqlite3.Connection,
    session_id: int,
    run_seq: int,
    record: RunRecord,
    pipeline: str | None,
) -> None:
    """Same contract as :func:`write_run`, on a caller-owned connection."""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO runs (session_id, run_seq, query, pipeline, created_at, run_data) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            session_id,
            run_seq,
            record.query,
            pipeline,
            now,
            json.dumps(record.to_json()),
        ),
    )


def write_run(session_id: int, run_seq: int, record: RunRecord, pipeline: str | None) -> None:
    """Insert one run row for ``record``. Writes to store.

    ``created_at`` is stamped with the current UTC time; ``run_data`` is
    the JSON-serialized record. Raises sqlite3.IntegrityError if
    (session_id, run_seq) already exists.
    """
    conn = connect()
    try:
        _write_run_on(conn, session_id, run_seq, record, pipeline)
        conn.commit()
    finally:
        conn.close()


def commit_run(
    pipeline: str | None, record: RunRecord, idle_gap_minutes: int = 30
) -> tuple[int, int]:
    """Atomically resolve/create a session, assign the next run_seq, and
    insert the run row. Writes to store.

    This is the race-free replacement for calling
    ``get_or_create_session()`` + ``next_run_seq()`` + ``write_run()`` as
    three separate connections: session resolution, run_seq assignment,
    and the insert all happen inside one ``BEGIN IMMEDIATE`` transaction,
    so no other writer can interleave a run insert for the same session
    between "compute run_seq" and "insert the row" — the exact race that
    used to raise ``sqlite3.IntegrityError`` on ``(session_id, run_seq)``
    under concurrent commits and get silently swallowed by the capture
    layer's fail-open contract.

    ``BEGIN IMMEDIATE`` acquires SQLite's write lock up front rather than
    on first write, so a concurrent caller blocks (and retries under the
    connection's default busy timeout) instead of racing to the insert.
    If a collision is nonetheless detected (belt-and-suspenders — this
    should be unreachable given the transaction above), this raises
    ``RuntimeError`` rather than silently retrying or swallowing it, since
    silent loss of a colliding write is the bug this function exists to
    eliminate.

    Returns ``(session_id, run_seq)``. Raises ``sqlite3.OperationalError``
    if the write lock can't be acquired before the connection's busy
    timeout elapses, and re-raises (after rolling back) any other error —
    callers (``Capture.commit()``) apply their own fail-open/strict-mode
    handling on top of this.
    """
    conn = connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        session_id = _get_or_create_session_on(conn, pipeline, idle_gap_minutes)
        run_seq = _next_run_seq_on(conn, session_id)
        try:
            _write_run_on(conn, session_id, run_seq, record, pipeline)
        except sqlite3.IntegrityError as e:
            raise RuntimeError(
                f"run_seq collision on session {session_id} seq {run_seq} inside an "
                "atomic transaction — this should be unreachable; investigate rather "
                "than retry."
            ) from e
        conn.commit()
        return session_id, run_seq
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def write_runs_batch(
    session_id: int,
    start_seq: int,
    records: list[RunRecord],
    pipeline: str | None,
) -> None:
    """Insert many run rows in one transaction. Writes to store.

    Records get consecutive run_seq values starting at ``start_seq`` and
    share one UTC ``created_at`` stamp.
    """
    now = datetime.now(timezone.utc).isoformat()
    rows = [
        (session_id, start_seq + i, record.query, pipeline, now, json.dumps(record.to_json()))
        for i, record in enumerate(records)
    ]
    conn = connect()
    try:
        conn.executemany(
            "INSERT INTO runs (session_id, run_seq, query, pipeline, created_at, run_data) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
    finally:
        conn.close()


_RUN_COLUMNS = (
    "session_id, run_seq, query, pipeline, created_at, "
    "run_data, eval_scores, risk_score, evaluated_at"
)


def get_run(session_id: int, run_seq: int) -> dict | None:
    """Fetch one run row as a dict, or None if it doesn't exist.

    Read-only query (though connecting may create/migrate the store).
    """
    conn = connect()
    try:
        row = conn.execute(
            f"SELECT {_RUN_COLUMNS} FROM runs WHERE session_id = ? AND run_seq = ?",
            (session_id, run_seq),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_latest_run() -> dict | None:
    """Fetch the most recently created run row, or None if the store is empty.

    Read-only query (though connecting may create/migrate the store).
    """
    conn = connect()
    try:
        row = conn.execute(
            f"SELECT {_RUN_COLUMNS} FROM runs ORDER BY created_at DESC LIMIT 1",
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_runs_in_session(session_id: int) -> list[dict]:
    """Fetch all run rows in a session, newest first (empty list if none).

    Read-only query (though connecting may create/migrate the store).
    """
    conn = connect()
    try:
        rows = conn.execute(
            f"SELECT {_RUN_COLUMNS} FROM runs WHERE session_id = ? ORDER BY created_at DESC",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Eval-score persistence (eval_scores lives on the runs table)
# ---------------------------------------------------------------------------


def write_eval_scores(
    session_id: int,
    run_seq: int,
    eval_scores: dict,
    risk_score: float,
) -> None:
    """Persist eval scores + risk on one run row. Writes to store.

    Stamps ``evaluated_at`` with the current UTC time. Silently updates
    zero rows if the run doesn't exist.
    """
    conn = connect()
    try:
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE runs SET eval_scores = ?, risk_score = ?, evaluated_at = ? "
            "WHERE session_id = ? AND run_seq = ?",
            (json.dumps(eval_scores), risk_score, now, session_id, run_seq),
        )
        conn.commit()
    finally:
        conn.close()


def write_eval_scores_batch(entries: list[tuple]) -> None:
    """Persist eval scores for many runs in one transaction. Writes to store.

    Each entry is ``(session_id, run_seq, eval_scores_dict, risk_score)``.
    No-op for an empty list.
    """
    if not entries:
        return
    conn = connect()
    try:
        now = datetime.now(timezone.utc).isoformat()
        rows = [
            (json.dumps(eval_scores), risk_score, now, session_id, run_seq)
            for session_id, run_seq, eval_scores, risk_score in entries
        ]
        conn.executemany(
            "UPDATE runs SET eval_scores = ?, risk_score = ?, evaluated_at = ? "
            "WHERE session_id = ? AND run_seq = ?",
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def get_eval_scores(session_id: int, run_seq: int) -> dict | None:
    """Fetch a run's stored eval scores (with ``risk_score`` merged in).

    Read-only query (though connecting may create/migrate the store).
    Returns None if the run doesn't exist or was never evaluated.
    """
    conn = connect()
    try:
        row = conn.execute(
            "SELECT eval_scores, risk_score FROM runs WHERE session_id = ? AND run_seq = ?",
            (session_id, run_seq),
        ).fetchone()
        if row is None or row["eval_scores"] is None:
            return None
        result = json.loads(row["eval_scores"])
        result["risk_score"] = row["risk_score"]
        return result
    finally:
        conn.close()


def get_all_evaluated_runs(pipeline: str | None = None) -> list[dict]:
    """Fetch every run row with non-null eval_scores, newest first.

    Read-only query (though connecting may create/migrate the store).
    ``pipeline=None`` means all pipelines, not the "__default" key.
    """
    conn = connect()
    try:
        sql = f"SELECT {_RUN_COLUMNS} FROM runs WHERE eval_scores IS NOT NULL"
        params: list = []
        if pipeline is not None:
            sql += " AND pipeline = ?"
            params.append(pipeline)
        sql += " ORDER BY created_at DESC"
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Benchmark persistence
# ---------------------------------------------------------------------------


def write_benchmark_entry(
    pipeline: str,
    factor: str,
    threshold: float | None,
    correlation: float | None,
    sample_count: int,
) -> None:
    """Upsert one benchmark row keyed (pipeline, factor). Writes to store."""
    conn = connect()
    try:
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT OR REPLACE INTO benchmark "
            "(pipeline, factor, threshold, correlation, sample_count, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (pipeline, factor, threshold, correlation, sample_count, now),
        )
        conn.commit()
    finally:
        conn.close()


def write_benchmark_entries_batch(entries: list[tuple]) -> None:
    """Upsert many benchmark rows in one transaction. Writes to store.

    Each entry is ``(pipeline, factor, threshold, correlation, sample_count)``.
    No-op for an empty list.
    """
    if not entries:
        return
    conn = connect()
    try:
        now = datetime.now(timezone.utc).isoformat()
        rows = [(p, f, t, c, s, now) for p, f, t, c, s in entries]
        conn.executemany(
            "INSERT OR REPLACE INTO benchmark "
            "(pipeline, factor, threshold, correlation, sample_count, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def get_benchmark(pipeline: str) -> list[dict]:
    """Fetch all benchmark rows for ``pipeline`` (empty list if none).

    Read-only query (though connecting may create/migrate the store).
    """
    conn = connect()
    try:
        rows = conn.execute(
            "SELECT factor, threshold, correlation, sample_count, updated_at "
            "FROM benchmark WHERE pipeline = ?",
            (pipeline,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Policy persistence
# ---------------------------------------------------------------------------


def write_policy(pipeline: str, policy: dict) -> None:
    """Upsert the policy dict for ``pipeline``. Writes to store."""
    conn = connect()
    try:
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT OR REPLACE INTO policies (pipeline, policy_data, updated_at) VALUES (?, ?, ?)",
            (pipeline, json.dumps(policy), now),
        )
        conn.commit()
    finally:
        conn.close()


def get_policy(pipeline: str) -> dict | None:
    """Fetch the stored policy dict for ``pipeline``, or None if unset.

    Read-only query (though connecting may create/migrate the store).
    """
    conn = connect()
    try:
        row = conn.execute(
            "SELECT policy_data FROM policies WHERE pipeline = ?",
            (pipeline,),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["policy_data"])
    finally:
        conn.close()


def delete_policy(pipeline: str) -> None:
    """Delete the stored policy for ``pipeline`` (no-op if unset).

    Writes to store. Subsequent ``get_policy`` calls return None, which
    callers treat as "use defaults".
    """
    conn = connect()
    try:
        conn.execute("DELETE FROM policies WHERE pipeline = ?", (pipeline,))
        conn.commit()
    finally:
        conn.close()
