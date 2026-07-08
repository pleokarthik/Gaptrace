"""
Targeted tests for the three FTS5 sync triggers (runs_fts_ins / _upd / _del).
Each test uses a migrated v3 database via the shared migrated_db fixture,
then manipulates the `runs` table directly with sqlite3 and asserts the
FTS5 index reflects the change.
"""

import sqlite3


def _fts_rowids(conn: sqlite3.Connection, term: str) -> set[int]:
    """Return the set of rowids the FTS5 index returns for a single term."""
    rows = conn.execute("SELECT rowid FROM runs_fts WHERE runs_fts MATCH ?", (term,)).fetchall()
    return {r[0] for r in rows}


def _insert_run(conn: sqlite3.Connection, session_id: int, run_seq: int, query: str):
    conn.execute(
        "INSERT INTO runs (session_id, run_seq, query, pipeline, created_at, run_data) "
        "VALUES (?, ?, ?, 'test', '2026-01-01T00:00:00+00:00', '{}')",
        (session_id, run_seq, query),
    )
    conn.commit()


class TestFTS5InsertTrigger:
    """runs_fts_ins: AFTER INSERT ON runs → INSERT into FTS5 index."""

    def test_new_row_is_searchable_immediately(self, migrated_db):
        """
        Trigger definition:
            AFTER INSERT ON runs BEGIN
                INSERT INTO runs_fts(rowid, query) VALUES (new.rowid, new.query);
            END

        Insert a run with a unique word not present in any existing row.
        The FTS5 index must return that row's rowid before any explicit rebuild.
        """
        with sqlite3.connect(str(migrated_db)) as conn:
            before = _fts_rowids(conn, "zephyrtestword")
            assert len(before) == 0, "precondition: term must be absent before insert"

            _insert_run(conn, 1, 99, "query about zephyrtestword performance")

            new_rowid = conn.execute(
                "SELECT rowid FROM runs WHERE session_id=1 AND run_seq=99"
            ).fetchone()[0]

            after = _fts_rowids(conn, "zephyrtestword")

        assert new_rowid in after, (
            f"INSERT trigger failed: rowid {new_rowid} not in FTS5 after insert "
            f"(FTS5 rowids: {after})"
        )

    def test_insert_trigger_does_not_affect_other_terms(self, migrated_db):
        """Inserting a row with term X must not pollute results for unrelated term Y."""
        with sqlite3.connect(str(migrated_db)) as conn:
            before_y = _fts_rowids(conn, "xanthusunrelatedword")
            _insert_run(conn, 1, 98, "query about alphatestinsert only")
            after_y = _fts_rowids(conn, "xanthusunrelatedword")

        assert before_y == after_y, "INSERT added hits for an unrelated term"


class TestFTS5UpdateTrigger:
    """runs_fts_upd: AFTER UPDATE OF query ON runs → delete old, insert new."""

    def test_old_text_purged_new_text_indexed(self, migrated_db):
        """
        Trigger definition:
            AFTER UPDATE OF query ON runs BEGIN
                INSERT INTO runs_fts(runs_fts, rowid, query)
                    VALUES ('delete', old.rowid, old.query);
                INSERT INTO runs_fts(rowid, query) VALUES (new.rowid, new.query);
            END

        Insert with query="alpha beta", confirm "alpha" is found.
        Update query to "gamma delta", confirm:
          - "alpha" returns zero hits for that rowid (old text purged)
          - "gamma" returns that rowid (new text indexed)
        """
        with sqlite3.connect(str(migrated_db)) as conn:
            _insert_run(conn, 1, 50, "alpha beta unique phrase")
            target_rowid = conn.execute(
                "SELECT rowid FROM runs WHERE session_id=1 AND run_seq=50"
            ).fetchone()[0]

            # Before update: "alpha" must be findable
            before_alpha = _fts_rowids(conn, "alpha")
            assert target_rowid in before_alpha, (
                "precondition: 'alpha' not in FTS5 after INSERT — INSERT trigger may be broken"
            )

            # Perform the update
            conn.execute(
                "UPDATE runs SET query='gamma delta unique phrase' "
                "WHERE session_id=1 AND run_seq=50"
            )
            conn.commit()

            after_alpha = _fts_rowids(conn, "alpha")
            after_gamma = _fts_rowids(conn, "gamma")

        assert target_rowid not in after_alpha, (
            f"UPDATE trigger failed to purge old text: rowid {target_rowid} "
            f"still returned for 'alpha' after update"
        )
        assert target_rowid in after_gamma, (
            f"UPDATE trigger failed to index new text: rowid {target_rowid} "
            f"not returned for 'gamma' after update"
        )

    def test_update_other_column_does_not_change_fts(self, migrated_db):
        """
        Trigger is AFTER UPDATE OF query — updating a non-query column must
        not corrupt the FTS5 index.
        """
        with sqlite3.connect(str(migrated_db)) as conn:
            _insert_run(conn, 1, 51, "deltaunchanged query text")
            target_rowid = conn.execute(
                "SELECT rowid FROM runs WHERE session_id=1 AND run_seq=51"
            ).fetchone()[0]

            before = _fts_rowids(conn, "deltaunchanged")
            assert target_rowid in before, "precondition failed"

            # Update a non-query column
            conn.execute("UPDATE runs SET pipeline='new_pipe' WHERE session_id=1 AND run_seq=51")
            conn.commit()

            after = _fts_rowids(conn, "deltaunchanged")

        assert before == after, "Updating a non-query column changed the FTS5 index unexpectedly"


class TestFTS5DeleteTrigger:
    """runs_fts_del: AFTER DELETE ON runs → remove from FTS5 index."""

    def test_deleted_row_no_longer_searchable(self, migrated_db):
        """
        Trigger definition:
            AFTER DELETE ON runs BEGIN
                INSERT INTO runs_fts(runs_fts, rowid, query)
                    VALUES ('delete', old.rowid, old.query);
            END

        Insert with a unique term, delete the row, confirm FTS5 no longer
        returns that rowid.
        """
        with sqlite3.connect(str(migrated_db)) as conn:
            _insert_run(conn, 1, 60, "omicrondeletetest unique search word")
            target_rowid = conn.execute(
                "SELECT rowid FROM runs WHERE session_id=1 AND run_seq=60"
            ).fetchone()[0]

            before = _fts_rowids(conn, "omicrondeletetest")
            assert target_rowid in before, (
                "precondition: INSERT trigger must have indexed the row first"
            )

            conn.execute("DELETE FROM runs WHERE session_id=1 AND run_seq=60")
            conn.commit()

            after = _fts_rowids(conn, "omicrondeletetest")

        assert target_rowid not in after, (
            f"DELETE trigger failed: rowid {target_rowid} still in FTS5 after row deletion"
        )
        assert len(after) == 0, f"Expected empty result set, got {after}"

    def test_delete_does_not_remove_other_rows(self, migrated_db):
        """Deleting one row must not remove other rows from FTS5."""
        with sqlite3.connect(str(migrated_db)) as conn:
            _insert_run(conn, 1, 61, "sigmaterm shared content")
            _insert_run(conn, 1, 62, "sigmaterm also here content")

            rowid_61 = conn.execute(
                "SELECT rowid FROM runs WHERE session_id=1 AND run_seq=61"
            ).fetchone()[0]
            rowid_62 = conn.execute(
                "SELECT rowid FROM runs WHERE session_id=1 AND run_seq=62"
            ).fetchone()[0]

            before = _fts_rowids(conn, "sigmaterm")
            assert {rowid_61, rowid_62}.issubset(before), "precondition: both rows indexed"

            conn.execute("DELETE FROM runs WHERE session_id=1 AND run_seq=61")
            conn.commit()

            after = _fts_rowids(conn, "sigmaterm")

        assert rowid_61 not in after, "deleted row still in FTS5"
        assert rowid_62 in after, "non-deleted row incorrectly removed from FTS5"
