from ragradar import store


class TestListSessions:
    def test_recency_order(self, populated_db):
        sessions = store.list_sessions()
        assert len(sessions) == 2
        assert sessions[0]["session_id"] == 2
        assert sessions[1]["session_id"] == 1

    def test_includes_run_count(self, populated_db):
        sessions = store.list_sessions()
        s2 = sessions[0]
        assert s2["run_count"] == 2

    def test_pipeline_filter(self, populated_db):
        sessions = store.list_sessions(pipeline="pipe_a")
        assert len(sessions) == 2
        sessions = store.list_sessions(pipeline="nonexistent")
        assert len(sessions) == 0

    def test_empty_db(self):
        sessions = store.list_sessions()
        assert sessions == []


class TestListRuns:
    def test_returns_correct_runs(self, populated_db):
        runs = store.list_runs(2)
        assert len(runs) == 2

    def test_recency_order(self, populated_db):
        runs = store.list_runs(2)
        assert runs[0]["run_seq"] == 2
        assert runs[1]["run_seq"] == 1

    def test_empty_session(self, populated_db):
        runs = store.list_runs(999)
        assert runs == []


class TestSearchRuns:
    def test_token_match(self, populated_db):
        results = store.search_runs(hint="score")
        assert len(results) >= 1
        for r in results:
            assert "score" in r["query"].lower()

    def test_exact_match(self, populated_db):
        results = store.search_runs(hint="score scale differences", exact=True)
        assert len(results) == 1
        assert "score scale differences" in results[0]["query"]

    def test_exact_no_match(self, populated_db):
        results = store.search_runs(hint="score fusion zebra", exact=True)
        assert len(results) == 0

    def test_multi_token_or(self, populated_db):
        results = store.search_runs(hint="BM25 RRF")
        assert len(results) == 3

    def test_session_filter(self, populated_db):
        results = store.search_runs(hint="score", session_id=2)
        for r in results:
            assert r["session_id"] == 2

    def test_pipeline_filter(self, populated_db):
        results = store.search_runs(pipeline="pipe_a")
        assert len(results) == 3
        results = store.search_runs(pipeline="nonexistent")
        assert len(results) == 0

    def test_date_filter(self, populated_db):
        results = store.search_runs(from_dt="2026-06-09")
        assert all(r["created_at"] >= "2026-06-09" for r in results)
        results = store.search_runs(to_dt="2026-06-08T23:59:59Z")
        assert all(r["created_at"] <= "2026-06-08T23:59:59Z" for r in results)

    def test_recent_n(self, populated_db):
        results = store.search_runs(recent_n=1)
        assert len(results) == 1

    def test_includes_session_title(self, populated_db):
        results = store.search_runs(hint="RRF")
        rrf = [r for r in results if r["session_id"] == 2]
        assert rrf[0]["session_title"] == "RRF investigation"


class TestGetLatestRun:
    def test_returns_most_recent(self, populated_db):
        run = store.get_latest_run()
        assert run is not None
        assert run["session_id"] == 2
        assert run["run_seq"] == 2

    def test_no_db(self):
        run = store.get_latest_run()
        assert run is None


class TestResolveTarget:
    def test_none_returns_latest(self, populated_db):
        run = store.resolve_target(None)
        assert run is not None
        assert run["session_id"] == 2
        assert run["run_seq"] == 2

    def test_exact_id(self, populated_db):
        run = store.resolve_target("s2r1")
        assert run is not None
        assert run["session_id"] == 2
        assert run["run_seq"] == 1

    def test_case_insensitive(self, populated_db):
        run = store.resolve_target("S2R1")
        assert run is not None
        assert run["session_id"] == 2

    def test_nonexistent_id(self, populated_db):
        run = store.resolve_target("s99r99")
        assert run is None

    def test_text_hint_single_match(self, populated_db):
        run = store.resolve_target("handle")
        assert isinstance(run, dict)
        assert run["query"] == "does RRF handle score scale differences"

    def test_text_hint_multiple_matches(self, populated_db):
        result = store.resolve_target("score")
        assert isinstance(result, list)
        assert len(result) >= 2


class TestRenameSession:
    def test_renames(self, populated_db):
        store.rename_session(1, "New Title")
        sessions = store.list_sessions()
        s1 = [s for s in sessions if s["session_id"] == 1][0]
        assert s1["title"] == "New Title"
