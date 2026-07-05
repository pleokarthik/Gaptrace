import json
import re
import sqlite3

import pytest
import ragradar_capture
from ragradar_capture.thread_local import clear_active_capture, get_active_capture
from ragradar_core import coerce, store
from ragradar_core.schema import RunRecord, TokenBudget

RUN_ID_RE = re.compile(r"^s\d+r\d+$")


@pytest.fixture(autouse=True)
def non_strict():
    """Every test starts and ends in the default never-raise mode."""
    ragradar_capture.set_strict(False)
    yield
    ragradar_capture.set_strict(False)


class TestCapture:
    def test_minimal_capture(self):
        ragradar_capture.capture("what is RRF?", "reciprocal rank fusion")
        with sqlite3.connect(str(store.db_path())) as conn:
            row = conn.execute("SELECT run_data FROM runs").fetchone()
        assert row is not None
        data = json.loads(row[0])
        assert data["query"] == "what is RRF?"
        assert data["response"] == "reciprocal rank fusion"

    def test_returns_valid_run_id_matching_db(self):
        run_id = ragradar_capture.capture("q", "r", pipeline="pipe")
        assert isinstance(run_id, str)
        assert RUN_ID_RE.match(run_id)

        with sqlite3.connect(str(store.db_path())) as conn:
            sid, seq = conn.execute("SELECT session_id, run_seq FROM runs").fetchone()
        assert run_id == f"s{sid}r{seq}"

    def test_capture_with_all_fields_round_trips(self):
        original = RunRecord(
            query="test query",
            response="test response",
            chunks=[
                {
                    "chunk_id": "c1",
                    "source_doc_id": "d1",
                    "content": "text",
                    "token_count": 10,
                }
            ],
            final_prompt="assembled prompt",
            token_budget={
                "total_limit": 4096,
                "chunks_allocated": 2000,
                "history_allocated": 500,
                "system_allocated": 800,
                "headroom": 796,
            },
            history_pre=[{"role": "user", "content": "hi", "tokens": 2}],
            history_post=[{"role": "user", "content": "hi", "tokens": 2}],
            eviction_reason="token_budget",
            cache_events=[{"chunk_id": "c1", "hit": True}],
            tool_calls=[{"tool_name": "search", "arguments": {"query": "RRF"}}],
            model="gpt-4",
            token_usage={
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            },
        )
        run_id = ragradar_capture.capture(
            "test query",
            "test response",
            pipeline="pipe",
            chunks=[
                {
                    "chunk_id": "c1",
                    "source_doc_id": "d1",
                    "content": "text",
                    "token_count": 10,
                }
            ],
            final_prompt="assembled prompt",
            token_budget={
                "total_limit": 4096,
                "chunks_allocated": 2000,
                "history_allocated": 500,
                "system_allocated": 800,
                "headroom": 796,
            },
            history_pre=[{"role": "user", "content": "hi", "tokens": 2}],
            history_post=[{"role": "user", "content": "hi", "tokens": 2}],
            eviction_reason="token_budget",
            cache_events=[{"chunk_id": "c1", "hit": True}],
            tool_calls=[{"tool_name": "search", "arguments": {"query": "RRF"}}],
            model="gpt-4",
            token_usage={
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            },
        )
        assert run_id is not None

        with sqlite3.connect(str(store.db_path())) as conn:
            row = conn.execute("SELECT run_data FROM runs").fetchone()
        stored = RunRecord.from_json(json.loads(row[0]))

        # The dict-kwarg shapes must coerce to the exact same record a
        # hand-built RunRecord serializes to. RunRecord.__init__ doesn't
        # coerce nested dicts itself, so compare via from_json on both.
        assert stored.to_json() == RunRecord.from_json(original.to_json()).to_json()

    def test_unknown_kwarg_raises_type_error(self):
        with pytest.raises(TypeError):
            ragradar_capture.capture("q", "r", chunk=[{"chunk_id": "c1"}])

    def test_token_budget_without_final_prompt_is_persisted(self):
        # Regression: the old **kwargs implementation only looked at
        # token_budget while handling final_prompt, silently dropping it
        # when final_prompt was absent.
        ragradar_capture.capture(
            "q",
            "r",
            token_budget={
                "total_limit": 4096,
                "chunks_allocated": 2000,
                "history_allocated": 500,
                "system_allocated": 800,
                "headroom": 796,
            },
        )
        with sqlite3.connect(str(store.db_path())) as conn:
            row = conn.execute("SELECT run_data FROM runs").fetchone()
        data = json.loads(row[0])
        assert data["final_prompt"] is None
        assert data["token_budget"]["headroom"] == 796


class TestStartAndCapture:
    def setup_method(self):
        clear_active_capture()

    def test_staged_flow_run_id_lifecycle(self):
        cap = ragradar_capture.start("test query", pipeline="test")
        assert cap.run_id is None

        cap.chunks(
            [
                {
                    "chunk_id": "c1",
                    "source_doc_id": "d1",
                    "content": "hello",
                    "token_count": 5,
                }
            ]
        )
        assert cap.run_id is None

        returned = cap.response("test response")
        assert returned is not None
        assert RUN_ID_RE.match(returned)
        assert cap.run_id == returned

        with sqlite3.connect(str(store.db_path())) as conn:
            sid, seq, run_data = conn.execute(
                "SELECT session_id, run_seq, run_data FROM runs"
            ).fetchone()
        assert returned == f"s{sid}r{seq}"
        data = json.loads(run_data)
        assert data["query"] == "test query"
        assert data["response"] == "test response"
        assert len(data["chunks"]) == 1

    def test_double_commit_same_id_writes_once(self):
        cap = ragradar_capture.start("q", pipeline="test")
        cap._record.response = "r"
        first = cap.commit()
        second = cap.commit()
        assert first is not None
        assert first == second == cap.run_id

        with sqlite3.connect(str(store.db_path())) as conn:
            count = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        assert count == 1

    def test_tool_call_appends_not_replaces(self):
        cap = ragradar_capture.start("test query", pipeline="test")
        cap.tool_call({"tool_name": "search", "arguments": {"query": "RRF"}})
        cap.tool_call({"tool_name": "fetch_url", "arguments": {"url": "x"}})
        cap.response("test response")

        with sqlite3.connect(str(store.db_path())) as conn:
            row = conn.execute("SELECT run_data FROM runs").fetchone()
        data = json.loads(row[0])
        assert len(data["tool_calls"]) == 2
        assert data["tool_calls"][0]["tool_name"] == "search"
        assert data["tool_calls"][1]["tool_name"] == "fetch_url"


class TestPrimitiveInputs:
    """The coercion boundary: naive callers pass dicts/tuples/ints and
    never need to know the schema dataclasses exist."""

    def setup_method(self):
        clear_active_capture()

    def _stored_record(self):
        with sqlite3.connect(str(store.db_path())) as conn:
            row = conn.execute("SELECT run_data FROM runs ORDER BY rowid DESC LIMIT 1").fetchone()
        return json.loads(row[0])

    def test_shorthand_turn_dicts_and_tuples(self):
        cap = ragradar_capture.start("q", pipeline="test")
        cap.history(
            pre=[{"user": "hello there"}, ("assistant", "hi!")],
            post=[{"user": "hello there"}],
            eviction_reason="token_budget",
        )
        cap.response("r")
        data = self._stored_record()
        assert data["history_pre"][0] == {
            "role": "user",
            "content": "hello there",
            "tokens": coerce.estimate_tokens("hello there"),
        }
        assert data["history_pre"][1]["role"] == "assistant"
        assert data["history_pre"][1]["content"] == "hi!"
        assert data["history_pre"][1]["tokens"] >= 1
        assert data["eviction_reason"] == "token_budget"

    def test_explicit_tokens_override_estimate(self):
        cap = ragradar_capture.start("q", pipeline="test")
        cap.history(
            pre=[{"user": "hello there", "tokens": 99}, {"role": "assistant", "content": "hi"}],
            post=[],
        )
        cap.response("r")
        data = self._stored_record()
        assert data["history_pre"][0]["tokens"] == 99
        # full role-dict without tokens also gets an estimate
        assert data["history_pre"][1]["tokens"] == coerce.estimate_tokens("hi")

    def test_minimal_chunk_dict_gets_defaults(self):
        cap = ragradar_capture.start("q", pipeline="test")
        cap.chunks(
            [
                {"content": "only content given", "rerank_score": 0.9},
                {"content": "second chunk"},
            ]
        )
        cap.response("r")
        chunks = self._stored_record()["chunks"]
        assert chunks[0]["chunk_id"] == "chunk_0"
        assert chunks[1]["chunk_id"] == "chunk_1"
        assert chunks[0]["source_doc_id"] == "unknown"
        assert chunks[0]["token_count"] == coerce.estimate_tokens("only content given")
        assert chunks[0]["rerank_score"] == 0.9

    def test_chunk_without_content_still_errors(self):
        ragradar_capture.set_strict(True)
        cap = ragradar_capture.start("q", pipeline="test")
        with pytest.raises(TypeError):
            cap.chunks([{"rerank_score": 0.9}])

    def test_int_token_budget_headroom_from_prompt(self):
        cap = ragradar_capture.start("q", pipeline="test")
        prompt = "p" * 400  # ~100 estimated tokens
        cap.context(prompt, 4096)
        cap.response("r")
        budget = self._stored_record()["token_budget"]
        assert budget["total_limit"] == 4096
        assert budget["headroom"] == 4096 - coerce.estimate_tokens(prompt)

    def test_partial_budget_dict_headroom_from_allocations(self):
        run_id = ragradar_capture.capture(
            "q",
            "r",
            token_budget={
                "total_limit": 4096,
                "chunks_allocated": 2800,
                "history_allocated": 600,
                "system_allocated": 500,
            },
        )
        assert run_id is not None
        budget = self._stored_record()["token_budget"]
        assert budget["headroom"] == 4096 - 2800 - 600 - 500

    def test_cache_mapping_and_tuples(self):
        cap = ragradar_capture.start("q", pipeline="test")
        cap.cache({"c1": True, "c2": False})
        cap.response("r")
        events = self._stored_record()["cache_events"]
        assert {(e["chunk_id"], e["hit"]) for e in events} == {("c1", True), ("c2", False)}

        cap = ragradar_capture.start("q2", pipeline="test")
        cap.cache([("c3", 1)])
        cap.response("r2")
        events = self._stored_record()["cache_events"]
        assert events == [{"chunk_id": "c3", "hit": True, "cache_source": None}]

    def test_semantic_cache_all_fields(self):
        cap = ragradar_capture.start("q", pipeline="test")
        cap.semantic_cache(
            checked=True,
            hit=True,
            similarity_score=0.94,
            threshold=0.9,
            cached_query="a near-duplicate question",
            cached_at="2026-07-01T00:00:00+00:00",
            registered=True,
        )
        cap.response("r")
        cache = self._stored_record()["cache"]
        assert cache == {
            "checked": True,
            "hit": True,
            "similarity_score": 0.94,
            "threshold": 0.9,
            "cached_query": "a near-duplicate question",
            "cached_at": "2026-07-01T00:00:00+00:00",
            "registered": True,
        }

    def test_semantic_cache_defaults_on_miss(self):
        cap = ragradar_capture.start("q", pipeline="test")
        cap.semantic_cache(checked=True)
        cap.response("r")
        cache = self._stored_record()["cache"]
        assert cache["checked"] is True
        assert cache["hit"] is False
        assert cache["similarity_score"] is None
        assert cache["registered"] is False

    def test_no_semantic_cache_call_leaves_cache_none(self):
        cap = ragradar_capture.start("q", pipeline="test")
        cap.response("r")
        assert self._stored_record()["cache"] is None

    def test_metadata_filter_all_fields(self):
        cap = ragradar_capture.start("q", pipeline="test")
        cap.metadata_filter(
            applied=True,
            candidate_count=10,
            excluded_count=4,
            filters={"source": "internal"},
        )
        cap.response("r")
        filt = self._stored_record()["filter"]
        assert filt == {
            "applied": True,
            "candidate_count": 10,
            "excluded_count": 4,
            "filters": {"source": "internal"},
        }

    def test_metadata_filter_defaults(self):
        cap = ragradar_capture.start("q", pipeline="test")
        cap.metadata_filter(applied=False)
        cap.response("r")
        filt = self._stored_record()["filter"]
        assert filt["applied"] is False
        assert filt["candidate_count"] is None
        assert filt["excluded_count"] is None
        assert filt["filters"] is None

    def test_no_metadata_filter_call_leaves_filter_none(self):
        cap = ragradar_capture.start("q", pipeline="test")
        cap.response("r")
        assert self._stored_record()["filter"] is None

    def test_token_usage_total_derived(self):
        ragradar_capture.capture(
            "q",
            "r",
            token_usage={"input_tokens": 100, "output_tokens": 50},
        )
        usage = self._stored_record()["token_usage"]
        assert usage["total_tokens"] == 150

    def test_capture_one_liner_all_primitives_round_trips(self):
        run_id = ragradar_capture.capture(
            "q",
            "r",
            chunks=[{"content": "chunk text", "retrieval_score": 0.8}],
            history_pre=[{"user": "hi"}],
            history_post=[{"assistant": "hello"}],
            token_budget=8192,
            cache_events={"chunk_0": True},
            tool_calls=[{"tool_name": "search", "arguments": {}}],
            token_usage={"input_tokens": 10, "output_tokens": 5},
            pipeline="test",
        )
        assert run_id is not None
        data = self._stored_record()
        record = RunRecord.from_json(data)
        assert record.chunks[0].token_count == coerce.estimate_tokens("chunk text")
        assert record.history_pre[0].role == "user"
        assert record.token_budget.total_limit == 8192
        assert record.cache_events[0].hit is True
        assert record.token_usage.total_tokens == 15


class TestFailureSilence:
    def test_capture_failure_never_raises_and_returns_none(self, monkeypatch):
        def fail(*a, **k):
            raise RuntimeError("boom")

        monkeypatch.setattr("ragradar_core.store.commit_run", fail)
        assert ragradar_capture.capture("q", "r") is None

    def test_capture_method_failure_never_raises(self):
        cap = ragradar_capture.start("q", pipeline="test")
        cap.chunks("not a list of dicts")
        cap.context(None)
        cap.history(None, None)
        cap.cache("bad")
        cap.tool_call("not a dict")

    def test_commit_failure_never_raises_returns_none(self, monkeypatch):
        def fail(*a, **k):
            raise RuntimeError("db error")

        monkeypatch.setattr("ragradar_core.store.commit_run", fail)
        cap = ragradar_capture.start("q", pipeline="test")
        cap._record.response = "r"
        assert cap.commit() is None
        assert cap.run_id is None


class TestStrictMode:
    def test_bad_chunk_raises_in_strict(self):
        ragradar_capture.set_strict(True)
        cap = ragradar_capture.start("q", pipeline="test")
        with pytest.raises(TypeError):
            cap.chunks([{"bad_field": 1}])

    def test_commit_error_raises_in_strict(self, monkeypatch):
        def fail(*a, **k):
            raise RuntimeError("db error")

        monkeypatch.setattr("ragradar_core.store.commit_run", fail)
        ragradar_capture.set_strict(True)
        cap = ragradar_capture.start("q", pipeline="test")
        cap._record.response = "r"
        with pytest.raises(RuntimeError, match="db error"):
            cap.commit()

    def test_env_var_enables_strict(self, monkeypatch):
        monkeypatch.setenv("RAGRADAR_CAPTURE_STRICT", "1")
        cap = ragradar_capture.start("q", pipeline="test")
        with pytest.raises(TypeError):
            cap.chunks([{"bad_field": 1}])

    def test_non_strict_logs_to_errors_log(self, ragradar_home):
        cap = ragradar_capture.start("q", pipeline="test")
        cap.chunks([{"bad_field": 1}])  # swallowed

        errors_log = ragradar_home / ".ragradar" / "errors.log"
        assert errors_log.exists()
        assert "capture.chunks() failed" in errors_log.read_text(encoding="utf-8")


class TestThreadLocal:
    def setup_method(self):
        clear_active_capture()

    def test_proxy_routes_to_active_capture(self):
        cap = ragradar_capture.start("proxy test", pipeline="test")
        ragradar_capture.chunks(
            [
                {
                    "chunk_id": "c1",
                    "source_doc_id": "d1",
                    "content": "hello",
                    "token_count": 5,
                }
            ]
        )
        assert cap._record.chunks is not None
        assert len(cap._record.chunks) == 1
        assert cap._record.chunks[0].chunk_id == "c1"

        ragradar_capture.tool_call({"tool_name": "search", "arguments": {"query": "RRF"}})
        assert len(cap._record.tool_calls) == 1
        assert cap._record.tool_calls[0].tool_name == "search"

        run_id = ragradar_capture.response("proxy response")
        assert run_id == cap.run_id
        assert run_id is not None

    def test_semantic_cache_proxy_routes_to_active_capture(self):
        cap = ragradar_capture.start("proxy test", pipeline="test")
        ragradar_capture.semantic_cache(checked=True, hit=True, similarity_score=0.95, threshold=0.9)
        assert cap._record.cache is not None
        assert cap._record.cache.hit is True
        assert cap._record.cache.similarity_score == 0.95

    def test_metadata_filter_proxy_routes_to_active_capture(self):
        cap = ragradar_capture.start("proxy test", pipeline="test")
        ragradar_capture.metadata_filter(applied=True, candidate_count=10, excluded_count=3)
        assert cap._record.filter is not None
        assert cap._record.filter.applied is True
        assert cap._record.filter.excluded_count == 3

    def test_proxy_without_active_capture_is_silent(self):
        ragradar_capture.chunks([])
        ragradar_capture.context("prompt")
        ragradar_capture.tool_call({"tool_name": "search", "arguments": {}})
        ragradar_capture.semantic_cache(checked=True)
        assert ragradar_capture.response("r") is None
        assert ragradar_capture.commit() is None

    def test_proxy_without_active_capture_logs(self, ragradar_home):
        ragradar_capture.chunks([])
        errors_log = ragradar_home / ".ragradar" / "errors.log"
        assert errors_log.exists()
        assert "no active capture" in errors_log.read_text(encoding="utf-8")

    def test_active_capture_accessible(self):
        cap = ragradar_capture.start("q", pipeline="test")
        assert get_active_capture() is cap

    def test_context_proxy_budget_flow(self):
        cap = ragradar_capture.start("proxy q", pipeline="test")
        ragradar_capture.context(
            "final prompt",
            {
                "total_limit": 100,
                "chunks_allocated": 50,
                "history_allocated": 20,
                "system_allocated": 10,
                "headroom": 20,
            },
        )
        assert isinstance(cap._record.token_budget, TokenBudget)
        run_id = ragradar_capture.response("resp")
        assert run_id is not None
