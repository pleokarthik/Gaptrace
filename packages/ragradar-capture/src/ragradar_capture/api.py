import logging
import os

from ragradar_core import store
from ragradar_core.coerce import (
    coerce_cache_events,
    coerce_cache_record,
    coerce_chunks,
    coerce_token_budget,
    coerce_token_usage,
    coerce_tool_call,
    coerce_turns,
)
from ragradar_core.schema import (
    RunRecord,
    TokenBudget,
    TokenUsage,
    ToolCallRecord,
)

from ragradar_capture.thread_local import get_active_capture, set_active_capture

_strict = False


def set_strict(enabled: bool) -> None:
    """Toggle strict mode for this process. Pure (sets a module flag).

    In strict mode, conversion/commit errors inside capture calls RAISE
    instead of being logged to ~/.ragradar/errors.log — use it in development
    to surface instrumentation bugs. The default (False) keeps the
    production never-raise contract. The RAGRADAR_CAPTURE_STRICT=1 environment
    variable enables strict mode without a code change.
    """
    global _strict
    _strict = enabled


def _strict_enabled() -> bool:
    return _strict or os.environ.get("RAGRADAR_CAPTURE_STRICT", "") == "1"


def _get_logger():
    logger = logging.getLogger("ragradar-capture")
    if not logger.handlers:
        log_dir = store._ragradar_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(str(log_dir / "errors.log"))
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [ragradar-capture] %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.ERROR)
    return logger


def _handle_error(where: str, exc: Exception) -> None:
    """Strict mode: re-raise. Otherwise log to errors.log and swallow."""
    if _strict_enabled():
        raise exc
    try:
        _get_logger().error("%s failed: %s", where, exc)
    except Exception:
        pass


class Capture:
    """The action object for staged instrumentation of one pipeline run.

    Built by ragradar_capture.start(); each method records one pipeline stage
    onto the underlying RunRecord ("run" stays the data noun). Methods
    never raise unless strict mode is on — failures go to
    ~/.ragradar/errors.log and the pipeline continues.
    """

    def __init__(self, query: str, pipeline: str | None = None):
        self._query = query
        self._pipeline = pipeline
        self._record = RunRecord(query=query, response="")
        self._run_id: str | None = None

    @property
    def run_id(self) -> str | None:
        """The committed run's sNrN id, or None before commit (or if the
        commit failed in non-strict mode). Pure."""
        return self._run_id

    def chunks(self, chunks: list) -> None:
        """Record the retrieval stage: a list of chunks.

        Plain dicts need only "content" — ids and token counts are
        filled in (see ragradar_core.coerce); ChunkRecords pass through
        untouched. Mutates this capture only (store write happens at
        commit). Never raises unless strict mode is on.
        """
        try:
            self._record.chunks = coerce_chunks(chunks)
        except Exception as e:
            _handle_error("capture.chunks()", e)

    def context(
        self, final_prompt: str, token_budget: TokenBudget | dict | int | None = None
    ) -> None:
        """Record the assembly stage: the final prompt and optional budget.

        token_budget takes a bare int (the total limit), a partial dict,
        or a TokenBudget; missing headroom/allocations are derived (see
        ragradar_core.coerce). Mutates this capture only. Never raises
        unless strict mode is on.
        """
        try:
            self._record.final_prompt = final_prompt
            if token_budget is not None:
                self._record.token_budget = coerce_token_budget(token_budget, final_prompt)
        except Exception as e:
            _handle_error("capture.context()", e)

    def history(self, pre: list, post: list, eviction_reason: str | None = None) -> None:
        """Record the history-management stage: turns before/after eviction.

        Turns take the shorthand {"user": "..."} / {"assistant": "..."}
        dicts, ("role", "content") pairs, full dicts, or Turn objects;
        token counts are estimated unless given (see ragradar_core.coerce).
        ``eviction_reason`` matches capture()'s parameter of the same
        name. Mutates this capture only. Never raises unless strict mode
        is on.
        """
        try:
            self._record.history_pre = coerce_turns(pre)
            self._record.history_post = coerce_turns(post)
            self._record.eviction_reason = eviction_reason
        except Exception as e:
            _handle_error("capture.history()", e)

    def response(
        self,
        response: str,
        token_usage: TokenUsage | dict | None = None,
        model: str | None = None,
    ) -> str | None:
        """Record the LLM output stage, then auto-commit.

        Writes to store (via commit). Returns the committed run's sNrN id,
        or None if the write failed in non-strict mode. Never raises
        unless strict mode is on.
        """
        try:
            self._record.response = response
            self._record.model = model
            if token_usage is not None:
                self._record.token_usage = coerce_token_usage(token_usage)
            return self.commit()
        except Exception as e:
            _handle_error("capture.response()", e)
            return None

    def cache(self, events) -> None:
        """Record cache events.

        Takes a {chunk_id: hit} mapping for the whole call, or a list of
        ("chunk_id", hit) pairs / dicts / CacheEvents. Mutates this
        capture only. Never raises unless strict mode is on.
        """
        try:
            self._record.cache_events = coerce_cache_events(events)
        except Exception as e:
            _handle_error("capture.cache()", e)

    def semantic_cache(
        self,
        checked: bool,
        hit: bool = False,
        similarity_score: float | None = None,
        threshold: float | None = None,
        cached_query: str | None = None,
        cached_at: str | None = None,
        registered: bool = False,
    ) -> None:
        """Record the semantic-cache stage: whether this query's answer
        was served from (or checked against) a semantic cache, before
        retrieval ran.

        This is the query-level cache check — distinct from cache()'s
        per-chunk retrieval-cache events. Mutates this capture only
        (store write happens at commit). Never raises unless strict
        mode is on.
        """
        try:
            self._record.cache = coerce_cache_record(
                {
                    "checked": checked,
                    "hit": hit,
                    "similarity_score": similarity_score,
                    "threshold": threshold,
                    "cached_query": cached_query,
                    "cached_at": cached_at,
                    "registered": registered,
                }
            )
        except Exception as e:
            _handle_error("capture.semantic_cache()", e)

    def tool_call(self, call: ToolCallRecord | dict) -> None:
        """Append one tool call (ToolCallRecord or dict) — never replaces.

        Mutates this capture only. Never raises unless strict mode is on.
        """
        try:
            record = coerce_tool_call(call)
            if self._record.tool_calls is None:
                self._record.tool_calls = []
            self._record.tool_calls.append(record)
        except Exception as e:
            _handle_error("capture.tool_call()", e)

    def commit(self) -> str | None:
        """Write the run to the store and return its sNrN id.

        Writes to store. Idempotent: a second call returns the same id
        without writing again. Returns None if the write failed in
        non-strict mode (the failure is logged to ~/.ragradar/errors.log);
        never raises unless strict mode is on.

        Session resolution, run_seq assignment, and the insert happen in
        one atomic transaction (``store.commit_run``) so concurrent
        commits to the same session can't race on ``(session_id,
        run_seq)`` — see ``ragradar_core.store.commit_run`` for why this
        needs to be one call rather than three.
        """
        if self._run_id is not None:
            return self._run_id
        try:
            session_id, run_seq = store.commit_run(self._pipeline, self._record)
            self._run_id = f"s{session_id}r{run_seq}"
            return self._run_id
        except Exception as e:
            _handle_error("capture.commit()", e)
            return None


def start(query: str, pipeline: str | None = None) -> Capture:
    """Begin a staged capture and make it this thread's active capture.

    Pure until commit (only thread-local state is set here). Returns the
    Capture handle; the same capture is also reachable through the
    module-level proxy functions from any code on this thread.
    """
    cap = Capture(query, pipeline)
    set_active_capture(cap)
    return cap


def capture(
    query: str,
    response: str,
    *,
    chunks: list | None = None,
    final_prompt: str | None = None,
    token_budget: TokenBudget | dict | int | None = None,
    history_pre: list | None = None,
    history_post: list | None = None,
    eviction_reason: str | None = None,
    cache_events: list | dict | None = None,
    tool_calls: list | None = None,
    model: str | None = None,
    token_usage: TokenUsage | dict | None = None,
    pipeline: str | None = None,
) -> str | None:
    """One-line capture: record a complete run in a single call.

    Every argument takes plain primitives — chunk dicts needing only
    "content", {"user": "..."} history turns, a bare int token_budget,
    a {chunk_id: hit} cache mapping — as well as the schema dataclasses
    (see ragradar_core.coerce for the exact coercion rules).

    Writes to store. Returns the committed run's sNrN id (e.g. "s2r3"),
    or None if an internal failure was swallowed in non-strict mode (the
    failure is logged to ~/.ragradar/errors.log). Never raises unless strict
    mode is on — except for unknown keyword arguments, which fail
    naturally with TypeError at call time (the signature is explicit,
    not **kwargs).

    token_budget is persisted whether or not final_prompt is given.
    """
    try:
        cap = Capture(query, pipeline)
        cap._record.response = response
        if chunks is not None:
            cap.chunks(chunks)
        if final_prompt is not None:
            cap._record.final_prompt = final_prompt
        if token_budget is not None:
            cap._record.token_budget = coerce_token_budget(token_budget, final_prompt)
        if history_pre is not None or history_post is not None:
            cap.history(history_pre or [], history_post or [], eviction_reason)
        elif eviction_reason is not None:
            cap._record.eviction_reason = eviction_reason
        if cache_events is not None:
            cap.cache(cache_events)
        if tool_calls is not None:
            for call in tool_calls:
                cap.tool_call(call)
        if model is not None:
            cap._record.model = model
        if token_usage is not None:
            cap._record.token_usage = coerce_token_usage(token_usage)
        return cap.commit()
    except Exception as e:
        _handle_error("capture()", e)
        return None


# Thread-local proxies — free functions that delegate to the thread's
# active Capture so deep call stacks don't need the handle threaded
# through every signature. With no active capture they log and no-op
# (never raise), matching the fail-open contract.


def chunks(chunks: list) -> None:
    """Proxy for the active capture's .chunks(). Mutates the active
    capture; logs and no-ops if there is none."""
    cap = get_active_capture()
    if cap is None:
        _get_logger().error("chunks() called with no active capture")
        return
    cap.chunks(chunks)


def context(final_prompt: str, token_budget: TokenBudget | dict | int | None = None) -> None:
    """Proxy for the active capture's .context(). Mutates the active
    capture; logs and no-ops if there is none."""
    cap = get_active_capture()
    if cap is None:
        _get_logger().error("context() called with no active capture")
        return
    cap.context(final_prompt, token_budget)


def history(pre: list, post: list, eviction_reason: str | None = None) -> None:
    """Proxy for the active capture's .history(). Mutates the active
    capture; logs and no-ops if there is none."""
    cap = get_active_capture()
    if cap is None:
        _get_logger().error("history() called with no active capture")
        return
    cap.history(pre, post, eviction_reason)


def response(
    response: str,
    token_usage: TokenUsage | dict | None = None,
    model: str | None = None,
) -> str | None:
    """Proxy for the active capture's .response() (which auto-commits —
    writes to store). Returns the run id, or None with a log entry if
    there is no active capture."""
    cap = get_active_capture()
    if cap is None:
        _get_logger().error("response() called with no active capture")
        return None
    return cap.response(response, token_usage, model)


def cache(events) -> None:
    """Proxy for the active capture's .cache(). Mutates the active
    capture; logs and no-ops if there is none."""
    cap = get_active_capture()
    if cap is None:
        _get_logger().error("cache() called with no active capture")
        return
    cap.cache(events)


def semantic_cache(
    checked: bool,
    hit: bool = False,
    similarity_score: float | None = None,
    threshold: float | None = None,
    cached_query: str | None = None,
    cached_at: str | None = None,
    registered: bool = False,
) -> None:
    """Proxy for the active capture's .semantic_cache(). Mutates the
    active capture; logs and no-ops if there is none."""
    cap = get_active_capture()
    if cap is None:
        _get_logger().error("semantic_cache() called with no active capture")
        return
    cap.semantic_cache(checked, hit, similarity_score, threshold, cached_query, cached_at, registered)


def tool_call(call: ToolCallRecord | dict) -> None:
    """Proxy for the active capture's .tool_call(). Mutates the active
    capture; logs and no-ops if there is none."""
    cap = get_active_capture()
    if cap is None:
        _get_logger().error("tool_call() called with no active capture")
        return
    cap.tool_call(call)


def commit() -> str | None:
    """Proxy for the active capture's .commit() (writes to store).

    Returns the run id, or None with a log entry if there is no active
    capture."""
    cap = get_active_capture()
    if cap is None:
        _get_logger().error("commit() called with no active capture")
        return None
    return cap.commit()
