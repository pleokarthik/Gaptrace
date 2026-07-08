from pathlib import Path

from gaptrace_core import store as _core_store
from gaptrace_core.schema import RunRecord

from gaptrace.explain.analyzers import (
    cache as cache_mod,
)
from gaptrace.explain.analyzers import (
    duplicates as duplicates_mod,
)
from gaptrace.explain.analyzers import (
    history as history_mod,
)
from gaptrace.explain.analyzers import (
    scores as scores_mod,
)
from gaptrace.explain.analyzers import (
    tokens as tokens_mod,
)
from gaptrace.explain.analyzers import (
    truncation as truncation_mod,
)


def _esc(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def _section(title: str, content: str) -> str:
    return (
        f"<details open>\n  <summary>{_esc(title)}</summary>\n"
        f"  <pre>{_esc(content)}</pre>\n</details>\n"
    )


def render(record: RunRecord, run_id: str) -> Path:
    reports_dir = _core_store._gaptrace_dir() / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out = reports_dir / f"{run_id}.html"

    sections = ""

    tok = tokens_mod.analyze(record)
    if tok:
        body = f"Total: {tok['total_tokens']}"
        if tok["model_limit"]:
            body += f" / {tok['model_limit']} ({tok['utilisation_pct']}%)"
        body += f"\nChunks:   {tok['chunks_tokens']}"
        body += f"\nHistory:  {tok['history_tokens']}"
        body += f"\nSystem:   {tok['system_tokens']}"
        body += f"\nHeadroom: {tok['headroom']}"
        if tok["per_chunk"]:
            body += "\n\nPer chunk:"
            for pc in tok["per_chunk"]:
                body += f"\n  {pc['chunk_id']}: {pc['token_count']}"
        sections += _section("Token Usage", body)

    sc = scores_mod.analyze(record)
    if sc:
        body = ""
        if sc["retrieval_scores"]:
            body += f"Retrieval: {sc['bottom_retrieval']:.2f}–{sc['top_retrieval']:.2f}\n"
        if sc["rerank_scores"]:
            body += f"Rerank:    {sc['bottom_rerank']:.2f}–{sc['top_rerank']:.2f}\n"
        if sc["rerank_delta"] is not None:
            body += f"Rerank delta: {sc['rerank_delta']:+.4f}\n"
        body += f"Low-score ratio: {sc['low_score_ratio']:.0%}"
        sections += _section("Chunk Scores", body)

    dup = duplicates_mod.analyze(record)
    if dup:
        body = f"Path dups: {len(dup['path_dups'])}\n"
        body += f"Window dups: {len(dup['window_dups'])}\n"
        body += f"Duplicate ratio: {dup['duplicate_ratio']:.0%}"
        for d in dup["path_dups"]:
            body += f"\n  [PATH DUP] {d['chunk_id']} via {', '.join(d['paths'])}"
        for d in dup["window_dups"]:
            ids = ", ".join(d["chunk_ids"])
            body += f"\n  [WINDOW DUP] {ids} (source: {d['source_doc_id']})"
        sections += _section("Duplicate Chunks", body)

    tr = truncation_mod.analyze(record)
    if tr:
        body = f"Truncated: {tr['truncated_count']} chunks\n"
        body += f"High-score truncations: {tr['high_score_truncations']}\n"
        body += f"Severity: {tr['severity']}"
        for tc in tr["truncated_chunks"]:
            body += f"\n  {tc['chunk_id']}: retrieval={tc['score']}, rerank={tc['rerank_score']}"
        sections += _section("Truncation", body)

    hist = history_mod.analyze(record)
    if hist:
        body = (
            f"Turns: {hist['pre_turn_count']} -> {hist['post_turn_count']} "
            f"({hist['dropped_turn_count']} dropped)\n"
        )
        if hist["eviction_reason"]:
            body += f"Reason: {hist['eviction_reason']}\n"
        if hist["pre_tokens"] is not None:
            body += f"Pre tokens: {hist['pre_tokens']}\n"
        if hist["post_tokens"] is not None:
            body += f"Post tokens: {hist['post_tokens']}\n"
        for t in hist["dropped_turns"]:
            body += f"\n  [{t.role}] {t.content}"
        sections += _section("Dropped History", body)

    ca = cache_mod.analyze(record)
    if ca:
        body = f"Hits: {ca['hits']}/{ca['total_events']} ({ca['hit_ratio']:.0%})\n"
        if ca["hit_chunks"]:
            body += f"Hit chunks: {', '.join(ca['hit_chunks'])}\n"
        if ca["miss_chunks"]:
            body += f"Miss chunks: {', '.join(ca['miss_chunks'])}"
        sections += _section("Cache Hits", body)

    if record.final_prompt:
        sections += _section("Final Prompt", record.final_prompt)

    model_line = ""
    if record.model:
        model_line = f"<p><b>Model:</b> {_esc(record.model)}</p>"

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>gaptrace report — {_esc(run_id)}</title>
<style>
body {{ font-family: system-ui, monospace; max-width: 900px; margin: 2em auto; padding: 0 1em; }}
details {{ margin: 1em 0; border: 1px solid #ccc; border-radius: 4px; padding: 0.5em 1em; }}
summary {{ cursor: pointer; font-weight: bold; padding: 0.3em 0; }}
pre {{ background: #f5f5f5; padding: 1em; overflow-x: auto; white-space: pre-wrap; }}
h1 {{ color: #333; }}
.meta {{ color: #666; margin-bottom: 2em; }}
</style>
</head>
<body>
<h1>gaptrace report — {_esc(run_id)}</h1>
<div class="meta">
<p><b>Query:</b> {_esc(record.query)}</p>
<p><b>Response:</b> {_esc(record.response[:500])}</p>
{model_line}
</div>
{sections}
</body>
</html>"""

    out.write_text(html, encoding="utf-8")
    return out
