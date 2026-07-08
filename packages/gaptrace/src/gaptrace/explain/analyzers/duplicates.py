from gaptrace_core.schema import RunRecord


def analyze(record: RunRecord) -> dict | None:
    if not record.chunks:
        return None

    # PATH DUP: same chunk_id appears multiple times with different retrieval_path
    chunk_paths: dict[str, list[str]] = {}
    for c in record.chunks:
        chunk_paths.setdefault(c.chunk_id, [])
        if c.retrieval_path:
            chunk_paths[c.chunk_id].append(c.retrieval_path)

    path_dups = [
        {"chunk_id": cid, "paths": paths} for cid, paths in chunk_paths.items() if len(paths) > 1
    ]

    # WINDOW DUP: same source_doc_id, overlapping content (substring check)
    by_source: dict[str, list] = {}
    for c in record.chunks:
        by_source.setdefault(c.source_doc_id, []).append(c)

    window_dups = []
    for source_id, chunks in by_source.items():
        if len(chunks) < 2:
            continue
        dup_ids: set[str] = set()
        for i, a in enumerate(chunks):
            for b in chunks[i + 1 :]:
                if a.content in b.content or b.content in a.content:
                    dup_ids.add(a.chunk_id)
                    dup_ids.add(b.chunk_id)
        if dup_ids:
            window_dups.append({"chunk_ids": sorted(dup_ids), "source_doc_id": source_id})

    dup_chunk_ids: set[str] = set()
    for d in path_dups:
        dup_chunk_ids.add(d["chunk_id"])
    for d in window_dups:
        dup_chunk_ids.update(d["chunk_ids"])

    total = len(record.chunks)
    return {
        "path_dups": path_dups,
        "window_dups": window_dups,
        "semantic_dups": [],
        "duplicate_ratio": len(dup_chunk_ids) / total if total else 0.0,
    }
