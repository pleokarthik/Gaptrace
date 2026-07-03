import json
from datetime import datetime
from pathlib import Path

from ragradar_core import store


def export(pipeline: str | None = None, output_path: Path | None = None) -> Path:
    runs = store.get_all_evaluated_runs(pipeline)

    records = []
    for r in runs:
        if r["pipeline"] and r["pipeline"].endswith("__seeded"):
            continue
        run_data = json.loads(r["run_data"])
        if not run_data.get("chunks") or not run_data.get("response"):
            continue
        records.append((r, run_data))

    if output_path is None:
        pipe_name = pipeline or "all"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exports_dir = store._ragradar_dir() / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)
        output_path = exports_dir / f"{pipe_name}_ragas_{timestamp}.jsonl"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for row, run_data in records:
            entry = {
                "question": run_data["query"],
                "answer": run_data["response"],
                "contexts": [c["content"] for c in run_data.get("chunks", [])],
                "ground_truth": None,
                "run_id": f"s{row['session_id']}r{row['run_seq']}",
                "pipeline": row["pipeline"],
                "evaluated_at": row["evaluated_at"],
            }
            f.write(json.dumps(entry) + "\n")

    return output_path
