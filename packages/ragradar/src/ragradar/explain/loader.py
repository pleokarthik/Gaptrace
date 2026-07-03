import json

from ragradar_core.schema import RunRecord


def load_run_record(run_row: dict) -> RunRecord:
    data = json.loads(run_row["run_data"])
    return RunRecord.from_json(data)
