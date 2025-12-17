from pathlib import Path
from typing import Any


def get_run_ids(
    db_path: str | Path, params: dict[str, Any], data_path: str | Path | None = None
):
    if data_path is None:
        return []

    run_ids = []
    for path in Path(data_path).glob("*.npz"):
        run_id = int(path.stem)
        run_ids.append(run_id)
    return run_ids
