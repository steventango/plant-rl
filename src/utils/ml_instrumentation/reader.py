from pathlib import Path
from typing import Any
import traceback
import connectorx as cx

import utils.ml_instrumentation._utils.sqlite as sqlu


def get_run_ids(
    db_path: str | Path, params: dict[str, Any], data_path: str | Path | None = None
):
    constraints = " AND ".join(
        f"[{k}]={sqlu.maybe_quote(v)}" for k, v in params.items()
    )

    query = f"SELECT id FROM _metadata_ WHERE {constraints}"
    try:
        return cx.read_sql(f"sqlite://{db_path}", query, return_type="polars")[
            "id"
        ].to_list()
    except BaseException:
        traceback.print_exc()
        print("Error occurred while fetching run IDs from DB")

    if data_path is None:
        return []

    run_ids = []
    for path in Path(data_path).glob("*.npz"):
        run_id = int(path.stem)
        run_ids.append(run_id)
    return run_ids
