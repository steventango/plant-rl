import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

DATA_ROOT = Path("/data")
APP_ROOT = Path(__file__).parent

app = FastAPI()

app.mount("/static", StaticFiles(directory=DATA_ROOT), name="static")
templates = Jinja2Templates(directory=APP_ROOT / "templates")


@app.get("/", response_class=HTMLResponse)
async def list_datasets(request: Request):
    # Recursively find all raw.csv files to identify dataset directories
    dataset_paths = DATA_ROOT.rglob("raw.csv")
    datasets = []
    for p in dataset_paths:
        try:
            df = pd.read_csv(p)
            if not df.empty:
                latest_event = df.iloc[-1]
                time = latest_event["time"]
                image_name = latest_event["image_name"]
                dataset_path = p.parent
                image_dir = "images" if (dataset_path / "images").exists() else ""
                image_path = f"/static/{p.parent.relative_to(DATA_ROOT)}/{image_dir}/{image_name}"
                datasets.append(
                    {
                        "path": p.parent.relative_to(DATA_ROOT),
                        "time": time,
                        "image_path": image_path,
                    }
                )
        except (pd.errors.EmptyDataError, KeyError):
            # Ignore empty or malformed CSV files
            continue

    datasets.sort(key=lambda x: x["time"], reverse=True)

    return templates.TemplateResponse(
        "datasets.html", {"request": request, "datasets": datasets}
    )


@app.get("/dataset/{dataset_name:path}", response_class=HTMLResponse)
async def show_dataset(request: Request, dataset_name: str):
    dataset_path = DATA_ROOT / dataset_name
    csv_path = dataset_path / "raw.csv"
    if not csv_path.exists():
        return Response(content="Dataset not found", status_code=404)

    df = pd.read_csv(csv_path)

    image_dir = "images" if (dataset_path / "images").exists() else ""
    events = []
    for time, group in df.groupby("time"):
        row = group.iloc[0]
        left_image_name = row["image_name"]
        left_image_name = left_image_name.replace("_right", "_left")
        right_image_name = left_image_name.replace("_left", "_right")

        left_image_path = None
        if left_image_name:
            left_image_path = f"/static/{dataset_name}/{image_dir}/{left_image_name}"

        right_image_path = None
        if right_image_name:
            right_image_path = f"/static/{dataset_name}/{image_dir}/{right_image_name}"

        actions = {
            f"action_{i}": group.iloc[0][f"action.{i}"]
            for i in range(6)
            if f"action.{i}" in group.columns
        }
        mean_area = None
        if "area" in group.columns:
            mean_area = group["area"].mean()

        events.append(
            {
                "time": time,
                "left_image_path": left_image_path,
                "right_image_path": right_image_path,
                **actions,  # Include all action data
                "mean_area": mean_area,
            }
        )

    events_json = json.dumps({"events": events})

    return templates.TemplateResponse(
        "dataset.html",
        {
            "request": request,
            "dataset_name": dataset_name,
            "events_json": events_json,
        },
    )
