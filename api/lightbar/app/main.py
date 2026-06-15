import os
import subprocess
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import List
from zoneinfo import ZoneInfoNotFoundError

import numpy as np
from fastapi import Depends, FastAPI, Response
from pydantic import BaseModel
from typing_extensions import Annotated

from .lightbar import Lightbar
from .scheduler import Scheduler
from .zones import ZONES

zone = ZONES.get(os.getenv("ZONE", "mitacs-zone2"))


class Action(BaseModel):
    array: List[List[float]]


class ScheduleEntry(BaseModel):
    time: str
    action: List[List[float]]


class ScheduleBody(BaseModel):
    timezone: str
    entries: List[ScheduleEntry]


@lru_cache(maxsize=None)
def get_lightbar():
    return Lightbar(zone)


scheduler = Scheduler(get_lightbar)


@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler.load()
    scheduler.start()
    try:
        yield
    finally:
        scheduler.stop()


app = FastAPI(lifespan=lifespan)


@app.put("/action", response_class=Response)
def update_action(action: Action, lightbar: Annotated[any, Depends(get_lightbar)]):
    lightbar.step(np.array(action.array))


@app.get("/action/latest")
def get_current_action(lightbar: Lightbar = Depends(get_lightbar)):
    return {
        "action": lightbar.action.tolist() if lightbar.action is not None else None,
        "safe_action": lightbar.safe_action.tolist()
        if lightbar.safe_action is not None
        else None,
    }


@app.put("/schedule", response_class=Response)
def put_schedule(body: ScheduleBody):
    """Replace the on-device fallback schedule (clears old + sets new)."""
    entries = [entry.model_dump() for entry in body.entries]
    try:
        scheduler.set(body.timezone, entries)
    except ZoneInfoNotFoundError:
        return Response(
            content=f"Unknown timezone: {body.timezone}",
            media_type="text/plain",
            status_code=422,
        )
    except ValueError as e:
        return Response(content=str(e), media_type="text/plain", status_code=422)


@app.delete("/schedule", response_class=Response)
def delete_schedule():
    scheduler.clear()


@app.get("/schedule")
def get_schedule():
    return scheduler.snapshot()


@app.post("/reset", response_class=Response)
def reset(lightbar: Annotated[Lightbar, Depends(get_lightbar)]):
    lightbar.reset()


@app.post("/recover", response_class=Response)
def recover(lightbar: Annotated[Lightbar, Depends(get_lightbar)]):
    lightbar.scl_recover()


@app.get("/scan")
def scan():
    try:
        result = subprocess.run(
            ["i2cdetect", "-y", "1"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except FileNotFoundError:
        return Response(
            content="i2c-tools is not installed in this container",
            media_type="text/plain",
            status_code=503,
        )
    except subprocess.TimeoutExpired:
        return Response(
            content="i2cdetect timed out (bus may be wedged)",
            media_type="text/plain",
            status_code=504,
        )
    return Response(content=result.stdout, media_type="text/plain")
