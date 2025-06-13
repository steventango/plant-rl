import os
from functools import lru_cache
from typing import List

import numpy as np
from fastapi import Depends, FastAPI, Response
from pydantic import BaseModel
from typing_extensions import Annotated

from .lightbar import Lightbar
from .zones import ZONES

app = FastAPI()
zone = ZONES.get(os.getenv("ZONE", "mitacs-zone2"))


class Action(BaseModel):
    array: List[List[float]]


@lru_cache(maxsize=None)
def get_lightbar():
    return Lightbar(zone)


@app.put("/action", response_class=Response)
def update_action(action: Action, lightbar: Annotated[any, Depends(get_lightbar)]):
    lightbar.step(np.array(action.array))


@app.get("/action/latest")
def get_current_action(lightbar: Lightbar = Depends(get_lightbar)):
    return {
        "action": lightbar.action.tolist() if lightbar.action is not None else None,
        "safe_action": lightbar.safe_action.tolist() if lightbar.safe_action is not None else None
    }
