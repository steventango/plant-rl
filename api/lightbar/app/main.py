import json
from typing import List

import numpy as np
from fastapi import Depends, FastAPI, Response
from pydantic import BaseModel
from typing_extensions import Annotated

from .lightbar import Lightbar
from .zones import zone2

app = FastAPI()
zone = zone2


class Action(BaseModel):
    array: List[float]


def get_lightbar_left():
    return Lightbar(zone.left)


def get_lightbar_right():
    return Lightbar(zone.right)


@app.put("/action/left", response_class=Response)
def update_action_left(action: Action, lightbar: Annotated[any, Depends(get_lightbar_left)]):
    lightbar.step(np.array(action.array))


@app.put("/action/right", response_class=Response)
def update_action_right(action: Action, lightbar: Annotated[any, Depends(get_lightbar_right)]):
    lightbar.step(np.array(action.array))
