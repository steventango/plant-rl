from typing import List

import numpy as np
from fastapi import Depends, FastAPI, Response
from pydantic import BaseModel
from typing_extensions import Annotated

from .lightbar import Lightbar

app = FastAPI()


class Action(BaseModel):
    array: List[List[float]]


def get_lightbar():
    return Lightbar()


@app.put("/action", response_class=Response)
def update_action(action: Action, lightbar: Annotated[any, Depends(get_lightbar)]):
    lightbar.step(np.array(action.array))
