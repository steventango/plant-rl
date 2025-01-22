import numpy as np
from flask import Flask, request

from api.lightbar import convert_to_duty_cycle, set_duty_cycle

app = Flask(__name__)


@app.get("/observation")
def observation():
    raise NotImplementedError


@app.put("/action")
def action():
    data = request.get_json()
    data = np.array(data)
    duty_cycle = convert_to_duty_cycle(data)
    set_duty_cycle(duty_cycle)
    return duty_cycle.tolist()
