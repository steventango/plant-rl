import numpy as np
from fastapi.testclient import TestClient

from ..app.main import app, get_lightbar_left, get_lightbar_right
from .mock_lightbar import MockLightbar

lightbar_left = MockLightbar(0x69)
lightbar_right = MockLightbar(0x71)


def get_mock_lightbar_left():
    return lightbar_left


def get_mock_lightbar_right():
    return lightbar_right


app.dependency_overrides[get_lightbar_left] = get_mock_lightbar_left
app.dependency_overrides[get_lightbar_right] = get_mock_lightbar_right

client = TestClient(app)


def test_action_left():
    response = client.put("/action/left", json={"array": np.ones(6).tolist()})
    assert response.status_code == 200
    assert lightbar_left.i2c.data[0x69][3][0] == [0, 0x06, 0, 0, 0x55, 0x05]
    assert lightbar_left.i2c.data[0x69][3][1] == [0, 0x0A, 0, 0, 0x55, 0x05]
    assert lightbar_left.i2c.data[0x69][3][2] == [0, 0x1E, 0, 0, 0x55, 0x05]
    assert lightbar_left.i2c.data[0x69][3][3] == [0, 0x1A, 0, 0, 0x55, 0x05]
    assert lightbar_left.i2c.data[0x69][3][4] == [0, 0x16, 0, 0, 0x55, 0x05]
    assert lightbar_left.i2c.data[0x69][3][5] == [0, 0x0E, 0, 0, 0x55, 0x05]


def test_action_right():
    response = client.put("/action/right", json={"array": np.ones(6).tolist()})
    assert response.status_code == 200
    assert lightbar_right.i2c.data[0x71][3][0] == [0, 0x06, 0, 0, 0x55, 0x05]
    assert lightbar_right.i2c.data[0x71][3][1] == [0, 0x0A, 0, 0, 0x55, 0x05]
    assert lightbar_right.i2c.data[0x71][3][2] == [0, 0x1E, 0, 0, 0x55, 0x05]
    assert lightbar_right.i2c.data[0x71][3][3] == [0, 0x1A, 0, 0, 0x55, 0x05]
    assert lightbar_right.i2c.data[0x71][3][4] == [0, 0x16, 0, 0, 0x55, 0x05]
    assert lightbar_right.i2c.data[0x71][3][5] == [0, 0x0E, 0, 0, 0x55, 0x05]
