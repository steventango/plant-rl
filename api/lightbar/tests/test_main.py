import numpy as np
from fastapi.testclient import TestClient

from ..app.main import app, get_lightbar
from ..app.zones import Zone
from .mock_lightbar import MockLightbar

lightbar = MockLightbar(Zone(0x69, 0x71))


def get_mock_lightbar():
    return lightbar


app.dependency_overrides[get_lightbar] = get_mock_lightbar

client = TestClient(app)


def test_action():
    response = client.put("/action", json={"array": np.ones((2, 6)).tolist()})
    assert response.status_code == 200
    assert lightbar.i2c.data[0x69][3][0] == [0, 0x06, 0, 0, 0x55, 0x05]
    assert lightbar.i2c.data[0x69][3][1] == [0, 0x0A, 0, 0, 0x55, 0x05]
    assert lightbar.i2c.data[0x69][3][2] == [0, 0x1E, 0, 0, 0x55, 0x05]
    assert lightbar.i2c.data[0x69][3][3] == [0, 0x1A, 0, 0, 0x55, 0x05]
    assert lightbar.i2c.data[0x69][3][4] == [0, 0x16, 0, 0, 0x55, 0x05]
    assert lightbar.i2c.data[0x69][3][5] == [0, 0x0E, 0, 0, 0x55, 0x05]
    assert lightbar.i2c.data[0x71][3][0] == [0, 0x06, 0, 0, 0x55, 0x05]
    assert lightbar.i2c.data[0x71][3][1] == [0, 0x0A, 0, 0, 0x55, 0x05]
    assert lightbar.i2c.data[0x71][3][2] == [0, 0x1E, 0, 0, 0x55, 0x05]
    assert lightbar.i2c.data[0x71][3][3] == [0, 0x1A, 0, 0, 0x55, 0x05]
    assert lightbar.i2c.data[0x71][3][4] == [0, 0x16, 0, 0, 0x55, 0x05]
    assert lightbar.i2c.data[0x71][3][5] == [0, 0x0E, 0, 0, 0x55, 0x05]


def test_action_latest():
    response = client.get("/action/latest")
    assert response.status_code == 200
    assert response.json() == {"action": None, "safe_action": None}

    response = client.put("/action", json={"array": np.ones((2, 6)).tolist()})
    assert response.status_code == 200

    response = client.get("/action/latest")
    assert response.status_code == 200
    assert response.json() == {
        "action": np.ones((2, 6)).tolist(),
        "safe_action": (np.ones((2, 6)) / 3).tolist(),
    }
