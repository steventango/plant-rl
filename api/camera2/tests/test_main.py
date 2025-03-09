import io
from functools import cache

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from ..app.main import app, get_camera
from .mock_camera import MockCamera


@cache
def get_mock_camera():
    return MockCamera()


app.dependency_overrides[get_camera] = get_mock_camera

client = TestClient(app)


def test_observation():
    response = client.get("/observation")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["content-disposition"] == 'inline; filename="observation.png"'

    image = Image.open(io.BytesIO(response.content))
    array = np.array(image)
    assert array.shape == (1944, 2592, 3)
    assert array.dtype == np.uint8


def test_get_camera_singleton():
    camera1 = get_mock_camera()
    camera2 = get_mock_camera()
    assert camera1 is camera2
