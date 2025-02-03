import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from ..app.main import app, get_camera
from .mock_picamzero import Camera


def get_mock_camera():
    return Camera()


app.dependency_overrides[get_camera] = get_mock_camera

client = TestClient(app)


def test_observation():
    response = client.get("/observation")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpg"
    assert response.headers["content-disposition"] == 'inline; filename="observation.jpg"'

    image = Image.open(io.BytesIO(response.content))
    array = np.array(image)
    assert array.shape == (1944, 2592, 3)
    assert array.dtype == np.uint8
