import numpy as np
import pytest

from ..app import app as _app


@pytest.fixture
def app():
    yield _app


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def runner(app):
    return app.test_cli_runner()


def test_action(client):
    response = client.put("/action", json=np.zeros((2, 6)).tolist())
    assert response
    assert response.status_code == 200
