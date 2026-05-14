from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from ..app.main import app

client = TestClient(app)


def test_homepage():
    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200, json=lambda: None)
        response = client.get("/")
        assert response.status_code == 200
