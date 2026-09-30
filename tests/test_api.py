"""HTTP contract. The model is not loaded."""

import numpy as np
from fastapi.testclient import TestClient

from app.main import app


def test_health() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_generate_returns_text(monkeypatch) -> None:
    monkeypatch.setattr("app.main.generate_text", lambda image: '{"s":[2,2],"r":[]}')
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    ok, encoded = __import__("cv2").imencode(".png", image)
    assert ok
    client = TestClient(app)
    response = client.post(
        "/v1/generate",
        files={"image": ("drawing.png", encoded.tobytes(), "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["text"] == '{"s":[2,2],"r":[]}'
    assert body["width"] == 2
    assert body["height"] == 2


def test_generate_rejects_empty() -> None:
    client = TestClient(app)
    response = client.post(
        "/v1/generate",
        files={"image": ("drawing.png", b"", "image/png")},
    )
    assert response.status_code == 400
