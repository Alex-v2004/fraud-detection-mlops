import sys
sys.path.append('.')

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Fraud Detection API" in response.json()["message"]


def test_predict_legit():
    payload = {
        "features": [0.1, -1.2, 0.8, 0.5, -0.3, 1.2, 0.4, -0.8, 0.2, 0.6,
                     -0.4, 0.9, -1.1, 0.3, 0.7, -0.5, 1.0, -0.2, 0.8, -0.6,
                     0.4, -0.9, 0.1, 0.5, -0.3, 0.7, -0.8, 0.2, 150.0, 0.5]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "is_fraud" in response.json()
    assert "confidence" in response.json()


def test_predict_wrong_features():
    payload = {"features": [0.1, 0.2]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 400


def test_predict_fraud():
    payload = {
        "features": [-3.0, 4.0, -5.0, 3.5, -2.0, 1.5, -4.0, 3.0, -2.5, 1.0,
                     -3.5, 4.5, -6.0, 2.0, 1.5, -2.0, 3.0, -1.5, 2.5, -2.0,
                     1.0, -3.0, 2.0, 1.5, -2.5, 3.0, -2.0, 1.0, 1.0, 0.5]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200