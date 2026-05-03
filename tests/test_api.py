import os

import joblib
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from core.config import FEATURE_COLUMNS, settings


def ensure_model_exists():
    if not os.path.exists(settings.champion_model_path):
        X = pd.DataFrame(
            [
                {
                    "event_value": 1.0,
                    "amount": 10.0,
                    "hour": 9,
                    "is_mobile": 1,
                    "lag_event_value_1": 0.8,
                    "lag_amount_1": 9.0,
                    "rolling_mean_event_value_5": 0.95,
                    "rolling_mean_amount_5": 10.2,
                    "rolling_std_event_value_5": 0.15,
                    "rolling_mobile_rate_5": 0.6,
                    "events_seen": 5,
                },
                {
                    "event_value": 0.4,
                    "amount": 5.0,
                    "hour": 15,
                    "is_mobile": 0,
                    "lag_event_value_1": 0.5,
                    "lag_amount_1": 5.5,
                    "rolling_mean_event_value_5": 0.45,
                    "rolling_mean_amount_5": 5.2,
                    "rolling_std_event_value_5": 0.07,
                    "rolling_mobile_rate_5": 0.2,
                    "events_seen": 5,
                },
                {
                    "event_value": 1.2,
                    "amount": 11.0,
                    "hour": 11,
                    "is_mobile": 1,
                    "lag_event_value_1": 1.0,
                    "lag_amount_1": 10.5,
                    "rolling_mean_event_value_5": 1.05,
                    "rolling_mean_amount_5": 10.8,
                    "rolling_std_event_value_5": 0.12,
                    "rolling_mobile_rate_5": 0.8,
                    "events_seen": 6,
                },
                {
                    "event_value": 0.3,
                    "amount": 4.5,
                    "hour": 18,
                    "is_mobile": 0,
                    "lag_event_value_1": 0.4,
                    "lag_amount_1": 5.0,
                    "rolling_mean_event_value_5": 0.38,
                    "rolling_mean_amount_5": 4.9,
                    "rolling_std_event_value_5": 0.08,
                    "rolling_mobile_rate_5": 0.1,
                    "events_seen": 6,
                },
            ]
        )
        y = [1, 0, 1, 0]
        model = LogisticRegression(max_iter=1000)
        model.fit(X[FEATURE_COLUMNS], y)

        os.makedirs(os.path.dirname(settings.champion_model_path), exist_ok=True)
        joblib.dump(model, settings.champion_model_path)

        meta = {
            "status": "promoted",
            "metrics": {"accuracy": 1.0},
            "info": {"feature_columns": FEATURE_COLUMNS},
        }
        os.makedirs(os.path.dirname(settings.champion_meta_path), exist_ok=True)
        import json

        with open(settings.champion_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)


ensure_model_exists()

from api.app import app  # noqa: E402

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_ready():
    response = client.get("/ready")
    assert response.status_code == 200
    assert "ready" in response.json()


def test_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.json()
    assert "prediction_requests" in body
    assert "online_updates" in body


def test_model_info():
    response = client.get("/model_info")
    assert response.status_code == 200
    assert "status" in response.json()


def test_predict():
    payload = {
        "event_value": 1.0,
        "amount": 10.0,
        "hour": 9,
        "is_mobile": 1,
        "lag_event_value_1": 0.8,
        "lag_amount_1": 9.0,
        "rolling_mean_event_value_5": 0.95,
        "rolling_mean_amount_5": 10.2,
        "rolling_std_event_value_5": 0.15,
        "rolling_mobile_rate_5": 0.6,
        "events_seen": 5,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert "probability_positive" in body
    assert body["model_ready"] is True