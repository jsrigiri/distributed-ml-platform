import pandas as pd

from core.config import FEATURE_COLUMNS
from models.train import train_models


def test_train_models_returns_results():
    rows = []
    for i in range(60):
        rows.append(
            {
                "event_value": 1.0 + 0.1 * (i % 3),
                "amount": 10.0 + i,
                "hour": i % 24,
                "is_mobile": i % 2,
                "lag_event_value_1": 0.9 + 0.1 * (i % 3),
                "lag_amount_1": 9.0 + i,
                "rolling_mean_event_value_5": 1.0,
                "rolling_mean_amount_5": 12.0,
                "rolling_std_event_value_5": 0.2,
                "rolling_mobile_rate_5": 0.6,
                "events_seen": 5 + i,
                "target_clf": i % 2,
                "target_reg": float(i) / 10.0,
            }
        )

    df = pd.DataFrame(rows)
    results = train_models(df, FEATURE_COLUMNS)

    assert "logistic_regression" in results
    assert "xgboost_classifier" in results
    assert "lightgbm_classifier" in results
    assert "xgboost_regressor" in results
    assert "lightgbm_regressor" in results

    for _, result in results.items():
        assert "model" in result
        assert "metrics" in result