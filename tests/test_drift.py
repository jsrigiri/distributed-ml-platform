import tempfile
from pathlib import Path

import pandas as pd

from monitoring.drift import (
    compute_feature_baseline,
    detect_drift,
    load_baseline,
    save_baseline,
)


def test_compute_feature_baseline():
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0],
            "y": [10.0, 11.0, 12.0],
        }
    )

    baseline = compute_feature_baseline(df, ["x", "y"])

    assert "x" in baseline
    assert "mean" in baseline["x"]
    assert "std" in baseline["x"]


def test_detect_drift_true():
    baseline = {
        "x": {
            "mean": 0.0,
            "std": 1.0,
        }
    }

    live = {"x": 5.0}

    drift_detected, report = detect_drift(
        live_features=live,
        baseline=baseline,
        threshold=3.0,
    )

    assert drift_detected is True
    assert report["x"]["drifted"] is True


def test_detect_drift_false():
    baseline = {
        "x": {
            "mean": 0.0,
            "std": 1.0,
        }
    }

    live = {"x": 1.0}

    drift_detected, report = detect_drift(
        live_features=live,
        baseline=baseline,
        threshold=3.0,
    )

    assert drift_detected is False
    assert report["x"]["drifted"] is False


def test_baseline_save_load_roundtrip():
    baseline = {
        "x": {
            "mean": 1.0,
            "std": 0.5,
        }
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "baseline.json"

        save_baseline(baseline, str(path))
        loaded = load_baseline(str(path))

        assert loaded["x"]["mean"] == 1.0
        assert loaded["x"]["std"] == 0.5