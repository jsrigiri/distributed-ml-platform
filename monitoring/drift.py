import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def compute_feature_baseline(
    df: pd.DataFrame,
    feature_columns: List[str],
) -> Dict[str, Dict[str, float]]:
    baseline = {}

    for col in feature_columns:
        mean = float(df[col].mean())
        std = float(df[col].std())

        baseline[col] = {
            "mean": mean,
            "std": std if std > 0 else 1e-8,
        }

    return baseline


def save_baseline(baseline: dict, path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")


def load_baseline(path: str) -> dict:
    baseline_path = Path(path)
    if not baseline_path.exists():
        return {}
    return json.loads(baseline_path.read_text(encoding="utf-8"))


def detect_drift(
    live_features: Dict[str, float],
    baseline: Dict[str, Dict[str, float]],
    threshold: float = 3.0,
) -> Tuple[bool, Dict[str, Dict[str, float]]]:
    drift_report = {}
    drift_detected = False

    for feature, value in live_features.items():
        if feature not in baseline:
            continue

        train_mean = baseline[feature]["mean"]
        train_std = baseline[feature]["std"]

        z_score = abs((float(value) - train_mean) / train_std)

        is_drifted = z_score > threshold
        if is_drifted:
            drift_detected = True

        drift_report[feature] = {
            "value": float(value),
            "train_mean": float(train_mean),
            "train_std": float(train_std),
            "z_score": float(z_score),
            "drifted": bool(is_drifted),
        }

    return drift_detected, drift_report