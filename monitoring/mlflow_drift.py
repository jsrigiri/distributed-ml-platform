import json
from pathlib import Path
from typing import Dict

import mlflow


def summarize_drift_report(drift_report: Dict[str, Dict[str, float]]) -> dict:
    if not drift_report:
        return {
            "max_drift_zscore": 0.0,
            "mean_drift_zscore": 0.0,
            "num_drifted_features": 0,
            "num_checked_features": 0,
        }

    zscores = [float(v["z_score"]) for v in drift_report.values()]
    num_drifted = sum(1 for v in drift_report.values() if v["drifted"])

    return {
        "max_drift_zscore": float(max(zscores)),
        "mean_drift_zscore": float(sum(zscores) / len(zscores)),
        "num_drifted_features": int(num_drifted),
        "num_checked_features": int(len(drift_report)),
    }


def log_drift_to_mlflow(
    drift_detected: bool,
    drift_report: Dict[str, Dict[str, float]],
    threshold: float,
    artifact_dir: str = "artifacts/reports",
) -> dict:
    summary = summarize_drift_report(drift_report)

    output_dir = Path(artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "latest_drift_report.json"
    report_payload = {
        "drift_detected": bool(drift_detected),
        "threshold": float(threshold),
        "summary": summary,
        "report": drift_report,
    }

    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    with mlflow.start_run(run_name="drift_monitoring"):
        mlflow.log_param("drift_method", "zscore")
        mlflow.log_param("drift_threshold", float(threshold))

        mlflow.log_metric("drift_detected", int(drift_detected))
        mlflow.log_metric("max_drift_zscore", summary["max_drift_zscore"])
        mlflow.log_metric("mean_drift_zscore", summary["mean_drift_zscore"])
        mlflow.log_metric("num_drifted_features", summary["num_drifted_features"])
        mlflow.log_metric("num_checked_features", summary["num_checked_features"])

        mlflow.log_artifact(str(report_path))

    return summary