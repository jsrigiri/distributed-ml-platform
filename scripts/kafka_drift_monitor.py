import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from kafka import KafkaConsumer

from core.config import FEATURE_COLUMNS, settings
from monitoring.drift import detect_drift, load_baseline
from monitoring.mlflow_drift import log_drift_to_mlflow
from pipelines.online_features import OnlineFeatureBuilder


def main():
    baseline = load_baseline(settings.drift_baseline_path)

    if not baseline:
        raise FileNotFoundError(
            "Drift baseline not found. Run `python main.py` first."
        )

    consumer = KafkaConsumer(
        settings.kafka_topic_events,
        bootstrap_servers=settings.kafka_bootstrap_servers,
        group_id=f"{settings.kafka_consumer_group}-drift",
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    )

    builder = OnlineFeatureBuilder(warmup=settings.online_warmup)

    print(f"Kafka drift monitor listening to topic: {settings.kafka_topic_events}")

    for msg in consumer:
        event = msg.value

        print("RAW EVENT RECEIVED:", event)

        print(
            "is_drifted=",
            event.get("is_drifted"),
            "event_value=",
            event.get("event_value"),
            "amount=",
            event.get("amount"),
        )

        raw_live_features = {
            "event_value": float(event["event_value"]),
            "amount": float(event["amount"]),
            "hour": int(event["hour"]),
            "is_mobile": int(event["is_mobile"]),
        }

        print("Baseline has keys:", list(baseline.keys())[:20])
        print("Drift threshold:", settings.drift_zscore_threshold)

        raw_drift_detected, raw_drift_report = detect_drift(
            live_features=raw_live_features,
            baseline=baseline,
            threshold=settings.drift_zscore_threshold,
        )

        print("RAW DRIFT DETECTED:", raw_drift_detected)
        print("RAW DRIFT REPORT:", raw_drift_report)

        top_raw = sorted(
            raw_drift_report.items(),
            key=lambda x: x[1]["z_score"],
            reverse=True,
        )[:4]

        print("RAW DRIFT CHECK:")
        for feature, stats in top_raw:
            print(
                feature,
                "value=", round(stats["value"], 4),
                "mean=", round(stats["train_mean"], 4),
                "std=", round(stats["train_std"], 4),
                "z=", round(stats["z_score"], 4),
                "drifted=", stats["drifted"],
            )

        if raw_drift_detected:
            summary = log_drift_to_mlflow(
                drift_detected=True,
                drift_report=raw_drift_report,
                threshold=settings.drift_zscore_threshold,
            )
            print(f"RAW DRIFT DETECTED and logged to MLflow: {summary}")

        features = builder.update(
            user_id=int(event["user_id"]),
            event_value=float(event["event_value"]),
            amount=float(event["amount"]),
            hour=int(event["hour"]),
            is_mobile=int(event["is_mobile"]),
        )

        if features is None:
            print(
                f"user_id={event['user_id']} warming up "
                f"event_time_idx={event['event_time_idx']}"
            )
            continue

        # Keep only model feature columns
        live_features = {
            col: features[col]
            for col in FEATURE_COLUMNS
            if col in features
        }

        drift_detected, drift_report = detect_drift(
            live_features=live_features,
            baseline=baseline,
            threshold=settings.drift_zscore_threshold,
        )

        top_drift = sorted(
            drift_report.items(),
            key=lambda x: x[1]["z_score"],
            reverse=True,
        )[:5]

        print("Top drift scores:")
        for feature, stats in top_drift:
            print(
                feature,
                "value=", stats["value"],
                "mean=", stats["train_mean"],
                "std=", stats["train_std"],
                "z=", stats["z_score"],
                "drifted=", stats["drifted"],
            )

        max_z = max(
            [v["z_score"] for v in drift_report.values()],
            default=0.0,
        )

        print(
            f"user_id={event['user_id']} "
            f"drift_detected={drift_detected} "
            f"max_zscore={max_z:.4f}"
        )

        if drift_detected:
            summary = log_drift_to_mlflow(
                drift_detected=drift_detected,
                drift_report=drift_report,
                threshold=settings.drift_zscore_threshold,
            )
            print(f"Logged drift to MLflow: {summary}")


if __name__ == "__main__":
    main()