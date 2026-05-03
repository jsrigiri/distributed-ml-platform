import json
import time
from pathlib import Path

import pandas as pd
from kafka import KafkaProducer

from core.config import settings


def json_serializer(value):
    return json.dumps(value).encode("utf-8")


def main():
    producer = KafkaProducer(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        value_serializer=json_serializer,
    )

    #df = pd.read_csv(settings.raw_data_path)
    #df = pd.read_csv(settings.raw_data_drift_path)
    df = pd.read_csv("data/raw_events_with_drift.csv")
    print(df[["event_value", "amount", "is_drifted"]].head(5))

    for _, row in df.iterrows():
        event = {
            "user_id": int(row["user_id"]),
            "event_time_idx": int(row["event_time_idx"]),
            "event_value": float(row["event_value"]),
            "amount": float(row["amount"]),
            "hour": int(row["hour"]),
            "is_mobile": int(row["is_mobile"]),
            "target_reg": float(row["target_reg"]),
            "target_clf": int(row["target_clf"]),
        }

        producer.send(settings.kafka_topic_events, event)
        print(f"Sent event: {event}")
        time.sleep(0.1)

    producer.flush()
    producer.close()


if __name__ == "__main__":
    main()