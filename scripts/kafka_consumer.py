import json

from kafka import KafkaConsumer

from core.config import settings
from pipelines.online_features import OnlineFeatureBuilder


def main():
    consumer = KafkaConsumer(
        settings.kafka_topic_events,
        bootstrap_servers=settings.kafka_bootstrap_servers,
        group_id=settings.kafka_consumer_group,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    )

    builder = OnlineFeatureBuilder(warmup=settings.online_warmup)

    print(f"Listening to Kafka topic: {settings.kafka_topic_events}")

    for msg in consumer:
        event = msg.value

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
        else:
            print(
                f"user_id={event['user_id']} features_ready={features}"
            )


if __name__ == "__main__":
    main()