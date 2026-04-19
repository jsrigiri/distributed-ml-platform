from models.predict import predict_from_features


def run_inference(online_store, user_id: int):
    row = online_store.get(user_id)
    if row is None:
        return None

    feature_row = {
        "event_count": row.get("event_count", 0.0),
        "event_value_mean": row.get("event_value_mean", 0.0),
        "event_value_std": row.get("event_value_std", 0.0),
        "amount_mean": row.get("amount_mean", 0.0),
        "amount_std": row.get("amount_std", 0.0),
        "mobile_rate": row.get("mobile_rate", 0.0),
        "avg_hour": row.get("avg_hour", 0.0),
        "amount_per_event": row.get("amount_per_event", 0.0),
        "event_value_x_mobile": row.get("event_value_x_mobile", 0.0),
    }

    pred = predict_from_features(feature_row)
    return pred