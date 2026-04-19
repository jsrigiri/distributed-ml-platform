from datetime import UTC, datetime


class MetricsRegistry:
    def __init__(self):
        self.prediction_requests = 0
        self.last_prediction = None
        self.online_updates = 0
        self.online_features_ready = 0
        self.started_at = datetime.now(UTC).isoformat()

    def log_prediction(self, value: float):
        self.prediction_requests += 1
        self.last_prediction = float(value)

    def log_online_update(self, ready: bool):
        self.online_updates += 1
        if ready:
            self.online_features_ready += 1

    def snapshot(self):
        return {
            "prediction_requests": self.prediction_requests,
            "last_prediction": self.last_prediction,
            "online_updates": self.online_updates,
            "online_features_ready": self.online_features_ready,
            "started_at": self.started_at,
        }


metrics = MetricsRegistry()