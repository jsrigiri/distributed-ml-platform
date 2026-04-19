from collections import defaultdict, deque
from math import sqrt
from typing import Dict, Optional


class UserState:
    def __init__(self, warmup: int):
        self.warmup = warmup
        self.event_values = deque(maxlen=warmup)
        self.amounts = deque(maxlen=warmup)
        self.mobile_flags = deque(maxlen=warmup)
        self.events_seen = 0
        self.last_event_value = None
        self.last_amount = None


class OnlineFeatureBuilder:
    def __init__(self, warmup: int = 5):
        if warmup < 2:
            raise ValueError("warmup must be at least 2")
        self.warmup = warmup
        self.states = defaultdict(lambda: UserState(warmup))

    @staticmethod
    def _mean(values):
        return sum(values) / len(values)

    @staticmethod
    def _std(values):
        n = len(values)
        if n < 2:
            return 0.0
        mu = sum(values) / n
        var = sum((x - mu) ** 2 for x in values) / (n - 1)
        return sqrt(var)

    def update(
        self,
        user_id: int,
        event_value: float,
        amount: float,
        hour: int,
        is_mobile: int,
    ) -> Optional[Dict[str, float]]:
        state = self.states[user_id]
        state.events_seen += 1

        lag_event_value_1 = state.last_event_value
        lag_amount_1 = state.last_amount

        state.event_values.append(float(event_value))
        state.amounts.append(float(amount))
        state.mobile_flags.append(int(is_mobile))

        state.last_event_value = float(event_value)
        state.last_amount = float(amount)

        if (
            len(state.event_values) < self.warmup
            or lag_event_value_1 is None
            or lag_amount_1 is None
        ):
            return None

        return {
            "event_value": float(event_value),
            "amount": float(amount),
            "hour": int(hour),
            "is_mobile": int(is_mobile),
            "lag_event_value_1": float(lag_event_value_1),
            "lag_amount_1": float(lag_amount_1),
            "rolling_mean_event_value_5": float(self._mean(state.event_values)),
            "rolling_mean_amount_5": float(self._mean(state.amounts)),
            "rolling_std_event_value_5": float(self._std(state.event_values)),
            "rolling_mobile_rate_5": float(self._mean(state.mobile_flags)),
            "events_seen": int(state.events_seen),
        }