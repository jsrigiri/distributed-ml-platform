from pathlib import Path

import numpy as np
import pandas as pd

np.random.seed(42)

n_users = 500
events_per_user = 40

rows = []

# -----------------------------
# Normal data
# -----------------------------
for user_id in range(1, n_users + 1):
    base_activity = np.random.uniform(0.2, 1.5)

    for t in range(events_per_user):
        event_value = np.random.normal(loc=base_activity, scale=0.5)
        amount = np.random.lognormal(mean=2.0, sigma=0.5)
        hour = np.random.randint(0, 24)
        is_mobile = np.random.randint(0, 2)

        rows.append(
            {
                "user_id": user_id,
                "event_time_idx": t,
                "event_value": float(event_value),
                "amount": float(amount),
                "hour": int(hour),
                "is_mobile": int(is_mobile),
                "is_drifted": 0,
            }
        )

df = pd.DataFrame(rows)


# -----------------------------
# Create user-level targets
# -----------------------------
agg = (
    df.groupby("user_id")
    .agg(
        mean_event_value=("event_value", "mean"),
        mean_amount=("amount", "mean"),
        mobile_rate=("is_mobile", "mean"),
    )
    .reset_index()
)

score = (
    0.8 * agg["mean_event_value"]
    + 0.3 * agg["mean_amount"] / agg["mean_amount"].std()
    + 0.4 * agg["mobile_rate"]
    + np.random.normal(0, 0.5, len(agg))
)

agg["target_reg"] = score
agg["target_clf"] = (score > score.median()).astype(int)

df = df.merge(
    agg[["user_id", "target_reg", "target_clf"]],
    on="user_id",
    how="left",
)


# -----------------------------
# Drifted data for Kafka monitor
# -----------------------------
drift_rows = []

drift_users = range(1, 6) #range(1, 101)
drift_events_per_user = 20

for user_id in drift_users:
    for t in range(events_per_user, events_per_user + drift_events_per_user):
        # intentionally extreme values vs training distribution
        event_value = np.random.normal(loc=25.0, scale=2.0)
        amount = np.random.lognormal(mean=8.0, sigma=0.5)
        hour = np.random.randint(18, 24)
        is_mobile = 1

        event_value = 100.0
        amount = 10000.0
        hour = 23
        is_mobile = 1

        drift_rows.append(
            {
                "user_id": user_id,
                "event_time_idx": t,
                "event_value": float(event_value),
                "amount": float(amount),
                "hour": int(hour),
                "is_mobile": int(is_mobile),
                "is_drifted": 1,
                "target_reg": float(score.mean() + 5.0),
                "target_clf": 1,
            }
        )

df_drift = pd.DataFrame(drift_rows)


# -----------------------------
# Save files
# -----------------------------
Path("data").mkdir(exist_ok=True)

# Normal data for Spark training
df.to_csv("data/raw_events.csv", index=False)

# Combined data for Kafka streaming demo
df_stream = pd.concat([df_drift, df], ignore_index=True)
df_stream.to_csv("data/raw_events_with_drift.csv", index=False)

print(f"Saved {len(df)} normal rows to data/raw_events.csv")
print(f"Saved {len(df_stream)} streaming rows to data/raw_events_with_drift.csv")
print(f"Drift rows: {len(df_drift)}")
print(df_stream.tail())