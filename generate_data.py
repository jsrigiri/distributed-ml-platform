from pathlib import Path

import numpy as np
import pandas as pd

np.random.seed(42)

n_users = 500
events_per_user = 40

rows = []
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
            }
        )

df = pd.DataFrame(rows)

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

Path("data").mkdir(exist_ok=True)
df.to_csv("data/raw_events.csv", index=False)

print(f"Saved {len(df)} rows to data/raw_events.csv")
print(df.head())