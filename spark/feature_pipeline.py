from pyspark.sql import functions as F


def build_features(events_df):
    features = (
        events_df.groupBy("user_id")
        .agg(
            F.count("*").alias("event_count"),
            F.avg("event_value").alias("event_value_mean"),
            F.stddev("event_value").alias("event_value_std"),
            F.avg("amount").alias("amount_mean"),
            F.stddev("amount").alias("amount_std"),
            F.avg("is_mobile").alias("mobile_rate"),
            F.avg("hour").alias("avg_hour"),
            F.max("target_reg").alias("target_reg"),
            F.max("target_clf").alias("target_clf"),
        )
        .fillna(0.0)
    )

    features = (
        features
        .withColumn("amount_per_event", F.col("amount_mean") / (F.col("event_count") + F.lit(1e-8)))
        .withColumn("event_value_x_mobile", F.col("event_value_mean") * F.col("mobile_rate"))
    )

    return features