from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def get_spark(app_name: str = "distributed_ml_platform") -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .getOrCreate()
    )


def build_batch_features(spark: SparkSession, input_path: str):
    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(input_path)
    )

    w_user_order = Window.partitionBy("user_id").orderBy("event_time_idx")
    w5 = Window.partitionBy("user_id").orderBy("event_time_idx").rowsBetween(-4, 0)
    w_count = Window.partitionBy("user_id").orderBy("event_time_idx").rowsBetween(
        Window.unboundedPreceding, 0
    )

    features = (
        df.withColumn("lag_event_value_1", F.lag("event_value", 1).over(w_user_order))
        .withColumn("lag_amount_1", F.lag("amount", 1).over(w_user_order))
        .withColumn("rolling_mean_event_value_5", F.avg("event_value").over(w5))
        .withColumn("rolling_mean_amount_5", F.avg("amount").over(w5))
        .withColumn("rolling_std_event_value_5", F.stddev_samp("event_value").over(w5))
        .withColumn("rolling_mobile_rate_5", F.avg("is_mobile").over(w5))
        .withColumn("events_seen", F.count(F.lit(1)).over(w_count))
        .dropna(
            subset=[
                "lag_event_value_1",
                "lag_amount_1",
                "rolling_mean_event_value_5",
                "rolling_mean_amount_5",
                "rolling_std_event_value_5",
                "rolling_mobile_rate_5",
            ]
        )
    )

    return features.select(
        "user_id",
        "event_time_idx",
        "event_value",
        "amount",
        "hour",
        "is_mobile",
        "lag_event_value_1",
        "lag_amount_1",
        "rolling_mean_event_value_5",
        "rolling_mean_amount_5",
        "rolling_std_event_value_5",
        "rolling_mobile_rate_5",
        "events_seen",
        "target_clf",
        "target_reg",
    )