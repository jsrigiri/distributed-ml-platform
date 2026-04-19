from pyspark.sql import SparkSession
from core.config import RAW_DATA_PATH, OFFLINE_FEATURE_PATH
from spark.feature_pipeline import build_features


def run_batch_job():
    spark = (
        SparkSession.builder
        .appName("distributed-ml-platform")
        .master("local[*]")
        .getOrCreate()
    )

    events_df = spark.read.csv(RAW_DATA_PATH, header=True, inferSchema=True)
    feature_df = build_features(events_df)

    feature_df.write.mode("overwrite").parquet(OFFLINE_FEATURE_PATH)
    print(f"Saved offline features to {OFFLINE_FEATURE_PATH}")

    spark.stop()


if __name__ == "__main__":
    run_batch_job()