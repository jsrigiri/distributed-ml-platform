from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from core.config import settings


def get_streaming_spark(app_name: str = "distributed_ml_platform_streaming") -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .getOrCreate()
    )


def run_streaming_pipeline():
    spark = get_streaming_spark()

    schema = """
        user_id INT,
        event_time_idx INT,
        event_value DOUBLE,
        amount DOUBLE,
        hour INT,
        is_mobile INT,
        target_reg DOUBLE,
        target_clf INT
    """

    stream_df = (
        spark.readStream
        .schema(schema)
        .option("maxFilesPerTrigger", 1)
        .json(settings.stream_input_path)
    )

    # Simple streaming transformations
    featured = (
        stream_df
        .withColumn("event_value_x_amount", F.col("event_value") * F.col("amount"))
        .withColumn("is_late_hour", F.when(F.col("hour") >= 18, 1).otherwise(0))
    )

    query = (
        featured.writeStream
        .format("parquet")
        .option("path", settings.stream_output_path)
        .option("checkpointLocation", settings.stream_checkpoint_path)
        .outputMode("append")
        .start()
    )

    print("Streaming pipeline started.")
    print(f"Watching: {settings.stream_input_path}")
    print(f"Writing to: {settings.stream_output_path}")

    query.awaitTermination()