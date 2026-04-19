from core.config import settings
from core.logging import get_logger, setup_logging
from pipelines.batch_features import build_batch_features, get_spark
from pipelines.training import run_training_pipeline
from store.offline_store import OfflineFeatureStore

setup_logging()
logger = get_logger(__name__)


def main():
    logger.info("starting_batch_pipeline")
    spark = get_spark(settings.app_name)

    try:
        batch_df = build_batch_features(spark, settings.raw_data_path)
        logger.info("batch_features_built")

        store = OfflineFeatureStore(settings.offline_store_path)
        store.write_spark_df(batch_df)
        logger.info("offline_store_written path=%s", settings.offline_store_path)

        _, model_metrics = run_training_pipeline(settings.offline_store_path)
        logger.info("training_completed metrics=%s", model_metrics)

        print("Distributed ML platform pipeline completed successfully.")
    finally:
        spark.stop()
        logger.info("spark_stopped")


if __name__ == "__main__":
    main()