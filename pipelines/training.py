import mlflow
import mlflow.sklearn

from core.config import FEATURE_COLUMNS, settings
from core.exceptions import ModelPromotionError
from core.logging import get_logger
from models.registry import ModelRegistry
from models.train import train_models
from store.offline_store import OfflineFeatureStore
from scripts.promote_challenger import promote

logger = get_logger(__name__)


def run_training_pipeline(store_path: str):
    store = OfflineFeatureStore(store_path)
    df = store.read_pandas_df()

    with mlflow.start_run(run_name="distributed_ml_platform_training"):
        mlflow.log_param("random_state", settings.random_state)
        mlflow.log_param("test_size", settings.test_size)
        mlflow.log_param("online_warmup", settings.online_warmup)

        results = train_models(df, FEATURE_COLUMNS)

        best_model_name = None
        best_score = -1.0
        best_model = None

        all_metrics = {}

        for name, result in results.items():
            metrics = result["metrics"]
            all_metrics[name] = metrics

            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(f"{name}_{metric_name}", metric_value)

            if "accuracy" in metrics:
                score = metrics["accuracy"]
                logger.info("%s accuracy=%.4f", name, score)

                if score > best_score:
                    best_score = score
                    best_model_name = name
                    best_model = result["model"]

        if best_model is None:
            raise ModelPromotionError("No classifier model available for promotion.")

        if best_score < settings.min_accuracy_to_promote:
            raise ModelPromotionError(
                f"Model did not meet promotion threshold: {best_score:.4f}"
            )

        mlflow.log_param("best_model_name", best_model_name)
        mlflow.log_metric("best_model_accuracy", best_score)
        mlflow.sklearn.log_model(best_model, artifact_path="best_model")

        challenger_registry = ModelRegistry(settings.challenger_model_path, settings.challenger_meta_path)
        challenger_registry.save(
            best_model,
            {
                "status": "challenger",
                "best_model": best_model_name,
                "score": best_score,
                "all_models": all_metrics,
            },
        )

        promote()

        return best_model, best_score