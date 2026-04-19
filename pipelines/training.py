from core.config import FEATURE_COLUMNS, settings
from core.logging import get_logger
from models.registry import ModelRegistry
from models.train import train_models
from store.offline_store import OfflineFeatureStore

logger = get_logger(__name__)


def run_training_pipeline(store_path: str):
    store = OfflineFeatureStore(store_path)
    df = store.read_pandas_df()

    results = train_models(df, FEATURE_COLUMNS)

    # Select best classifier (by accuracy)
    best_model_name = None
    best_score = -1
    best_model = None

    for name, result in results.items():
        if "metrics" in result and "accuracy" in result["metrics"]:
            score = result["metrics"]["accuracy"]
            logger.info(f"{name} accuracy={score:.4f}")

            if score > best_score:
                best_score = score
                best_model_name = name
                best_model = result["model"]

    logger.info(f"BEST MODEL: {best_model_name} score={best_score:.4f}")

    registry = ModelRegistry(settings.model_path, settings.model_meta_path)
    registry.save(
        best_model,
        {
            "best_model": best_model_name,
            "score": best_score,
            "all_models": {k: v["metrics"] for k, v in results.items()},
        },
    )

    return best_model, best_score