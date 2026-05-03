from core.config import settings
from models.predict import Predictor
from models.registry import ModelRegistry
from pipelines.online_features import OnlineFeatureBuilder

champion_predictor = Predictor(settings.champion_model_path)
challenger_predictor = Predictor(settings.challenger_model_path)

champion_registry = ModelRegistry(
    settings.champion_model_path,
    settings.champion_meta_path,
)
challenger_registry = ModelRegistry(
    settings.challenger_model_path,
    settings.challenger_meta_path,
)

online_builder = OnlineFeatureBuilder(warmup=settings.online_warmup)