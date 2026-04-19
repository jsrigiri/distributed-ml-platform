from core.config import settings
from models.predict import Predictor
from models.registry import ModelRegistry
from pipelines.online_features import OnlineFeatureBuilder

predictor = Predictor(settings.model_path)
registry = ModelRegistry(settings.model_path, settings.model_meta_path)
online_builder = OnlineFeatureBuilder(warmup=settings.online_warmup)