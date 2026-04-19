import os

import joblib
import pandas as pd

from core.config import FEATURE_COLUMNS
from core.exceptions import ModelNotReadyError


class Predictor:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.reload_if_exists()

    def reload_if_exists(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)

    def is_ready(self) -> bool:
        return self.model is not None

    def predict(self, df: pd.DataFrame):
        if self.model is None:
            raise ModelNotReadyError("Model is not loaded")
        return self.model.predict(df[FEATURE_COLUMNS])

    def predict_proba(self, df: pd.DataFrame):
        if self.model is None:
            raise ModelNotReadyError("Model is not loaded")
        return self.model.predict_proba(df[FEATURE_COLUMNS])