from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "distributed_ml_platform"
    app_env: str = "dev"
    log_level: str = "INFO"

    raw_data_path: str = "data/raw_events.csv"
    offline_store_path: str = "data/offline_features.parquet"

    model_dir: str = "artifacts/models"
    report_dir: str = "artifacts/reports"
    model_path: str = "artifacts/models/model.pkl"
    model_meta_path: str = "artifacts/models/model_meta.json"

    online_warmup: int = 5
    random_state: int = 42
    test_size: float = 0.2
    min_accuracy_to_promote: float = 0.60

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()

Path(settings.model_dir).mkdir(parents=True, exist_ok=True)
Path(settings.report_dir).mkdir(parents=True, exist_ok=True)
Path("data").mkdir(parents=True, exist_ok=True)

FEATURE_COLUMNS = [
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
]

TARGET_COLUMN = "target_clf"