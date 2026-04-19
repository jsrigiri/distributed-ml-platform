import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict

import joblib


class ModelRegistry:
    def __init__(self, model_path: str, meta_path: str):
        self.model_path = Path(model_path)
        self.meta_path = Path(meta_path)

    def save(self, model, metadata: Dict[str, Any]) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, self.model_path)

        payload = {
            **metadata,
            "saved_at_utc": datetime.now(UTC).isoformat(),
        }
        self.meta_path.write_text(json.dumps(payload, indent=2))

    def exists(self) -> bool:
        return self.model_path.exists() and self.meta_path.exists()

    def load_metadata(self) -> Dict[str, Any]:
        if not self.meta_path.exists():
            return {}
        return json.loads(self.meta_path.read_text())