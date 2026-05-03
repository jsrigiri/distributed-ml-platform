import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from core.config import settings
from models.promotion import should_promote_classifier
from models.registry import ModelRegistry


def promote():
    challenger_registry = ModelRegistry(
        settings.challenger_model_path,
        settings.challenger_meta_path,
    )
    champion_registry = ModelRegistry(
        settings.champion_model_path,
        settings.champion_meta_path,
    )

    if not challenger_registry.exists():
        raise FileNotFoundError("Challenger model artifacts not found.")

    challenger_meta = challenger_registry.load_metadata()
    challenger_best_model = challenger_meta.get("best_model", "")
    challenger_metrics = challenger_meta.get("all_models", {}).get(
        challenger_best_model, {}
    )

    if not champion_registry.exists():
        Path(settings.champion_model_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(settings.challenger_model_path, settings.champion_model_path)
        shutil.copy2(settings.challenger_meta_path, settings.champion_meta_path)
        print("No champion found. Challenger promoted to champion.")
        return True

    champion_meta = champion_registry.load_metadata()
    champion_best_model = champion_meta.get("best_model", "")
    champion_metrics = champion_meta.get("all_models", {}).get(
        champion_best_model, {}
    )

    promote_decision, reason = should_promote_classifier(
        champion_metrics=champion_metrics,
        challenger_metrics=challenger_metrics,
    )

    if not promote_decision:
        print(f"Promotion rejected: {reason}")
        return False

    Path(settings.champion_model_path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(settings.challenger_model_path, settings.champion_model_path)
    shutil.copy2(settings.challenger_meta_path, settings.champion_meta_path)

    print(f"Challenger promoted to champion: {reason}")
    return True


if __name__ == "__main__":
    promote()