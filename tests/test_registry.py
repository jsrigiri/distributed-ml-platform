from pathlib import Path
import tempfile

from models.registry import ModelRegistry


def test_registry_metadata_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = str(Path(tmpdir) / "model.pkl")
        meta_path = str(Path(tmpdir) / "model_meta.json")

        registry = ModelRegistry(model_path, meta_path)
        registry.save(model={"dummy": True}, metadata={"status": "promoted"})

        assert registry.exists()
        meta = registry.load_metadata()
        assert meta["status"] == "promoted"
        assert "saved_at_utc" in meta