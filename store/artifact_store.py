from pathlib import Path


class ArtifactStore:
    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, name: str) -> str:
        return str(self.root / name)