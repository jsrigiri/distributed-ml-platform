import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipelines.streaming import run_streaming_pipeline


if __name__ == "__main__":
    run_streaming_pipeline()