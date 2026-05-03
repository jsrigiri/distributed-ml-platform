import json
import time
from pathlib import Path

import pandas as pd

from core.config import settings


def main():
    input_path = Path(settings.stream_input_path)
    input_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(settings.raw_data_path)

    batch_size = 50
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i + batch_size]
        records = batch.to_dict(orient="records")

        out_file = input_path / f"events_{i//batch_size:04d}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

        print(f"Wrote {len(records)} events to {out_file}")
        time.sleep(2)


if __name__ == "__main__":
    main()