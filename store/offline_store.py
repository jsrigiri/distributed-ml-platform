import os
import shutil

import pandas as pd


class OfflineFeatureStore:
    def __init__(self, path: str):
        self.path = path

    def write_spark_df(self, spark_df) -> None:
        if os.path.exists(self.path):
            if os.path.isdir(self.path):
                shutil.rmtree(self.path)
            else:
                os.remove(self.path)
        spark_df.write.mode("overwrite").parquet(self.path)

    def write_pandas_df(self, df: pd.DataFrame) -> None:
        if os.path.exists(self.path):
            if os.path.isdir(self.path):
                shutil.rmtree(self.path)
            else:
                os.remove(self.path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        df.to_parquet(self.path, index=False)

    def read_pandas_df(self) -> pd.DataFrame:
        return pd.read_parquet(self.path)