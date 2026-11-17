import pandas as pd
import numpy as np
from IO_utils import load_file

class Healthanalyzer:
    def __init__(self, path: str):
        self.df: pd.DataFrame = load_file(path)

    def desc_stats(self) -> pd.DataFrame:
        return (
            self.df[["age", "weight", "height", "systolic_bp", "cholesterol"]]
            .agg(["mean", "median", "min", "max"])
        )
