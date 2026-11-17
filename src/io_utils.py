import pandas as pd
import numpy as np

def load_file(path: str) -> pd.DataFrame:
    """
    Läser in CSV-fil
    """
    df = pd.read_csv(path)
    print(f"Data har laddats in utan problem från: {path} (Totalt {len(df)} rader)")
    return df

def check_file(df: pd.DataFrame):
    """
    Ger oss infon om datan samt kollar så det inte finns dubletter
    """
    print("Data info:\n")
    df.info()

    print("Dubletter: ", df.duplicated().sum())
    