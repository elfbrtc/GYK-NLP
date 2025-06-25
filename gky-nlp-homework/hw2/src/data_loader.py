import pandas as pd

def load_dataset(json_path):
    """
    loading Kaggle News Category Dataset
    """
    df = pd.read_json(json_path, lines=True)
    return df[["category", "headline"]].dropna()
