import os
import joblib
import pandas as pd

def save_model(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

def load_model(path: str):
    return joblib.load(path)

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
