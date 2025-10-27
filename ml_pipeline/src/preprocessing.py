import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and standardizes raw Ahmedabad real estate data.
    Adjust the logic below according to your 1_data_extraction_and_cleaning.ipynb.
    """
    # Example cleaning logic (you’ll extend this from your notebook)
    df = df.copy()

    # Drop duplicates and NAs
    df.drop_duplicates(inplace=True)
    df.dropna(subset=['price'], inplace=True)

    # Clean price column
    df['price'] = (df['price']
                   .astype(str)
                   .str.replace('₹', '', regex=False)
                   .str.replace('Lac', '', regex=False)
                   .str.replace(',', '', regex=False)
                   .astype(float))

    # Example area cleaning
    df['value_area'] = df['value_area'].astype(str).str.replace('sqft', '').str.replace('sqyrd', '').str.strip()
    df['value_area'] = pd.to_numeric(df['value_area'], errors='coerce')

    # Drop unwanted columns
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')

    return df


def encode_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes categorical variables using LabelEncoder.
    """
    df = df.copy()
    label_encoders = {}
    cat_cols = df.select_dtypes(exclude=np.number).columns

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    joblib.dump(label_encoders, "data/processed/label_encoders.pkl")
    return df


def preprocess_pipeline(raw_path: str, save_path: str = "data/processed/cleaned_data.csv") -> pd.DataFrame:
    """
    Full preprocessing pipeline: load → clean → encode → save
    """
    df = pd.read_csv(raw_path)
    df = clean_data(df)
    df = encode_data(df)
    df.to_csv(save_path, index=False)
    return df
