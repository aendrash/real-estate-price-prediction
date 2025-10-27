import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def train_lgbm(df: pd.DataFrame, target_col='price'):
    X = df.select_dtypes(include=np.number).drop(columns=[target_col])
    y = np.log1p(df[target_col])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LGBMRegressor(
        objective='regression',
        metric='rmse',
        learning_rate=0.05,
        num_leaves=31,
        n_estimators=1000,
        random_state=42
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='rmse')
    joblib.dump(model, "data/processed/lgbm_model.pkl")

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"LightGBM RMSE: {rmse:.2f}, R²: {r2:.2f}")

    return model, X_test, y_test, preds
