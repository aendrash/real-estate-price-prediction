from src.preprocessing import preprocess_pipeline
from src.train import train_lgbm
from src.evaluate import evaluate_model
from src.utils import load_data

def run_full_pipeline(raw_csv_path):
    print("🔹 Starting preprocessing...")
    df = preprocess_pipeline(raw_csv_path)

    print("🔹 Training model...")
    model, X_test, y_test, preds = train_lgbm(df)

    print("🔹 Evaluating...")
    metrics = evaluate_model(y_test, preds)
    print(f"Final Metrics: {metrics}")

    return model, metrics
