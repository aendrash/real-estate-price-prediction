from src.pipeline import run_full_pipeline

if __name__ == "__main__":
    raw_data_path = "data/raw/ahmedabad.csv"
    model, metrics = run_full_pipeline(raw_data_path)
