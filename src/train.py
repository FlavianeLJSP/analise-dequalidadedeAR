"""Train a lightweight ML model to predict a pollutant (e.g., PM2.5).

Usage example:
    python -m src.train --pollutant pm25 --model decision_tree
"""
import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

from src import collector, preprocess


def train_model(pollutant: str = "pm25", model_type: str = "decision_tree", csv_path: str = None, out_dir: str = "models"):
    os.makedirs(out_dir, exist_ok=True)

    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=["datetime"])
        print(f"Loaded {len(df)} rows from CSV")
    else:
        print("No CSV provided or file missing; attempting to fetch recent data from OpenAQ...")
        df = collector.fetch_latest_measurements(limit=1000)

    X, y, scaler = preprocess.preprocess_for_model(df, pollutant=pollutant)
    if X.empty or y.empty:
        # fallback: synthetic tiny dataset to demonstrate pipeline
        print("No usable real data available; creating a small synthetic dataset as fallback.")
        import numpy as np
        rng = np.random.RandomState(0)
        n = 200
        X = pd.DataFrame({"hour": rng.randint(0, 24, n), "latitude": rng.uniform(-90, 90, n), "longitude": rng.uniform(-180, 180, n)})
        y = (np.sin(X["hour"]/24*2*3.1415) * 10 + rng.normal(0, 5, n)).astype(float)
        scaler = None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type == "decision_tree":
        model = DecisionTreeRegressor(max_depth=6, random_state=42)
    else:
        model = LinearRegression()

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    model_path = os.path.join(out_dir, f"air_quality_{pollutant}_{model_type}.pkl")
    joblib.dump({"model": model, "scaler": scaler, "columns": X_train.columns.tolist()}, model_path)

    print(f"Saved model to {model_path}")
    print(f"MSE: {mse:.3f}, R2: {r2:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pollutant", default="pm25")
    parser.add_argument("--model", choices=["decision_tree", "linear"], default="decision_tree")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--out", default="models")
    args = parser.parse_args()

    train_model(pollutant=args.pollutant, model_type=args.model, csv_path=args.csv, out_dir=args.out)


if __name__ == "__main__":
    main()
