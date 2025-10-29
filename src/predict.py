"""Load saved model and provide a simple predict function."""
import os
import joblib
import pandas as pd
from typing import Dict, Any


def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    obj = joblib.load(path)
    return obj.get("model"), obj.get("scaler")


def predict_from_reading(model, scaler, reading: Dict[str, Any]):
    """reading: dict with keys hour, latitude, longitude, city (optional)

    Returns predicted pollutant value (float).
    """
    import pandas as pd
    row = pd.DataFrame([reading])
    # ensure same columns: hour, latitude, longitude, and city dummies may be missing
    if "city" in row.columns:
        row["city"] = row["city"].fillna("unknown")
    # do a minimal preprocessing similar to training
    X = row[[c for c in ["hour", "latitude", "longitude"] if c in row.columns]].copy()
    if "city" in row.columns:
        X = pd.concat([X, pd.get_dummies(row["city"]).add_prefix("city_")], axis=1)
    # scale numeric if scaler available
    if scaler is not None:
        num_cols = [c for c in X.columns if X[c].dtype.kind in "biufc"]
        if num_cols:
            X[num_cols] = scaler.transform(X[num_cols])

    pred = model.predict(X)
    return float(pred[0])
