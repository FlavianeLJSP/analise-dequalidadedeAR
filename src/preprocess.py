"""Preprocessing utilities: cleaning, feature extraction, scaling.
"""
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib


def preprocess_for_model(df: pd.DataFrame, pollutant: str = "pm25") -> Tuple[pd.DataFrame, pd.Series, StandardScaler]:
    """Filter data for pollutant, create features and scale numeric columns.

    Returns X, y, scaler
    """
    # filter pollutant column names commonly 'pm25' or 'pm2_5' - OpenAQ uses 'pm25'
    df = df.copy()
    df = df[df["parameter"].str.lower() == pollutant.lower()]
    # drop missing value rows
    df = df.dropna(subset=["value", "latitude", "longitude", "datetime"]) 

    if df.empty:
        return pd.DataFrame(), pd.Series(dtype=float), None

    # feature: hour of day
    df["hour"] = df["datetime"].dt.hour

    # use city (as categorical), hour, lat, lon, and optionally other sensors
    X = df[["hour", "latitude", "longitude"]].copy()
    # encode city -> use simple label encoding by mapping
    X["city"] = df["city"].fillna("unknown")
    X = pd.get_dummies(X, columns=["city"], drop_first=True)

    y = df["value"].astype(float)

    scaler = StandardScaler()
    numeric_cols = [c for c in X.columns if X[c].dtype in [np.float64, np.int64]]
    if numeric_cols:
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    else:
        scaler = None

    return X, y, scaler


def save_scaler(scaler, path: str):
    joblib.dump(scaler, path)


def load_scaler(path: str):
    return joblib.load(path)
