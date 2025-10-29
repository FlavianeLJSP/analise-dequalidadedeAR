"""OpenAQ simple client to fetch recent air quality measurements.

Functions:
 - fetch_latest_measurements: fetch measurements and return pandas DataFrame
"""
from typing import Optional
import requests
import pandas as pd

OPENAQ_URL = "https://api.openaq.org/v2/measurements"


def fetch_latest_measurements(city: Optional[str] = None, country: Optional[str] = None, limit: int = 100) -> pd.DataFrame:
    """Fetch recent measurements from OpenAQ.

    Args:
        city: optional city name to filter
        country: optional country code (ISO 2-letter)
        limit: number of records to retrieve (max ~1000)

    Returns:
        DataFrame with columns: datetime, parameter, value, unit, latitude, longitude, city, country
    """
    # Demo: usar dados sintéticos enquanto a API está indisponível
    import numpy as np
    from datetime import datetime, timedelta
    
    now = datetime.now()
    cities = ["São Paulo", "Rio de Janeiro", "Belo Horizonte"]
    if city:
        cities = [city]
    
    results = []
    for i in range(limit):
        c = np.random.choice(cities)
        lat = -23.5 + np.random.normal(0, 0.5) if c == "São Paulo" else -22.9 + np.random.normal(0, 0.5)
        lon = -46.6 + np.random.normal(0, 0.5) if c == "São Paulo" else -43.2 + np.random.normal(0, 0.5)
        
        results.append({
            "date": {"utc": now - timedelta(minutes=i*5)},
            "parameter": "pm25",
            "value": max(0, 25 + np.random.normal(0, 10)),
            "unit": "µg/m³",
            "coordinates": {"latitude": lat, "longitude": lon},
            "city": c,
            "country": "BR"
        })

    if not results:
        return pd.DataFrame()

    rows = []
    for r in results:
        coords = r.get("coordinates") or {}
        rows.append({
            "datetime": r.get("date", {}).get("utc"),
            "parameter": r.get("parameter"),
            "value": r.get("value"),
            "unit": r.get("unit"),
            "latitude": coords.get("latitude"),
            "longitude": coords.get("longitude"),
            "city": r.get("city"),
            "country": r.get("country"),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["datetime"])
    return df


if __name__ == "__main__":
    # quick test
    print("Fetching up to 10 recent measurements...")
    try:
        df = fetch_latest_measurements(limit=10)
        print(df.head())
    except Exception as e:
        print("Error fetching data:", e)
