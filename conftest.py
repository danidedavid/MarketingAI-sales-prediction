import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Ensure project root (parent of tests/) is on sys.path so `src` can be imported
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

@pytest.fixture
def sample_df():
    """Small monthly dataset across two stores and two items with a simple pattern."""
    dates = pd.period_range("2022-01", "2022-04", freq="M").to_timestamp()
    rows = []
    for store in ["S1", "S2"]:
        for item in ["I1", "I2"]:
            for i, d in enumerate(dates, start=1):
                rows.append({
                    "year_month": d, "store": store, "item": item,
                    "sales": float(10 * i + (0 if store=="S1" else 5) + (0 if item=="I1" else 2)),
                    "mean_price": float(3.0 + 0.1 * i),
                    "region": "RJ", "category": "catA", "department": "dep1", "store_code": store,
                    "cluster": "0"
                })
    df = pd.DataFrame(rows)
    # Normalize to month type expected
    df["year_month"] = df["year_month"].values.astype("datetime64[M]")
    # Cast some categoricals to simulate real input
    for c in ["region", "category", "department", "store_code", "cluster", "store", "item"]:
        df[c] = df[c].astype("category")
    return df
