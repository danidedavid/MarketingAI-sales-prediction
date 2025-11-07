# =============================
# FILE: src/data.py
# =============================
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional, Tuple
import pandas as pd
import numpy as np

__all__ = [
    "read_parquet_efficient",
    "read_csv_efficient",
    "ensure_month_dtype",
    "downcast_numeric",
    "mem_mb",
    "limit_to_last_n_months",
]

def mem_mb(df: pd.DataFrame) -> float:
    """Return DataFrame memory usage in MB (deep)."""
    try:
        return float(df.memory_usage(deep=True).sum() / (1024 ** 2))
    except Exception:
        return float("nan")


def downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast integer/float columns to reduce memory footprint in-place and return df."""
    for c in df.select_dtypes(include=["int", "Int64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    for c in df.select_dtypes(include=["float"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="float")
    return df


def ensure_month_dtype(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Ensure date_col exists and is normalized to month-granularity (datetime64[M])."""
    if date_col not in df.columns:
        raise KeyError(f"Coluna de data ausente: {date_col}")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[date_col] = df[date_col].values.astype("datetime64[M]")
    return df


def _categorify_reasonable(df: pd.DataFrame) -> pd.DataFrame:
    obj_cols = df.select_dtypes(include=["object", "string"]).columns
    for c in obj_cols:
        nunq = df[c].nunique(dropna=True)
        if nunq < 5000 and nunq / max(1, len(df)) < 0.5:
            df[c] = df[c].astype("category")
    if "cluster" in df.columns:
        df["cluster"] = df["cluster"].astype("category")
    return df


def read_parquet_efficient(path: str | Path, date_col: Optional[str] = None) -> pd.DataFrame:
    """Read Parquet with pyarrow, downcast numerics, optional month-normalized date, categorify."""
    df = pd.read_parquet(path, engine="pyarrow")
    downcast_numeric(df)
    if date_col:
        ensure_month_dtype(df, date_col)
    _categorify_reasonable(df)
    return df


def read_csv_efficient(path: str | Path, date_col: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    downcast_numeric(df)
    if date_col:
        ensure_month_dtype(df, date_col)
    _categorify_reasonable(df)
    return df


def limit_to_last_n_months(df: pd.DataFrame, date_col: str, n_months: int) -> pd.DataFrame:
    months = sorted(df[date_col].dt.to_period("M").unique())
    if n_months and len(months) > n_months:
        keep = set(months[-n_months:])
        return df[df[date_col].dt.to_period("M").isin(keep)].reset_index(drop=True)
    return df


# =============================
# FILE: src/features.py
# =============================
from __future__ import annotations
from typing import List, Sequence
import numpy as np
import pandas as pd

__all__ = [
    "add_calendar_features_inplace",
    "add_group_lag1_inplace",
    "add_group_rollings_inplace",
    "feature_list",
    "ensure_categories",
]

def add_calendar_features_inplace(df: pd.DataFrame, date_col: str) -> None:
    """Add year/month/quarter + cyclical encodings from a month-normalized date column."""
    df["year"] = df[date_col].dt.year.astype("int16")
    df["month"] = df[date_col].dt.month.astype("int8")
    df["quarter"] = df[date_col].dt.quarter.astype("int8")
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12).astype("float32")
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12).astype("float32")


def add_group_lag1_inplace(df: pd.DataFrame, keys: Sequence[str], date_col: str, target: str) -> None:
    """Compute lag1 per group(keys); fill NA with group median then global median."""
    df.sort_values([*keys, date_col], inplace=True, kind="stable")
    df["lag1"] = df.groupby(keys, observed=True)[target].shift(1).astype("float32")
    med = df.groupby(keys, observed=True)[target].transform("median")
    df["lag1"] = df["lag1"].fillna(med).fillna(df[target].median()).astype("float32")


def add_group_rollings_inplace(
    df: pd.DataFrame,
    keys: Sequence[str],
    date_col: str,
    target: str,
    windows: Sequence[int] = (3, 6),
) -> None:
    """Optionally compute rolling means (shifted by 1) per group for given windows."""
    df.sort_values([*keys, date_col], inplace=True, kind="stable")
    for w in windows:
        name = f"roll{w}"
        df[name] = (
            df.groupby(keys, observed=True)[target]
              .transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
              .astype("float32")
        )
        med = df.groupby(keys, observed=True)[target].transform("median")
        df[name] = df[name].fillna(med).fillna(df[target].median()).astype("float32")


def feature_list(df: pd.DataFrame, date_col: str, target: str) -> List[str]:
    return [c for c in df.columns if c not in {date_col, target}]


def ensure_categories(df: pd.DataFrame, categoricals: Sequence[str]) -> pd.DataFrame:
    X = df.copy()
    for c in categoricals:
        if c in X.columns:
            X[c] = X[c].astype("category")
    return X


# =============================
# FILE: src/models.py
# =============================
from __future__ import annotations
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, FunctionTransformer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
try:
    from lightgbm import LGBMRegressor
except Exception:  # make optional
    LGBMRegressor = None
from sklearn.metrics import mean_squared_error

__all__ = [
    "build_ridge",
    "build_random_forest",
    "build_lgbm",
    "rmse",
    "temporal_last_month_split",
    "evaluate_models_simple",
]

def _pre_num_cat(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(with_mean=False), num_cols),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
    ], remainder="drop", sparse_threshold=0.3)


def build_ridge(num_cols: List[str], cat_cols: List[str]) -> Pipeline:
    pre = _pre_num_cat(num_cols, cat_cols)
    return Pipeline([("pre", pre), ("est", Ridge(alpha=1.0, random_state=42))])


def build_random_forest(num_cols: List[str], cat_cols: List[str],
                        n_estimators: int = 150, max_depth: int | None = 20,
                        n_jobs: int = 2, seed: int = 42) -> Pipeline:
    pre = ColumnTransformer([
        ("num", "passthrough", num_cols),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
    ], remainder="drop", sparse_threshold=0.3)
    est = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=3,
        n_jobs=n_jobs,
        random_state=seed,
    )
    return Pipeline([("pre", pre), ("est", est)])


def build_lgbm(cat_cols: List[str], n_estimators: int = 400, n_jobs: int = 2, seed: int = 42) -> Pipeline:
    if LGBMRegressor is None:
        raise ImportError("lightgbm não instalado. Instale 'lightgbm' para usar build_lgbm().")

    def ensure_categories(X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in cat_cols:
            if c in X.columns:
                X[c] = X[c].astype("category")
        return X

    pre = FunctionTransformer(ensure_categories, validate=False)
    est = LGBMRegressor(
        objective="regression",
        learning_rate=0.1,
        n_estimators=n_estimators,
        num_leaves=31,
        min_child_samples=100,
        subsample=0.8,
        colsample_bytree=0.8,
        max_bin=255,
        n_jobs=n_jobs,
        random_state=seed,
    )
    return Pipeline([("pre", pre), ("est", est)])


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def temporal_last_month_split(df: pd.DataFrame, date_col: str, target_col: str, features: List[str]):
    months = sorted(df[date_col].dt.to_period("M").unique())
    val_month = months[-1]
    train_mask = df[date_col].dt.to_period("M") < val_month
    val_mask = df[date_col].dt.to_period("M") == val_month
    Xtr, ytr = df.loc[train_mask, features], df.loc[train_mask, target_col].astype("float32")
    Xva, yva = df.loc[val_mask, features], df.loc[val_mask, target_col].astype("float32")
    return (Xtr, ytr, Xva, yva), str(val_month)


def evaluate_models_simple(df: pd.DataFrame, date_col: str, target_col: str,
                           features: List[str], cat_cols: List[str], num_cols: List[str],
                           use_rf: bool = True, use_lgbm: bool = True) -> pd.DataFrame:
    """Train Ridge (+RF/LGBM opcional) com split temporal do último mês e retorna DataFrame de métricas."""
    (Xtr, ytr, Xva, yva), val_month = temporal_last_month_split(df, date_col, target_col, features)

    rows = []
    # Ridge
    ridge = build_ridge(num_cols, cat_cols)
    ridge.fit(Xtr, ytr)
    rows.append({"model": "Ridge", "rmse_val": rmse(yva, ridge.predict(Xva))})

    # RandomForest
    if use_rf:
        rf = build_random_forest(num_cols, cat_cols)
        rf.fit(Xtr, ytr)
        rows.append({"model": "RandomForest", "rmse_val": rmse(yva, rf.predict(Xva))})

    # LightGBM
    if use_lgbm and LGBMRegressor is not None:
        lgbm = build_lgbm(cat_cols)
        lgbm.fit(Xtr, ytr)
        rows.append({"model": "LightGBM", "rmse_val": rmse(yva, lgbm.predict(Xva))})

    return pd.DataFrame(rows).sort_values("rmse_val").reset_index(drop=True)


# =============================
# FILE: src/visualization.py
# =============================
from __future__ import annotations
from typing import Optional, Sequence
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

__all__ = [
    "plot_series_by_item_store",
    "plot_feature_importance_lgbm",
    "plot_residuals",
]

def plot_series_by_item_store(df: pd.DataFrame, date_col: str, target_col: str,
                              item: str, store: str, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Line plot do target ao longo do tempo para um par item×store."""
    subset = df[(df["item"] == item) & (df["store"] == store)].sort_values(date_col)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(subset[date_col].dt.to_timestamp(), subset[target_col].values, marker="o")
    ax.set_title(f"{target_col} – item={item} | store={store}")
    ax.set_xlabel("Mês")
    ax.set_ylabel(target_col)
    ax.grid(True, alpha=0.3)
    return ax


def plot_feature_importance_lgbm(model, feature_names: Sequence[str], top_k: int = 20,
                                 ax: Optional[plt.Axes] = None) -> Optional[plt.Axes]:
    """Plota importâncias se o estimador for LGBM.* com atributo feature_importances_."""
    try:
        est = getattr(model, "named_steps", {}).get("est", model)
        importances = getattr(est, "feature_importances_", None)
        if importances is None:
            return None
        idx = np.argsort(importances)[::-1][:top_k]
        names = np.array(feature_names)[idx]
        vals = np.array(importances)[idx]
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(range(len(names))[::-1], vals[idx][::-1])
        ax.set_yticks(range(len(names))[::-1])
        ax.set_yticklabels(names[::-1])
        ax.set_title("LightGBM – Importância de Features")
        ax.set_xlabel("Importância (gain)")
        ax.grid(True, axis="x", alpha=0.3)
        return ax
    except Exception:
        return None


def plot_residuals(y_true: pd.Series, y_pred: pd.Series, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Scatter de resíduos com linha y=0."""
    resid = np.asarray(y_true) - np.asarray(y_pred)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_pred, resid, s=10, alpha=0.6)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Predição")
    ax.set_ylabel("Resíduo (y - ŷ)")
    ax.set_title("Resíduos vs Predições")
    ax.grid(True, alpha=0.3)
    return ax
