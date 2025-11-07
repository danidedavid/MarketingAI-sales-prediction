import math
import numpy as np
import pandas as pd
import pytest

from src import data as data_mod
from src import features as feats_mod
from src import models as models_mod

# -----------------------
# UNIT TESTS: data.py
# -----------------------

def test_downcast_and_month_dtype(sample_df):
    df = sample_df.copy()
    # Make sales/int-like to test downcast function
    df["aux_int"] = (df["sales"] * 10).astype(int)
    mem_before = data_mod.mem_mb(df)
    data_mod.downcast_numeric(df)
    mem_after = data_mod.mem_mb(df)
    assert mem_after <= mem_before + 0.01  # no blow-up
    # ensure_month_dtype keeps month-level type
    df2 = df.copy()
    df2["year_month"] = df2["year_month"].astype("datetime64[ns]")
    df2 = data_mod.ensure_month_dtype(df2, "year_month")
    assert str(df2["year_month"].dtype) == "datetime64[M]"

def test_limit_to_last_n_months(sample_df):
    df = sample_df.copy()
    out = data_mod.limit_to_last_n_months(df, "year_month", 2)
    months = sorted(out["year_month"].dt.to_period("M").unique())
    assert len(months) == 2
    assert str(months[0]) == "2022-03" and str(months[1]) == "2022-04"


# -----------------------
# UNIT TESTS: features.py
# -----------------------

def test_calendar_features(sample_df):
    df = sample_df.copy()
    feats_mod.add_calendar_features_inplace(df, "year_month")
    for col in ["year", "month", "quarter", "month_sin", "month_cos"]:
        assert col in df.columns
    assert df["month"].between(1, 12).all()
    assert np.isfinite(df["month_sin"]).all() and np.isfinite(df["month_cos"]).all()

def test_group_lag1(sample_df):
    df = sample_df.copy()
    feats_mod.add_group_lag1_inplace(df, keys=["store","item"], date_col="year_month", target="sales")
    assert "lag1" in df.columns
    assert df["lag1"].isna().sum() == 0
    assert np.isfinite(df["lag1"]).all()

def test_feature_list(sample_df):
    df = sample_df.copy()
    feats_mod.add_calendar_features_inplace(df, "year_month")
    feats = feats_mod.feature_list(df, "year_month", "sales")
    assert "sales" not in feats and "year_month" not in feats
    assert "year" in feats and "month" in feats


# -----------------------
# UNIT TESTS: models.py
# -----------------------

def _prep_Xy(df):
    feats = feats_mod.feature_list(df, "year_month", "sales")
    cat_cols = list(df[feats].select_dtypes(include=["category","object","string"]).columns)
    num_cols = [c for c in feats if c not in cat_cols]
    return feats, cat_cols, num_cols

def test_build_and_fit_ridge(sample_df):
    df = sample_df.copy()
    feats_mod.add_calendar_features_inplace(df, "year_month")
    feats_mod.add_group_lag1_inplace(df, ["store","item"], "year_month", "sales")
    feats, cats, nums = _prep_Xy(df)
    (Xtr, ytr, Xva, yva), _ = models_mod.temporal_last_month_split(df, "year_month", "sales", feats)
    ridge = models_mod.build_ridge(nums, cats)
    ridge.fit(Xtr, ytr)
    pred = ridge.predict(Xva)
    assert len(pred) == len(yva)
    assert math.isfinite(models_mod.rmse(yva, pred))

@pytest.mark.skipif(models_mod.LGBMRegressor is None, reason="lightgbm not installed")
def test_build_and_fit_lgbm(sample_df):
    df = sample_df.copy()
    feats_mod.add_calendar_features_inplace(df, "year_month")
    feats_mod.add_group_lag1_inplace(df, ["store","item"], "year_month", "sales")
    feats, cats, nums = _prep_Xy(df)
    (Xtr, ytr, Xva, yva), _ = models_mod.temporal_last_month_split(df, "year_month", "sales", feats)
    lgbm = models_mod.build_lgbm(cats, n_estimators=50, n_jobs=1)  # tiny for speed
    lgbm.fit(Xtr, ytr)
    pred = lgbm.predict(Xva)
    assert len(pred) == len(yva)
    assert math.isfinite(models_mod.rmse(yva, pred))


# -----------------------
# INTEGRATION TEST
# -----------------------

def test_integration_end_to_end(sample_df):
    """End-to-end: features -> split -> train both -> compare metrics -> predict one example."""
    df = sample_df.copy()
    feats_mod.add_calendar_features_inplace(df, "year_month")
    feats_mod.add_group_lag1_inplace(df, ["store","item"], "year_month", "sales")

    feats = feats_mod.feature_list(df, "year_month", "sales")
    cats = list(df[feats].select_dtypes(include=["category","object","string"]).columns)
    nums = [c for c in feats if c not in cats]

    (Xtr, ytr, Xva, yva), val_month = models_mod.temporal_last_month_split(df, "year_month", "sales", feats)

    ridge = models_mod.build_ridge(nums, cats).fit(Xtr, ytr)
    yhat_r = ridge.predict(Xva)
    rmse_r = models_mod.rmse(yva, yhat_r)

    if models_mod.LGBMRegressor is not None:
        lgbm = models_mod.build_lgbm(cats, n_estimators=60, n_jobs=1).fit(Xtr, ytr)
        yhat_l = lgbm.predict(Xva)
        rmse_l = models_mod.rmse(yva, yhat_l)
        assert math.isfinite(rmse_l)
    else:
        rmse_l = float("inf")

    assert math.isfinite(rmse_r)
    one = Xva.iloc[[0]]
    _ = ridge.predict(one)
