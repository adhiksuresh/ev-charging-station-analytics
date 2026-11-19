"""Hour-ahead forecasting of charging-station energy.

Models are compared against two simple baselines on a chronological
train/test split:

* persistence     - next hour = this hour
* seasonal naive  - next hour = same hour yesterday
* XGBoost         - gradient-boosted trees on calendar, lag and rolling features

All rolling features are computed on values *up to the previous hour*
(``shift(1)`` before ``rolling``), so the model never sees the value it is
trying to predict.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURES = [
    "hour",
    "dayofweek",
    "weekend",
    "lag_1",
    "lag_2",
    "lag_24",
    "lag_48",
    "roll_mean_3",
    "roll_mean_24",
]


def hourly_series(energy: pd.DataFrame) -> pd.Series:
    """Resample interval energy to hourly totals (kWh)."""
    return energy["energy_kwh"].resample("h").sum().rename("energy_kwh")


def make_features(hourly: pd.Series) -> pd.DataFrame:
    """Calendar, lag and leakage-free rolling features for each hour."""
    df = hourly.to_frame("y")
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["weekend"] = (df.index.dayofweek >= 5).astype(int)
    for lag in (1, 2, 24, 48):
        df[f"lag_{lag}"] = df["y"].shift(lag)
    past = df["y"].shift(1)
    df["roll_mean_3"] = past.rolling(3).mean()
    df["roll_mean_24"] = past.rolling(24).mean()
    return df.dropna()


def metrics(y_true, y_pred) -> dict:
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
    }


def evaluate(hourly: pd.Series, test_fraction: float = 0.2, seed: int = 42):
    """Train XGBoost and baselines; return (metrics table, test predictions)."""
    from xgboost import XGBRegressor

    df = make_features(hourly)
    split = int(len(df) * (1 - test_fraction))
    train, test = df.iloc[:split], df.iloc[split:]

    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
    )
    model.fit(train[FEATURES], train["y"])

    preds = pd.DataFrame(
        {
            "actual": test["y"],
            "persistence": test["lag_1"],
            "seasonal_naive": test["lag_24"],
            "xgboost": model.predict(test[FEATURES]),
        },
        index=test.index,
    )
    table = pd.DataFrame(
        {name: metrics(preds["actual"], preds[name]) for name in ["persistence", "seasonal_naive", "xgboost"]}
    ).T
    table.index.name = "model"
    return table, preds
