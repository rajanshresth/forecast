"""
Moving-average baseline forecaster + walk-forward backtest error metrics,
for comparison against the Bayesian model.

Why a *seasonal* moving average (not a plain one):
    Your admissions data has a strong, real seasonal pattern (e.g. Autumn
    running well above Spring). A plain moving average over raw
    chronological values would blend both seasons together and forecast
    the same number regardless of which season is next -- a strawman that
    would obviously lose to any seasonal-aware method. A seasonal moving
    average -- averaging only past values from the *same* season -- is the
    fair, standard baseline for a 2-season-per-year series like this one.
"""

from typing import List, Dict, Any, Optional, Callable
import numpy as np
import pandas as pd


def seasonal_moving_average_forecast(
    df: pd.DataFrame,
    future_periods: List[Dict[str, Any]],
    window: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    df: historical data with columns season_idx, admissions
    future_periods: list of {"t", "season_idx", "period_label"}
    window: how many past same-season values to average.
        None = use all available same-season history (expanding average).

    Multi-step-ahead forecasts are generated recursively: once a future
    period is forecast, its value is folded into the history used for
    later same-season periods (otherwise a 4-period-ahead forecast
    covering 2 cycles of the same season would use identical history
    for both).
    """
    history = df[["season_idx", "admissions"]].copy()
    forecasts = []
    for fp in future_periods:
        same_season = history.loc[
            history["season_idx"] == fp["season_idx"], "admissions"
        ]
        if window is not None:
            same_season = same_season.tail(window)
        pred = (
            float(same_season.mean())
            if len(same_season)
            else float(history["admissions"].mean())
        )
        forecasts.append(
            {
                "period_label": fp["period_label"],
                "t": fp["t"],
                "forecast": pred,
            }
        )
        history = pd.concat(
            [
                history,
                pd.DataFrame([{"season_idx": fp["season_idx"], "admissions": pred}]),
            ],
            ignore_index=True,
        )
    return forecasts


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    err = actual - predicted
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    # guard against divide-by-zero if any actual admissions value is 0
    nonzero = actual != 0
    mape = (
        float(np.mean(np.abs(err[nonzero] / actual[nonzero])) * 100)
        if nonzero.any()
        else None
    )
    return {"mae": mae, "rmse": rmse, "mape": mape}


def backtest(
    df: pd.DataFrame,
    point_forecast_fn: Callable[[pd.DataFrame, Dict[str, Any]], float],
    min_train: int = 3,
) -> Dict[str, Any]:
    """
    Walk-forward (expanding window) backtest: for each historical point
    after the first `min_train`, forecast it using only the data strictly
    before it, then compare to the actual value.

    point_forecast_fn(train_df, target_period_dict) -> float
        target_period_dict has the same shape as one entry of
        `future_periods` ({"t", "season_idx", "period_label"}).

    Returns error metrics plus per-point detail. With only a handful of
    total data points, there may only be 1-3 backtest points -- treat the
    resulting error numbers as indicative, not statistically solid, until
    you have more seasons logged.
    """
    df = df.sort_values("t").reset_index(drop=True)
    details = []
    for i in range(min_train, len(df)):
        train = df.iloc[:i]
        target_row = df.iloc[i]
        target_period = {
            "t": int(target_row["t"]),
            "season_idx": int(target_row["season_idx"]),
            "period_label": target_row["period_label"],
        }
        pred = point_forecast_fn(train, target_period)
        actual = float(target_row["admissions"])
        details.append(
            {
                "period_label": target_row["period_label"],
                "actual": actual,
                "predicted": pred,
                "error": actual - pred,
            }
        )

    if not details:
        return {"n_test": 0, "mae": None, "rmse": None, "mape": None, "details": []}

    actual_arr = np.array([d["actual"] for d in details])
    pred_arr = np.array([d["predicted"] for d in details])
    metrics = _metrics(actual_arr, pred_arr)
    metrics["n_test"] = len(details)
    metrics["details"] = details
    return metrics
