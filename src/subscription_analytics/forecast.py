from __future__ import annotations

from collections.abc import Iterable

import duckdb
import numpy as np
import pandas as pd

METHODS = ("seasonal_naive", "moving_average", "linear_trend")


def _validate_history(history: pd.Series) -> pd.Series:
    if not isinstance(history.index, pd.DatetimeIndex):
        raise ValueError("history must have a DatetimeIndex")
    if history.empty:
        raise ValueError("history must not be empty")
    if history.index.has_duplicates or not history.index.is_monotonic_increasing:
        raise ValueError("history dates must be unique and sorted")
    if history.isna().any() or not np.isfinite(history.to_numpy(dtype=float)).all():
        raise ValueError("history must contain finite, non-missing values")
    expected = pd.date_range(history.index[0], periods=len(history), freq="D")
    actual_days = history.index.to_numpy(dtype="datetime64[D]")
    expected_days = expected.to_numpy(dtype="datetime64[D]")
    if not np.array_equal(actual_days, expected_days):
        raise ValueError("history must contain a complete daily index")
    return history.astype(float)


def forecast_series(history: pd.Series, horizon: int, method: str) -> pd.DataFrame:
    history = _validate_history(history)
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if method not in METHODS:
        raise ValueError(f"method must be one of {', '.join(METHODS)}")
    values = history.to_numpy()
    if method == "seasonal_naive":
        if len(values) >= 14:
            forecast = np.resize(values[-7:], horizon)
        else:
            forecast = np.repeat(values[-1], horizon)
    elif method == "moving_average":
        forecast = np.repeat(values[-min(28, len(values)):].mean(), horizon)
    else:
        if len(values) < 2:
            raise ValueError("linear_trend needs at least two history values")
        slope, intercept = np.polyfit(np.arange(len(values)), values, 1)
        forecast = intercept + slope * np.arange(len(values), len(values) + horizon)
    forecast = np.maximum(forecast, 0.0)
    dates = pd.date_range(history.index[-1] + pd.Timedelta(1, unit="D"), periods=horizon, freq="D")
    return pd.DataFrame({"date": dates, "forecast": forecast, "method": method})


def compare_forecasts(
    history: pd.Series,
    horizons: tuple[int, ...] = (30, 90),
) -> pd.DataFrame:
    history = _validate_history(history)
    if not horizons or any(h < 1 for h in horizons):
        raise ValueError("horizons must contain positive integers")
    results: list[dict[str, float | int | str | None]] = []
    for horizon in horizons:
        if len(history) <= horizon + 1:
            raise ValueError(f"history needs at least horizon + 2 observations for horizon {horizon}")
        errors: dict[str, list[float]] = {method: [] for method in METHODS}
        actuals: list[float] = []
        # Expanding-window evaluation: fit only on observations before each holdout.
        origin = len(history) - horizon
        while origin >= max(2, horizon):
            train = history.iloc[:origin]
            actual = history.iloc[origin:origin + horizon].to_numpy(dtype=float)
            for method in METHODS:
                predicted = forecast_series(train, horizon, method)["forecast"].to_numpy()
                errors[method].extend(np.abs(actual - predicted).tolist())
            actuals.extend(actual.tolist())
            origin -= horizon
        for method in METHODS:
            method_errors = np.asarray(errors[method], dtype=float)
            absolute_actuals = np.asarray(actuals, dtype=float)
            nonzero = absolute_actuals != 0
            mape = float(np.mean(method_errors[nonzero] / np.abs(absolute_actuals[nonzero])) * 100) if nonzero.any() else None
            results.append({
                "method": method,
                "horizon": horizon,
                "mae": float(method_errors.mean()),
                "mape": mape,
            })
    return pd.DataFrame(results)


def warehouse_series(con: duckdb.DuckDBPyConnection, metric: str) -> pd.Series:
    """Read a complete daily series from the mart; gaps in revenue become zero."""
    if metric == "daily_revenue":
        query = """
            SELECT d.date, coalesce(sum(r.revenue), 0)::DOUBLE AS value
            FROM mart.dim_date d
            LEFT JOIN mart.fact_revenue_daily r ON r.invoice_date = d.date
            GROUP BY d.date ORDER BY d.date
        """
    elif metric == "paid_subscribers":
        query = """
            SELECT d.date,
                   count(DISTINCT s.user_id) FILTER (
                       WHERE s.paid_start_at IS NOT NULL
                         AND CAST(s.paid_start_at AS DATE) <= d.date
                         AND (s.canceled_at IS NULL OR CAST(s.canceled_at AS DATE) > d.date)
                   )::DOUBLE AS value
            FROM mart.dim_date d CROSS JOIN mart.fact_subscriptions s
            GROUP BY d.date ORDER BY d.date
        """
    else:
        raise ValueError("metric must be 'daily_revenue' or 'paid_subscribers'")
    frame = con.execute(query).df()
    if frame.empty:
        raise ValueError("warehouse has no dates to forecast")
    index = pd.DatetimeIndex(frame.pop("date"), freq="D")
    return pd.Series(frame["value"].to_numpy(dtype=float), index=index, name=metric)
