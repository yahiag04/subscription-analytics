import duckdb
import pandas as pd
import pytest

from subscription_analytics.forecast import compare_forecasts, forecast_series, warehouse_series


def test_linear_trend_extends_known_slope():
    history = pd.Series([10.0, 12.0, 14.0, 16.0], index=pd.date_range("2026-01-01", periods=4))
    result = forecast_series(history, horizon=2, method="linear_trend")
    assert result["forecast"].tolist() == pytest.approx([18.0, 20.0])


def test_backtest_rejects_history_shorter_than_horizon():
    with pytest.raises(ValueError, match="history"):
        compare_forecasts(pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2026-01-01", periods=3)), horizons=(30,))


def test_forecast_rejects_gaps_and_unknown_methods():
    gap = pd.Series([1.0, 2.0], index=pd.to_datetime(["2026-01-01", "2026-01-03"]))
    with pytest.raises(ValueError, match="daily"):
        forecast_series(gap, horizon=1, method="linear_trend")
    with pytest.raises(ValueError, match="method"):
        forecast_series(pd.Series([1.0], index=pd.date_range("2026-01-01", periods=1)), 1, "xgboost")


def test_temporal_comparison_returns_all_models_for_each_horizon():
    history = pd.Series([float(x + x % 7) for x in range(180)], index=pd.date_range("2026-01-01", periods=180))
    result = compare_forecasts(history, horizons=(30, 90))
    assert set(result.method) == {"seasonal_naive", "moving_average", "linear_trend"}
    assert set(result.horizon) == {30, 90}
    assert result.mae.notna().all()


def test_warehouse_series_fills_calendar_gaps_with_zero():
    con = duckdb.connect(":memory:")
    con.execute("CREATE SCHEMA mart")
    con.execute("CREATE TABLE mart.dim_date(date DATE)")
    con.execute("INSERT INTO mart.dim_date VALUES ('2026-01-01'), ('2026-01-02'), ('2026-01-03')")
    con.execute("CREATE TABLE mart.fact_revenue_daily(invoice_date DATE, revenue DOUBLE)")
    con.execute("INSERT INTO mart.fact_revenue_daily VALUES ('2026-01-01', 10), ('2026-01-03', 5)")
    series = warehouse_series(con, "daily_revenue")
    assert series.tolist() == [10.0, 0.0, 5.0]
    assert series.index.freq == pd.offsets.Day()
    assert len(forecast_series(series, horizon=1, method="linear_trend")) == 1
