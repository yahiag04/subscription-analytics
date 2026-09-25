from subscription_analytics.dashboard_data import build_filtered_connection, warehouse_status
from subscription_analytics.generate import generate_dataset
from subscription_analytics.warehouse import build_warehouse


def test_missing_or_corrupt_warehouse_has_recovery_message(tmp_path):
    missing = warehouse_status(tmp_path / "missing.duckdb")
    assert missing and "build" in missing.lower()
    corrupt = tmp_path / "corrupt.duckdb"
    corrupt.write_text("not a duckdb file")
    message = warehouse_status(corrupt)
    assert message and "build" in message.lower()


def test_dashboard_filters_user_marts_by_channel_country_device(tmp_path):
    raw = tmp_path / "raw"
    generate_dataset(raw, seed=23, n_users=300, days=40)
    db = tmp_path / "warehouse.duckdb"
    build_warehouse(raw, db)
    connection = build_filtered_connection(db, "search", "DE", "ios")

    users = connection.execute("SELECT DISTINCT acquisition_channel, country, device FROM mart.dim_users").fetchall()
    assert all(row == ("search", "DE", "ios") for row in users)
    connection.close()


def test_normalize_date_accepts_duckdb_date_and_datetime():
    from datetime import date, datetime

    from subscription_analytics.dashboard_data import normalize_date

    assert normalize_date(date(2026, 1, 2)) == date(2026, 1, 2)
    assert normalize_date(datetime(2026, 1, 2, 12)) == date(2026, 1, 2)


def test_filtered_marts_preserve_date_types(tmp_path):
    raw = tmp_path / "raw"
    generate_dataset(raw, seed=24, n_users=30, days=30)
    db = tmp_path / "warehouse.duckdb"
    build_warehouse(raw, db)
    connection = build_filtered_connection(db)

    date_type = connection.execute("SELECT typeof(date) FROM mart.dim_date LIMIT 1").fetchone()[0]
    event_date_type = connection.execute("SELECT typeof(event_date) FROM mart.fact_events LIMIT 1").fetchone()[0]

    assert date_type == "DATE"
    assert event_date_type == "DATE"
    connection.close()


def test_series_frame_keeps_dates_as_a_column():
    import pandas as pd

    from subscription_analytics.dashboard_data import series_frame

    series = pd.Series([1.0, 2.0], index=pd.date_range("2026-01-01", periods=2), name="revenue")
    frame = series_frame(series, "Revenue")

    assert list(frame.columns) == ["date", "Revenue"]
    assert frame["Revenue"].tolist() == [1.0, 2.0]
