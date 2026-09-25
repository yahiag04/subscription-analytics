from __future__ import annotations

import os
import uuid
from pathlib import Path

import duckdb

RAW_SCHEMAS: dict[str, str] = {
    "users": "user_id VARCHAR, signup_date DATE, acquisition_channel VARCHAR, country VARCHAR, device VARCHAR",
    "events": "event_id VARCHAR, user_id VARCHAR, event_ts TIMESTAMP, event_type VARCHAR, feature_name VARCHAR",
    "subscriptions": "subscription_id VARCHAR, user_id VARCHAR, plan VARCHAR, trial_start_at TIMESTAMP, trial_end_at TIMESTAMP, paid_start_at TIMESTAMP, canceled_at TIMESTAMP, status VARCHAR, monthly_price DOUBLE",
    "acquisition": "spend_date DATE, acquisition_channel VARCHAR, new_users INTEGER, spend DOUBLE",
    "experiment_assignments": "assignment_id VARCHAR, user_id VARCHAR, variant VARCHAR, assigned_at TIMESTAMP, experiment_name VARCHAR",
}


def _copy_csv(connection: duckdb.DuckDBPyConnection, table: str, path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required raw input missing: {path}")
    escaped = str(path.resolve()).replace("'", "''")
    connection.execute(f"CREATE TABLE raw.{table} ({RAW_SCHEMAS[table]})")
    connection.execute(f"COPY raw.{table} FROM '{escaped}' (HEADER, DELIMITER ',', NULLSTR '')")


def build_warehouse(raw_dir: Path, db_path: Path) -> None:
    """Build raw, clean, and analytical marts into an atomic DuckDB file."""
    raw_dir, db_path = Path(raw_dir), Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = db_path.with_name(f".{db_path.name}.{uuid.uuid4().hex}.tmp")
    if temp_path.exists():
        temp_path.unlink()
    con = duckdb.connect(str(temp_path))
    try:
        con.execute("CREATE SCHEMA raw; CREATE SCHEMA clean; CREATE SCHEMA mart")
        for table in RAW_SCHEMAS:
            _copy_csv(con, table, raw_dir / f"{table}.csv")
            con.execute(f"CREATE TABLE clean.{table} AS SELECT DISTINCT * FROM raw.{table}")

        con.execute("""
            CREATE TABLE mart.dim_users AS
            SELECT u.user_id, u.signup_date, u.acquisition_channel, u.country, u.device
            FROM clean.users u
        """)
        con.execute("""
            CREATE TABLE mart.fact_events AS
            SELECT event_id, user_id, event_ts, CAST(event_ts AS DATE) AS event_date,
                   event_type, feature_name
            FROM clean.events
        """)
        con.execute("""
            CREATE TABLE mart.fact_subscriptions AS
            SELECT s.subscription_id, s.user_id, u.acquisition_channel, u.country, u.device,
                   s.plan, s.trial_start_at, s.trial_end_at, s.paid_start_at,
                   s.canceled_at, s.status, s.monthly_price
            FROM clean.subscriptions s LEFT JOIN clean.users u USING (user_id)
        """)
        con.execute("""
            CREATE TABLE mart.fact_acquisition_daily AS
            SELECT spend_date, acquisition_channel, new_users, spend
            FROM clean.acquisition
        """)
        con.execute("""
            CREATE TABLE mart.fact_experiment_assignments AS
            SELECT a.assignment_id, a.user_id, a.variant, a.assigned_at, a.experiment_name,
                   u.acquisition_channel, u.country, u.device
            FROM clean.experiment_assignments a LEFT JOIN clean.users u USING (user_id)
        """)
        con.execute("""
            CREATE TABLE mart.dim_date AS
            SELECT CAST(day AS DATE) AS date
            FROM generate_series(
                (SELECT min(spend_date) FROM clean.acquisition),
                (SELECT max(spend_date) FROM clean.acquisition),
                INTERVAL 1 DAY
            ) AS days(day)
        """)
        con.execute("""
            CREATE TABLE mart.fact_revenue_daily AS
            WITH date_bounds AS (
                SELECT max(spend_date) AS max_date FROM clean.acquisition
            ), invoices AS (
                SELECT s.subscription_id, s.user_id, s.paid_start_at::DATE AS paid_date,
                       s.canceled_at::DATE AS canceled_date, s.monthly_price,
                       CAST(s.paid_start_at + month_offset * INTERVAL '1 month' AS DATE) AS invoice_date
                FROM clean.subscriptions s
                CROSS JOIN date_bounds b
                CROSS JOIN LATERAL range(
                    0,
                    date_diff('month', s.paid_start_at::DATE,
                              least(coalesce(s.canceled_at::DATE, b.max_date), b.max_date)) + 1
                ) AS offsets(month_offset)
                WHERE s.paid_start_at IS NOT NULL
            )
            SELECT subscription_id || ':' || CAST(invoice_date AS VARCHAR) AS invoice_id,
                   subscription_id, user_id, invoice_date, monthly_price AS revenue
            FROM invoices
            WHERE invoice_date <= (SELECT max_date FROM date_bounds)
              AND (canceled_date IS NULL OR invoice_date < canceled_date)
        """)
        con.execute("CHECKPOINT")
        con.close()
        os.replace(temp_path, db_path)
    except Exception:
        con.close()
        if temp_path.exists():
            temp_path.unlink()
        raise
