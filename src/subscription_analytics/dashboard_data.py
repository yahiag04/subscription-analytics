from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

MART_TABLES = (
    "dim_users",
    "fact_events",
    "fact_subscriptions",
    "fact_acquisition_daily",
    "fact_experiment_assignments",
    "dim_date",
    "fact_revenue_daily",
)


def normalize_date(value: date | datetime) -> date:
    """Normalize DuckDB DATE and TIMESTAMP results to a Python date."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    raise TypeError(f"Expected date or datetime, got {type(value).__name__}")


def series_frame(series: pd.Series, value_name: str = "value") -> pd.DataFrame:
    """Convert a daily Series to Plotly-friendly `date` and value columns."""
    return series.rename(value_name).rename_axis("date").reset_index()


def warehouse_status(db_path: Path) -> str | None:
    """Return an actionable message if the requested DuckDB warehouse is unusable."""
    db_path = Path(db_path).expanduser()
    if not db_path.is_file():
        return f"Warehouse not found at {db_path}. Run `python -m subscription_analytics.cli build` first."
    try:
        with duckdb.connect(str(db_path), read_only=True) as con:
            for table in MART_TABLES:
                con.execute(f"SELECT 1 FROM mart.{table} LIMIT 1")
    except (duckdb.Error, OSError) as exc:
        return f"Warehouse is missing required marts or cannot be read ({exc}). Rebuild it with the documented build command."
    return None


def ensure_demo_warehouse(db_path: Path) -> None:
    """Create the reproducible demo warehouse when running from a clean checkout."""
    db_path = Path(db_path).expanduser()
    if db_path.is_file():
        return

    from .config import DEFAULT_DAYS, DEFAULT_SEED, DEFAULT_USERS
    from .generate import generate_dataset
    from .warehouse import build_warehouse

    raw_dir = db_path.parent / "raw"
    generate_dataset(raw_dir, seed=DEFAULT_SEED, n_users=DEFAULT_USERS, days=DEFAULT_DAYS)
    build_warehouse(raw_dir, db_path)


def _filtered_users_query(channel: str | None, country: str | None, device: str | None) -> tuple[str, list[Any]]:
    clauses = ["1=1"]
    params: list[Any] = []
    for column, value in (("acquisition_channel", channel), ("country", country), ("device", device)):
        if value and value != "All":
            clauses.append(f"{column} = ?")
            params.append(value)
    return " AND ".join(clauses), params


def build_filtered_connection(
    db_path: Path,
    channel: str | None = None,
    country: str | None = None,
    device: str | None = None,
) -> duckdb.DuckDBPyConnection:
    """Create filter-aware views over a read-only attached warehouse."""
    problem = warehouse_status(db_path)
    if problem:
        raise ValueError(problem)
    where, params = _filtered_users_query(channel, country, device)
    # Filter values are bound parameters in the user view; the attached path is escaped.
    escaped_path = str(Path(db_path).expanduser().resolve()).replace("'", "''")
    filtered = duckdb.connect(":memory:")
    try:
        filtered.execute(f"ATTACH '{escaped_path}' AS warehouse (READ_ONLY)")
        filtered.execute("CREATE SCHEMA mart")
        filtered.execute(
            f"CREATE TEMP TABLE _selected_users AS SELECT * FROM warehouse.mart.dim_users WHERE {where}",
            params,
        )
        filtered.execute("CREATE VIEW mart.dim_users AS SELECT * FROM _selected_users")
        filtered.execute("""
            CREATE VIEW mart.fact_events AS
            SELECT e.* FROM warehouse.mart.fact_events e
            JOIN mart.dim_users u USING (user_id)
        """)
        filtered.execute("""
            CREATE VIEW mart.fact_subscriptions AS
            SELECT s.* FROM warehouse.mart.fact_subscriptions s
            JOIN mart.dim_users u USING (user_id)
        """)
        filtered.execute("""
            CREATE VIEW mart.fact_experiment_assignments AS
            SELECT a.* FROM warehouse.mart.fact_experiment_assignments a
            JOIN mart.dim_users u USING (user_id)
        """)
        filtered.execute("""
            CREATE VIEW mart.fact_revenue_daily AS
            SELECT r.* FROM warehouse.mart.fact_revenue_daily r
            JOIN mart.dim_users u USING (user_id)
        """)
        filtered.execute("""
            CREATE VIEW mart.dim_date AS SELECT * FROM warehouse.mart.dim_date
        """)
        filtered.execute(f"""
            CREATE VIEW mart.fact_acquisition_daily AS
            WITH all_users AS (
                SELECT signup_date AS spend_date, acquisition_channel, count(*) AS all_users
                FROM warehouse.mart.dim_users GROUP BY signup_date, acquisition_channel
            ), selected_users AS (
                SELECT signup_date AS spend_date, acquisition_channel, count(*) AS selected_users
                FROM _selected_users
                GROUP BY signup_date, acquisition_channel
            )
            SELECT a.spend_date, a.acquisition_channel,
                   coalesce(s.selected_users, 0)::INTEGER AS new_users,
                   CASE WHEN t.all_users > 0
                        THEN a.spend * coalesce(s.selected_users, 0) / t.all_users
                        ELSE 0 END AS spend
            FROM warehouse.mart.fact_acquisition_daily a
            JOIN all_users t USING (spend_date, acquisition_channel)
            LEFT JOIN selected_users s USING (spend_date, acquisition_channel)
            WHERE coalesce(s.selected_users, 0) > 0
        """)
        return filtered
    except Exception:
        filtered.close()
        raise


def dashboard_options(db_path: Path) -> dict[str, list[str]]:
    with duckdb.connect(str(db_path), read_only=True) as con:
        return {
            "channel": ["All", *con.execute("SELECT DISTINCT acquisition_channel FROM mart.dim_users ORDER BY 1").fetchnumpy()["acquisition_channel"].tolist()],
            "country": ["All", *con.execute("SELECT DISTINCT country FROM mart.dim_users ORDER BY 1").fetchnumpy()["country"].tolist()],
            "device": ["All", *con.execute("SELECT DISTINCT device FROM mart.dim_users ORDER BY 1").fetchnumpy()["device"].tolist()],
        }


def full_quality_results(db_path: Path) -> list[dict[str, object]]:
    from .quality import run_quality_checks

    with duckdb.connect(str(db_path), read_only=True) as con:
        return run_quality_checks(con)
