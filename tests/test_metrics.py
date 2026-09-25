from datetime import date

import duckdb
import pytest
import pandas as pd

from subscription_analytics.metrics import (
    churn_rate,
    daily_active_users,
    retention_cohorts,
    trial_conversion,
    unit_economics,
)


@pytest.fixture
def metric_db():
    con = duckdb.connect(":memory:")
    con.execute("CREATE SCHEMA mart")
    con.execute("""
        CREATE TABLE mart.fact_subscriptions(
          subscription_id VARCHAR, user_id VARCHAR, trial_start_at TIMESTAMP,
          trial_end_at TIMESTAMP, paid_start_at TIMESTAMP, canceled_at TIMESTAMP,
          monthly_price DOUBLE, status VARCHAR
        )
    """)
    con.execute("""
        INSERT INTO mart.fact_subscriptions VALUES
          ('s1','u1','2026-01-01','2026-01-08','2026-01-08',NULL,9.0,'active'),
          ('s2','u2','2026-01-10','2026-01-17',NULL,NULL,9.0,'trial'),
          ('s3','u3','2026-01-29','2026-02-05',NULL,NULL,9.0,'trial'),
          ('s4','u4',NULL,NULL,'2026-01-15','2026-01-25',9.0,'canceled')
    """)
    con.execute("CREATE TABLE mart.fact_events(event_id VARCHAR, user_id VARCHAR, event_ts TIMESTAMP, event_date DATE)")
    con.execute("INSERT INTO mart.fact_events VALUES ('e1','u1','2026-01-08','2026-01-08'), ('e2','u1','2026-01-10','2026-01-10'), ('e3','u2','2026-01-17','2026-01-17')")
    con.execute("CREATE TABLE mart.dim_users(user_id VARCHAR, signup_date DATE)")
    con.execute("INSERT INTO mart.dim_users VALUES ('u1','2026-01-01'), ('u2','2026-01-10'), ('u3','2026-01-29')")
    con.execute("CREATE TABLE mart.dim_date(date DATE)")
    con.execute("INSERT INTO mart.dim_date SELECT day::DATE FROM generate_series('2026-01-01'::DATE, '2026-02-01'::DATE, INTERVAL 1 DAY) AS t(day)")
    con.execute("CREATE TABLE mart.fact_revenue_daily(invoice_id VARCHAR, subscription_id VARCHAR, user_id VARCHAR, invoice_date DATE, revenue DOUBLE)")
    con.execute("INSERT INTO mart.fact_revenue_daily VALUES ('i1','s1','u1','2026-01-08',9), ('i2','s2','u2','2026-01-17',9)")
    con.execute("CREATE TABLE mart.fact_acquisition_daily(spend_date DATE, acquisition_channel VARCHAR, new_users INTEGER, spend DOUBLE)")
    con.execute("INSERT INTO mart.fact_acquisition_daily VALUES ('2026-01-08','organic',1,2), ('2026-01-17','search',1,4)")
    return con


def test_trial_conversion_excludes_immature_trials(metric_db):
    result = trial_conversion(metric_db, as_of=date(2026, 2, 1), maturity_days=7)
    assert result["eligible_trials"] == 2
    assert result["converted_trials"] == 1
    assert result["rate"] == 0.5


def test_no_eligible_trials_returns_null_rate(metric_db):
    result = trial_conversion(metric_db, as_of=date(2026, 1, 7), maturity_days=7)
    assert result["eligible_trials"] == 0
    assert result["rate"] is None


def test_churn_uses_active_start_denominator(metric_db):
    result = churn_rate(metric_db, date(2026, 1, 8), date(2026, 1, 31))
    assert result["active_at_start"] == 1
    assert result["canceled"] == 0
    assert result["rate"] == 0.0


def test_churn_excludes_cancellations_from_users_added_after_period_start(metric_db):
    result = churn_rate(metric_db, date(2026, 1, 8), date(2026, 1, 31))
    assert result["active_at_start"] == 1
    assert result["canceled"] == 0
    assert result["rate"] == 0.0


def test_activity_windows_are_distinct_users(metric_db):
    result = daily_active_users(metric_db)
    jan_10 = result.loc[result.date == pd.Timestamp("2026-01-10")].iloc[0]
    jan_17 = result.loc[result.date == pd.Timestamp("2026-01-17")].iloc[0]
    assert jan_10.dau == 1
    assert jan_10.wau == 1
    assert jan_17.dau == 1
    assert jan_17.wau == 1
    assert jan_17.mau == 2


def test_cohort_retention_and_unit_economics_are_available(metric_db):
    cohorts = retention_cohorts(metric_db, periods=2)
    economics = unit_economics(metric_db, as_of=date(2026, 2, 1), horizon_days=30)
    assert {"cohort_month", "period_number", "retention_rate"}.issubset(cohorts.columns)
    assert economics["arpu"] is not None
    assert economics["cac"] is not None
