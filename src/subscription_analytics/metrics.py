from __future__ import annotations

from datetime import date
from typing import Any

import duckdb
import numpy as np
import pandas as pd


def daily_active_users(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Return DAU and rolling distinct WAU/MAU without multiplying every event by 30."""
    dates = [row[0] for row in con.execute("SELECT date FROM mart.dim_date ORDER BY date").fetchall()]
    users = pd.Index(con.execute("SELECT user_id FROM mart.dim_users").fetchnumpy()["user_id"])
    if not dates:
        return pd.DataFrame(columns=["date", "dau", "wau", "mau"])

    first_date, last_date = dates[0], dates[-1]
    activity = con.execute("""
        SELECT DISTINCT user_id, event_date
        FROM mart.fact_events
        WHERE event_date BETWEEN ? - 29 AND ?
    """, [first_date, last_date]).df()
    width = len(dates) + 29
    active = np.zeros((len(users), width), dtype=bool)
    if not activity.empty and len(users):
        user_positions = users.get_indexer(activity["user_id"])
        day_positions = (pd.to_datetime(activity["event_date"]) - pd.Timestamp(first_date)).dt.days.to_numpy() + 29
        valid = user_positions >= 0
        active[user_positions[valid], day_positions[valid]] = True

    dau = active[:, 29:29 + len(dates)].sum(axis=0)
    wau = np.array([
        active[:, day + 23:day + 30].any(axis=1).sum() for day in range(len(dates))
    ])
    mau = np.array([
        active[:, day:day + 30].any(axis=1).sum() for day in range(len(dates))
    ])
    return pd.DataFrame({"date": pd.to_datetime(dates), "dau": dau, "wau": wau, "mau": mau})


def trial_conversion(
    con: duckdb.DuckDBPyConnection,
    as_of: date,
    maturity_days: int = 7,
) -> dict[str, float | int | None]:
    if maturity_days < 0:
        raise ValueError("maturity_days must be nonnegative")
    eligible, converted = con.execute("""
        SELECT count(*), count(*) FILTER (
            WHERE paid_start_at IS NOT NULL
              AND CAST(paid_start_at AS DATE) <= CAST(trial_start_at AS DATE) + ?
        )
        FROM mart.fact_subscriptions
        WHERE trial_start_at IS NOT NULL
          AND CAST(trial_start_at AS DATE) + ? <= ?
    """, [maturity_days, maturity_days, as_of]).fetchone()
    return {
        "eligible_trials": int(eligible),
        "converted_trials": int(converted),
        "rate": float(converted / eligible) if eligible else None,
    }


def churn_rate(
    con: duckdb.DuckDBPyConnection,
    start: date,
    end: date,
) -> dict[str, float | int | None]:
    if end < start:
        raise ValueError("end must be on or after start")
    active, canceled = con.execute("""
        SELECT count(*) FILTER (
                   WHERE CAST(paid_start_at AS DATE) <= ?
                     AND (canceled_at IS NULL OR CAST(canceled_at AS DATE) > ?)
               ),
               count(*) FILTER (
                   WHERE paid_start_at IS NOT NULL
                     AND CAST(paid_start_at AS DATE) <= ?
                     AND canceled_at IS NOT NULL
                     AND CAST(canceled_at AS DATE) > ?
                     AND CAST(canceled_at AS DATE) <= ?
               )
        FROM mart.fact_subscriptions
        WHERE paid_start_at IS NOT NULL
    """, [start, start, start, start, end]).fetchone()
    return {
        "active_at_start": int(active),
        "canceled": int(canceled),
        "rate": float(canceled / active) if active else None,
    }


def retention_cohorts(
    con: duckdb.DuckDBPyConnection,
    periods: int = 6,
) -> pd.DataFrame:
    if periods < 1:
        raise ValueError("periods must be at least 1")
    return con.execute("""
        WITH cohorts AS (
            SELECT user_id, date_trunc('month', signup_date)::DATE AS cohort_month
            FROM mart.dim_users
        ), active_months AS (
            SELECT DISTINCT user_id, date_trunc('month', event_date)::DATE AS active_month
            FROM mart.fact_events
        ), grid AS (
            SELECT c.user_id, c.cohort_month, period_number,
                   (c.cohort_month + period_number * INTERVAL '1 month')::DATE AS period_month
            FROM cohorts c CROSS JOIN range(0, ?) AS p(period_number)
        ), grouped AS (
            SELECT g.cohort_month, g.period_number,
                   count(DISTINCT g.user_id) AS users,
                   count(DISTINCT a.user_id) AS retained_users,
                   max(d.date) AS last_observed_date,
                   max(g.period_month + INTERVAL '1 month' - INTERVAL '1 day') AS period_end
            FROM grid g
            LEFT JOIN active_months a ON a.user_id = g.user_id AND a.active_month = g.period_month
            CROSS JOIN (SELECT max(date) AS date FROM mart.dim_date) d
            GROUP BY g.cohort_month, g.period_number
        )
        SELECT cohort_month, period_number, users, retained_users,
               CASE WHEN period_end <= last_observed_date
                    THEN retained_users::DOUBLE / NULLIF(users, 0)
                    ELSE NULL END AS retention_rate
        FROM grouped ORDER BY cohort_month, period_number
    """, [periods]).df()


def unit_economics(
    con: duckdb.DuckDBPyConnection,
    as_of: date,
    horizon_days: int = 90,
) -> dict[str, Any]:
    if horizon_days < 1:
        raise ValueError("horizon_days must be at least 1")
    start = pd.Timestamp(as_of) - pd.Timedelta(horizon_days - 1, unit="D")
    end = pd.Timestamp(as_of)
    revenue, payers = con.execute("""
        SELECT coalesce(sum(revenue), 0), count(DISTINCT user_id)
        FROM mart.fact_revenue_daily
        WHERE invoice_date BETWEEN ? AND ?
    """, [start.date(), end.date()]).fetchone()
    new_payers = con.execute("""
        SELECT count(DISTINCT user_id)
        FROM mart.fact_subscriptions
        WHERE paid_start_at IS NOT NULL
          AND CAST(paid_start_at AS DATE) BETWEEN ? AND ?
    """, [start.date(), end.date()]).fetchone()[0]
    spend = con.execute("""
        SELECT coalesce(sum(spend), 0) FROM mart.fact_acquisition_daily
        WHERE spend_date BETWEEN ? AND ?
    """, [start.date(), end.date()]).fetchone()[0]

    arpu = float(revenue / payers) if payers else None
    monthly_arpu = float(revenue / payers / (horizon_days / 30)) if payers else None
    cac = float(spend / new_payers) if new_payers else None
    churn = churn_rate(con, start.date(), end.date())
    monthly_churn = None
    if churn["rate"] is not None:
        monthly_churn = float(1 - (1 - churn["rate"]) ** (30 / horizon_days))
    gross_margin = 0.80
    ltv = monthly_arpu * gross_margin / monthly_churn if monthly_arpu is not None and monthly_churn and monthly_churn > 0 else None
    payback = cac / (monthly_arpu * gross_margin) if cac is not None and monthly_arpu else None
    return {
        "arpu": arpu,
        "ltv": float(ltv) if ltv is not None else None,
        "cac": cac,
        "payback_months": float(payback) if payback is not None else None,
        "gross_margin_assumption": gross_margin,
        "observed_monthly_churn": monthly_churn,
        "paying_users": int(payers),
        "new_paying_users": int(new_payers),
        "revenue_horizon": float(revenue),
    }
