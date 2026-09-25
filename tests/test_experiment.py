from datetime import date, timedelta

import duckdb
import pytest

from subscription_analytics.experiment import analyze_onboarding_experiment


def make_experiment_db(control_n=100, treatment_n=100, control_conversions=30, treatment_conversions=40, assigned='2026-01-01'):
    con = duckdb.connect(":memory:")
    con.execute("CREATE SCHEMA mart")
    con.execute("CREATE TABLE mart.fact_experiment_assignments(assignment_id VARCHAR, user_id VARCHAR, variant VARCHAR, assigned_at TIMESTAMP, experiment_name VARCHAR)")
    con.execute("CREATE TABLE mart.fact_subscriptions(subscription_id VARCHAR, user_id VARCHAR, paid_start_at TIMESTAMP)")
    rows = []
    subs = []
    for variant, size, converted in (("control", control_n, control_conversions), ("treatment", treatment_n, treatment_conversions)):
        for i in range(size):
            user_id = f"{variant}-{i}"
            rows.append((f"a-{user_id}", user_id, variant, assigned, "onboarding_v2"))
            if i < converted:
                subs.append((f"s-{user_id}", user_id, (date.fromisoformat(assigned) + timedelta(days=5)).isoformat()))
    con.executemany("INSERT INTO mart.fact_experiment_assignments VALUES (?, ?, ?, ?, ?)", rows)
    if subs:
        con.executemany("INSERT INTO mart.fact_subscriptions VALUES (?, ?, ?)", subs)
    return con


def test_experiment_reports_lift_and_uncertainty():
    con = make_experiment_db()
    result = analyze_onboarding_experiment(con, date(2026, 2, 1))
    assert result["treatment_n"] == 100 and result["control_n"] == 100
    assert result["absolute_lift"] == pytest.approx(0.10)
    assert result["ci_low"] <= result["absolute_lift"] <= result["ci_high"]
    assert 0 <= result["p_value"] <= 1
    assert 0 <= result["power"] <= 1
    assert result["recommendation"]


def test_experiment_excludes_users_without_mature_window():
    con = make_experiment_db(assigned="2026-01-01")
    result = analyze_onboarding_experiment(con, date(2026, 1, 3), conversion_window_days=7)
    assert result["control_n"] + result["treatment_n"] == 0
    assert result["p_value"] is None
    assert result["reason"]


@pytest.mark.parametrize("conversions", [(0, 0), (100, 100), (0, 100)])
def test_experiment_handles_boundary_conversion_rates(conversions):
    con = make_experiment_db(control_n=100, treatment_n=100, control_conversions=conversions[0], treatment_conversions=conversions[1])
    result = analyze_onboarding_experiment(con, date(2026, 2, 1))
    assert 0 <= result["p_value"] <= 1
    assert result["ci_low"] <= result["ci_high"]
