from __future__ import annotations

from datetime import date
from math import sqrt
from typing import Any

import duckdb
from scipy.stats import norm


def _wilson(successes: int, total: int, z: float) -> tuple[float, float]:
    if total == 0:
        return (0.0, 1.0)
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    radius = z * sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _power(control_rate: float, treatment_rate: float, control_n: int, treatment_n: int, alpha: float) -> float:
    delta = treatment_rate - control_rate
    pooled = (control_rate + treatment_rate) / 2
    null_se = sqrt(max(pooled * (1 - pooled) * (1 / control_n + 1 / treatment_n), 0.0))
    alt_se = sqrt(max(control_rate * (1 - control_rate) / control_n + treatment_rate * (1 - treatment_rate) / treatment_n, 0.0))
    if alt_se == 0:
        return alpha if delta == 0 else 1.0
    if null_se == 0:
        return alpha if delta == 0 else 1.0
    critical = norm.ppf(1 - alpha / 2) * null_se
    effect = abs(delta)
    power = norm.cdf((-critical - effect) / alt_se) + 1 - norm.cdf((critical - effect) / alt_se)
    return float(min(1.0, max(0.0, power)))


def _mde(control_rate: float, control_n: int, treatment_n: int, alpha: float) -> float:
    low, high = 0.0, 1.0
    for _ in range(50):
        middle = (low + high) / 2
        treatment_rate = min(1.0, control_rate + middle)
        if _power(control_rate, treatment_rate, control_n, treatment_n, alpha) >= 0.80:
            high = middle
        else:
            low = middle
    return high


def analyze_onboarding_experiment(
    con: duckdb.DuckDBPyConnection,
    as_of: date,
    conversion_window_days: int = 7,
    alpha: float = 0.05,
) -> dict[str, Any]:
    if conversion_window_days < 1:
        raise ValueError("conversion_window_days must be at least 1")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    arms = con.execute("""
        WITH assignments AS (
            SELECT user_id, variant, assigned_at::DATE AS assigned_date
            FROM mart.fact_experiment_assignments
            WHERE experiment_name = 'onboarding_v2'
              AND assigned_at::DATE + ? <= ?
            QUALIFY row_number() OVER (PARTITION BY user_id ORDER BY assigned_at, assignment_id) = 1
        ), outcomes AS (
            SELECT a.user_id, a.variant,
                   coalesce(bool_or(s.paid_start_at IS NOT NULL
                       AND s.paid_start_at::DATE BETWEEN a.assigned_date
                           AND a.assigned_date + ?), false) AS converted
            FROM assignments a
            LEFT JOIN mart.fact_subscriptions s USING (user_id)
            GROUP BY a.user_id, a.variant
        )
        SELECT variant, count(*) AS total,
               count(*) FILTER (WHERE converted) AS converted
        FROM outcomes GROUP BY variant
    """, [conversion_window_days, as_of, conversion_window_days]).fetchall()
    by_variant = {variant: (int(total), int(converted)) for variant, total, converted in arms}
    control_n, control_success = by_variant.get("control", (0, 0))
    treatment_n, treatment_success = by_variant.get("treatment", (0, 0))
    result: dict[str, Any] = {
        "control_n": control_n,
        "treatment_n": treatment_n,
        "control_rate": control_success / control_n if control_n else None,
        "treatment_rate": treatment_success / treatment_n if treatment_n else None,
        "absolute_lift": None,
        "relative_lift": None,
        "ci_low": None,
        "ci_high": None,
        "p_value": None,
        "power": None,
        "minimum_detectable_effect": None,
        "reason": None,
        "recommendation": "Insufficient data: both experiment variants need mature users.",
    }
    if not control_n or not treatment_n:
        result["reason"] = "No mature assignments in both arms for the selected conversion window."
        return result

    p_control = control_success / control_n
    p_treatment = treatment_success / treatment_n
    lift = p_treatment - p_control
    result["absolute_lift"] = lift
    result["relative_lift"] = lift / p_control if p_control else None

    z = norm.ppf(1 - alpha / 2)
    c_low, c_high = _wilson(control_success, control_n, z)
    t_low, t_high = _wilson(treatment_success, treatment_n, z)
    result["ci_low"] = lift - sqrt((p_treatment - t_low) ** 2 + (c_high - p_control) ** 2)
    result["ci_high"] = lift + sqrt((t_high - p_treatment) ** 2 + (p_control - c_low) ** 2)

    pooled = (control_success + treatment_success) / (control_n + treatment_n)
    standard_error = sqrt(pooled * (1 - pooled) * (1 / control_n + 1 / treatment_n))
    if standard_error == 0:
        p_value = 1.0 if lift == 0 else 0.0
    else:
        z_score = lift / standard_error
        p_value = float(2 * norm.sf(abs(z_score)))
    result["p_value"] = p_value
    result["power"] = _power(p_control, p_treatment, control_n, treatment_n, alpha)
    result["minimum_detectable_effect"] = _mde(p_control, control_n, treatment_n, alpha)

    if result["ci_low"] > 0 and lift >= 0.02:
        result["recommendation"] = "Ship to a monitored rollout if retention and crash-rate guardrails remain healthy."
        result["reason"] = "The measured conversion lift is positive, statistically supported, and at least 2 percentage points."
    elif result["ci_high"] < 0:
        result["recommendation"] = "Do not ship this onboarding version; investigate the conversion loss."
        result["reason"] = "The confidence interval indicates a conversion decrease."
    elif result["power"] < 0.80:
        result["recommendation"] = "Extend the experiment before deciding; the current sample has low power."
        result["reason"] = "The achieved power is below 80%, so a useful effect could be missed."
    else:
        result["recommendation"] = "Keep the current onboarding and refine the test or target segment before rollout."
        result["reason"] = "The result does not establish a product-relevant positive lift."
    return result
