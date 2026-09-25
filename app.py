from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from subscription_analytics.dashboard_data import (
    build_filtered_connection,
    dashboard_options,
    full_quality_results,
    normalize_date,
    series_frame,
    warehouse_status,
)
from subscription_analytics.experiment import analyze_onboarding_experiment
from subscription_analytics.forecast import METHODS, compare_forecasts, forecast_series, warehouse_series
from subscription_analytics.metrics import (
    churn_rate,
    daily_active_users,
    retention_cohorts,
    trial_conversion,
    unit_economics,
)

st.set_page_config(page_title="Subscription Analytics", page_icon="◒", layout="wide")
st.markdown("""
<style>
    .block-container { padding-top: 1.8rem; max-width: 1440px; }
    [data-testid="stMetric"] { background: #202b3d; border: 1px solid #3b4a60; padding: 14px 16px; border-radius: 12px; }
    [data-testid="stMetric"] * { color: #f4f7fb !important; }
    .synthetic-note { color: #e4eaf3; background: #202b3d; border-left: 4px solid #4884d8; padding: 10px 14px; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

DB_PATH = Path(os.environ.get("ANALYTICS_DB", "data/warehouse.duckdb")).expanduser()


def show_number(value: float | int | None, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "—"
    if isinstance(value, int) or float(value).is_integer():
        return f"{int(value):,}{suffix}"
    return f"{value:,.1%}" if suffix == "%" else f"{value:,.2f}{suffix}"


def safe_forecasts(series: pd.Series) -> tuple[pd.DataFrame | None, pd.DataFrame | None, str | None]:
    try:
        scores = compare_forecasts(series, horizons=(30, 90))
        predictions = pd.concat([forecast_series(series, 90, method) for method in METHODS], ignore_index=True)
        return scores, predictions, None
    except ValueError as exc:
        return None, None, str(exc)


st.title("Subscription Product Analytics")
st.caption("Growth, monetization and retention · illustrative subscription app")
st.markdown('<div class="synthetic-note"><strong>Synthetic demo data.</strong> Figures and experiment effects are generated for this project and do not represent real company performance.</div>', unsafe_allow_html=True)

problem = warehouse_status(DB_PATH)
if problem:
    st.error(problem)
    st.code("python -m subscription_analytics.cli build --output data --seed 42 --users 5000 --days 180")
    st.stop()

with duckdb.connect(str(DB_PATH), read_only=True) as source:
    date_min, date_max = source.execute("SELECT min(date), max(date) FROM mart.dim_date").fetchone()
    date_min, date_max = normalize_date(date_min), normalize_date(date_max)
    options = dashboard_options(DB_PATH)

with st.sidebar:
    st.header("Filters")
    channel = st.selectbox("Acquisition channel", options["channel"])
    country = st.selectbox("Country", options["country"])
    device = st.selectbox("Device", options["device"])
    if date_min == date_max:
        date_range = (date_min, date_max)
    else:
        date_range = st.date_input("Date range", value=(date_min, date_max), min_value=date_min, max_value=date_max)
        if not isinstance(date_range, tuple) or len(date_range) != 2:
            date_range = (date_min, date_max)
    start_date, end_date = date_range

try:
    filtered = build_filtered_connection(
        DB_PATH,
        channel=None if channel == "All" else channel,
        country=None if country == "All" else country,
        device=None if device == "All" else device,
    )
except (ValueError, duckdb.Error) as exc:
    st.error(f"Could not load this filtered view: {exc}")
    st.stop()

with filtered:
    active = daily_active_users(filtered)
    active = active[(active.date.dt.date >= start_date) & (active.date.dt.date <= end_date)]
    revenue_series = warehouse_series(filtered, "daily_revenue")
    subscriber_series = warehouse_series(filtered, "paid_subscribers")
    revenue_series = revenue_series[revenue_series.index.date <= end_date]
    subscriber_series = subscriber_series[subscriber_series.index.date <= end_date]
    conversion = trial_conversion(filtered, as_of=end_date, maturity_days=7)
    churn = churn_rate(filtered, start=max(date_min, end_date - pd.Timedelta(29, unit="D")), end=end_date)
    economics = unit_economics(filtered, as_of=end_date, horizon_days=min(90, (end_date - date_min).days + 1))
    experiment = analyze_onboarding_experiment(filtered, as_of=end_date, conversion_window_days=7)
    cohorts = retention_cohorts(filtered, periods=6)
    revenue_scores, revenue_forecasts, revenue_error = safe_forecasts(revenue_series)
    subscriber_scores, subscriber_forecasts, subscriber_error = safe_forecasts(subscriber_series)

    latest = active.iloc[-1] if not active.empty else None
    kpis = st.columns(6)
    kpis[0].metric("DAU", show_number(int(latest.dau) if latest is not None else None))
    kpis[1].metric("WAU", show_number(int(latest.wau) if latest is not None else None))
    kpis[2].metric("MAU", show_number(int(latest.mau) if latest is not None else None))
    kpis[3].metric("Trial → paid", show_number(conversion["rate"], "%"))
    kpis[4].metric("30d churn", show_number(churn["rate"], "%"))
    kpis[5].metric("Revenue / 90d", show_number(economics["revenue_horizon"], " $"))

    st.caption(f"{len(filtered.execute('SELECT user_id FROM mart.dim_users').fetchall()):,} users in selected segment · reporting through {end_date:%d %b %Y}")

    left, right = st.columns([1.35, 1])
    with left:
        st.subheader("Engagement")
        if active.empty:
            st.info("No activity in the selected range.")
        else:
            chart = px.line(active, x="date", y=["dau", "wau", "mau"], labels={"value": "Active users", "date": "Date", "variable": "Window"}, color_discrete_sequence=["#2864dc", "#23a899", "#f19d38"])
            chart.update_layout(legend_title_text="", margin=dict(l=0, r=0, t=10, b=0), height=310)
            st.plotly_chart(chart, width="stretch")
    with right:
        st.subheader("Monetization & acquisition")
        cards = st.columns(2)
        cards[0].metric("ARPU · 90d", show_number(economics["arpu"], " $"))
        cards[1].metric("Modeled LTV", show_number(economics["ltv"], " $"))
        cards[0].metric("CAC / new payer", show_number(economics["cac"], " $"))
        cards[1].metric("CAC payback", show_number(economics["payback_months"], " mo"))
        st.caption(f"LTV/payback assume {economics['gross_margin_assumption']:.0%} gross margin. CAC spend is apportioned by selected users within channel/day.")
        revenue_chart_data = series_frame(revenue_series, "Revenue")
        if not revenue_chart_data.empty:
            revenue_chart = px.bar(revenue_chart_data[revenue_chart_data.date.dt.date >= start_date], x="date", y="Revenue", color_discrete_sequence=["#2864dc"])
            revenue_chart.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0), height=205)
            st.plotly_chart(revenue_chart, width="stretch")

    st.subheader("Revenue & subscriber outlook")
    forecast_cols = st.columns(2)
    for col, label, history, scores, predictions, error in (
        (forecast_cols[0], "Daily revenue", revenue_series, revenue_scores, revenue_forecasts, revenue_error),
        (forecast_cols[1], "Paid subscribers", subscriber_series, subscriber_scores, subscriber_forecasts, subscriber_error),
    ):
        with col:
            st.markdown(f"**{label} · next 90 days**")
            if error:
                st.info(f"Not enough daily history to compare forecasts: {error}")
                continue
            assert scores is not None and predictions is not None
            best_30 = scores[scores.horizon == 30].sort_values("mae").iloc[0]
            best_90 = scores[scores.horizon == 90].sort_values("mae").iloc[0]
            selected_predictions = predictions[predictions.method.isin({best_30.method, best_90.method})]
            history_frame = series_frame(history, "value")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=history_frame.date, y=history_frame.value, name="Observed", line=dict(color="#183b66", width=2)))
            for method in selected_predictions.method.unique():
                forecast = selected_predictions[selected_predictions.method == method]
                fig.add_trace(go.Scatter(x=forecast.date, y=forecast.forecast, name=f"Forecast · {method.replace('_', ' ')}", line=dict(dash="dash")))
            fig.update_layout(height=280, margin=dict(l=0, r=0, t=5, b=0), legend=dict(orientation="h", y=-0.28))
            st.plotly_chart(fig, width="stretch")
            st.caption(f"30d best: {best_30.method.replace('_', ' ')} (MAE {best_30.mae:,.2f}) · 90d best: {best_90.method.replace('_', ' ')} (MAE {best_90.mae:,.2f}). Historical temporal backtest; no random split.")
            with st.expander(f"Compare all methods · {label.lower()}"):
                st.dataframe(scores, hide_index=True, width="stretch")

    st.subheader("Onboarding experiment")
    exp_cols = st.columns([1, 1, 2])
    exp_cols[0].metric("Control conversion", show_number(experiment["control_rate"], "%"))
    exp_cols[1].metric("Treatment conversion", show_number(experiment["treatment_rate"], "%"))
    with exp_cols[2]:
        if experiment["absolute_lift"] is None:
            st.info(experiment["recommendation"])
        else:
            st.markdown(f"**{experiment['recommendation']}**")
            st.caption(f"Absolute lift {experiment['absolute_lift']:+.1%} · 95% CI [{experiment['ci_low']:+.1%}, {experiment['ci_high']:+.1%}] · p={experiment['p_value']:.3f} · power={experiment['power']:.0%} · 80% MDE={experiment['minimum_detectable_effect']:.1%}")
            st.caption(experiment["reason"])
    st.caption(f"Mature users: control {experiment['control_n']:,} · treatment {experiment['treatment_n']:,}. Conversion window: 7 days after assignment.")

    lower_left, lower_right = st.columns([1.35, 1])
    with lower_left:
        st.subheader("Signup cohort retention")
        if cohorts.empty:
            st.info("No cohort data available for this segment.")
        else:
            pivot = cohorts.pivot(index="cohort_month", columns="period_number", values="retention_rate")
            pivot.columns = ["Month 0" if c == 0 else f"Month {c}" for c in pivot.columns]
            st.dataframe(pivot.style.format("{:.0%}", na_rep="—"), width="stretch")
    with lower_right:
        st.subheader("Data quality")
        quality = full_quality_results(DB_PATH)
        failed = [check for check in quality if not check["passed"]]
        st.metric("Checks passed", f"{len(quality) - len(failed)} / {len(quality)}", delta="All clear" if not failed else f"{len(failed)} failed", delta_color="normal" if not failed else "inverse")
        with st.expander("Check details"):
            for check in quality:
                st.write(("✅" if check["passed"] else "❌") + f" {check['name']} · {check['failures']} issue(s)")

st.caption("Definitions: active users are distinct users with an event in a trailing calendar window. Churn uses effective cancellation dates. LTV and CAC payback are modeled assumptions. Synthetic demo · USD-equivalent prices.")
