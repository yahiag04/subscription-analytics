# Subscription Product Analytics & Forecasting Platform

A compact, end-to-end analytics demo for a fictional subscription app. All data is synthetic and deterministic; it does not represent a real company's customers or performance.

## Quick start

Requires Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python -m subscription_analytics.cli build --output data --seed 42 --users 5000 --days 180
python -m subscription_analytics.cli quality --db data/warehouse.duckdb
streamlit run app.py
```

The build command writes five CSVs under `data/raw/` and creates `data/warehouse.duckdb`. It prints table row counts and runs quality checks. The `quality` command exits nonzero if a blocking check fails. Use `--help` on either command for options. The random seed and simulation end date are fixed for reproducibility; the onboarding treatment has a simulated +3 percentage-point trial-conversion lift by default.

## Pipeline and warehouse

The generator creates users, session/content/feature events, subscription lifecycles, daily acquisition spend, and randomized onboarding assignments. The DuckDB pipeline preserves three layers:

- `raw`: typed copies of generated CSVs, including duplicate rows for auditability.
- `clean`: exact duplicate rows removed, parsed dates/timestamps, same fields and grains.
- `mart`: `dim_users`, `fact_events` (one row per event), `fact_subscriptions` (one row per subscription), `fact_acquisition_daily` (one row per channel/day), `fact_experiment_assignments` (one row per assignment), `fact_revenue_daily` (one row per monthly invoice), and `dim_date` (one row per calendar day).

Revenue is booked on monthly subscription billing dates. A cancellation stops later invoices. Synthetic acquisition spend uses a channel-level CAC assumption applied to new users acquired on each day.

## Metric definitions

- **DAU / WAU / MAU:** distinct users with at least one event in the trailing 1 / 7 / 30 calendar days.
- **Trial-to-paid conversion:** converted trials divided by trials whose full conversion window has elapsed. The default maturity window is seven days.
- **Churn:** effective cancellations in the selected period divided by paid subscriptions active at the beginning of the period. A zero denominator returns no rate.
- **Cohort retention:** users in each signup month with at least one event in each cohort month; immature future periods are blank.
- **ARPU:** invoice revenue in the selected horizon divided by distinct paying users in that horizon.
- **LTV:** modeled monthly ARPU times an 80% gross-margin assumption divided by estimated monthly churn. It is a scenario estimate, not an observed lifetime value.
- **CAC payback:** acquisition cost per new payer divided by monthly ARPU after the same gross-margin assumption.

## Forecast and experiment interpretation

Forecasts compare seasonal-naive, moving-average, and linear-trend methods using a temporal holdout; the implementation reports MAE and MAPE when defined. Forecast intervals are not causal claims and should be read alongside the baseline error and recent trend.

The onboarding experiment assigns one variant per user and simulates a known lift. Conversion is measured within seven days after assignment, excluding users whose full window is not yet observed. The analysis reports absolute/relative lift, confidence interval, p-value, and achieved power. A statistical result alone does not determine whether the feature is worth shipping: consider the lift size, confidence bounds, implementation cost, and whether the effect is large enough to matter to the product.

## Limitations

This is a synthetic, single-product demo with simplified monthly billing, fixed prices, and assumed acquisition costs. It omits refunds, taxes, app-store fees, currency conversion, reactivations, and live ingestion. Forecasts use intentionally interpretable baselines rather than complex ML. No personally identifying information is generated.

### Dashboard filters

The sidebar filters channel, country, device, and reporting dates. User-based metrics follow the selected segment. Acquisition spend is apportioned within channel/day using selected users' share of signups because source spend is generated at channel/day grain; interpret segmented CAC as an estimate.
