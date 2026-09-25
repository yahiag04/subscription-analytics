from __future__ import annotations

import duckdb


def _result(connection, name: str, sql: str, id_column: str | None = None) -> dict[str, object]:
    rows = connection.execute(sql).fetchall()
    failures = sum(int(row[0]) for row in rows)
    samples = [str(row[1]) for row in rows if len(row) > 1 and row[1] is not None][:5]
    return {
        "name": name,
        "passed": failures == 0,
        "failures": failures,
        "details": {"sample_ids": samples},
    }


def _table_exists(connection, table: str, schema: str = "raw") -> bool:
    return connection.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_schema=? AND table_name=?",
        [schema, table],
    ).fetchone()[0] > 0


def run_quality_checks(connection: duckdb.DuckDBPyConnection) -> list[dict[str, object]]:
    """Report duplicate keys, nulls, impossible dates, and revenue mismatches."""
    results: list[dict[str, object]] = []
    required = ("users", "events", "subscriptions", "acquisition", "experiment_assignments")
    missing = [f"{schema}.{table}" for schema in ("raw", "clean", "mart")
               for table in (("users", "events", "subscriptions", "acquisition", "experiment_assignments") if schema == "raw" else
                             ("users", "events", "subscriptions", "acquisition", "experiment_assignments") if schema == "clean" else
                             ("dim_users", "fact_events", "fact_subscriptions", "fact_acquisition_daily", "fact_experiment_assignments", "dim_date", "fact_revenue_daily"))
               if not _table_exists(connection, table, schema)]
    results.append({
        "name": "required_tables_present",
        "passed": not missing,
        "failures": len(missing),
        "details": {"missing_tables": missing},
    })
    if not _table_exists(connection, "events"):
        return results

    results.append(_result(connection, "unique_event_id", """
        SELECT count(*) - count(DISTINCT event_id), min(event_id)
        FROM raw.events
    """))
    results.append(_result(connection, "required_event_fields", """
        SELECT count_if(event_id IS NULL OR user_id IS NULL OR event_ts IS NULL), min(event_id)
        FROM raw.events
    """))
    if _table_exists(connection, "users"):
        results.append(_result(connection, "valid_event_time", """
            SELECT count(*), min(e.event_id)
            FROM raw.events e LEFT JOIN raw.users u USING (user_id)
            WHERE u.user_id IS NULL OR CAST(e.event_ts AS DATE) < u.signup_date
               OR CAST(e.event_ts AS DATE) > (SELECT max(spend_date) FROM raw.acquisition)
        """))
        results.append(_result(connection, "unique_user_id", """
            SELECT count(*) - count(DISTINCT user_id), min(user_id) FROM raw.users
        """))
    if _table_exists(connection, "subscriptions"):
        results.append(_result(connection, "valid_subscription_timing", """
            SELECT count(*), min(subscription_id)
            FROM raw.subscriptions
            WHERE (paid_start_at IS NOT NULL AND trial_end_at IS NOT NULL AND paid_start_at < trial_end_at)
               OR (canceled_at IS NOT NULL AND paid_start_at IS NOT NULL AND canceled_at < paid_start_at)
        """))
    if _table_exists(connection, "fact_revenue_daily", "mart") and _table_exists(connection, "fact_subscriptions", "mart"):
        results.append(_result(connection, "revenue_subscription_match", """
            WITH date_bounds AS (
                SELECT max(spend_date) AS max_date FROM clean.acquisition
            ), expected AS (
                SELECT s.subscription_id, s.user_id,
                       CAST(s.paid_start_at + month_offset * INTERVAL '1 month' AS DATE) AS invoice_date,
                       s.monthly_price AS revenue
                FROM mart.fact_subscriptions s
                CROSS JOIN date_bounds b
                CROSS JOIN LATERAL range(
                    0,
                    date_diff('month', CAST(s.paid_start_at AS DATE),
                              least(coalesce(CAST(s.canceled_at AS DATE), b.max_date), b.max_date)) + 1
                ) AS offsets(month_offset)
                WHERE s.paid_start_at IS NOT NULL
                  AND CAST(s.paid_start_at + month_offset * INTERVAL '1 month' AS DATE) <= b.max_date
                  AND (s.canceled_at IS NULL OR CAST(s.paid_start_at + month_offset * INTERVAL '1 month' AS DATE) < CAST(s.canceled_at AS DATE))
            ), discrepancies AS (
                SELECT coalesce(r.invoice_id, e.subscription_id || ':' || CAST(e.invoice_date AS VARCHAR)) AS invoice_id
                FROM expected e
                FULL OUTER JOIN mart.fact_revenue_daily r
                  ON r.subscription_id = e.subscription_id AND r.user_id = e.user_id
                 AND r.invoice_date = e.invoice_date
                WHERE e.subscription_id IS NULL OR r.subscription_id IS NULL
                   OR abs(r.revenue - e.revenue) > 0.005
            )
            SELECT count(*), min(invoice_id) FROM discrepancies
        """))
    return results
