import duckdb

from subscription_analytics.quality import run_quality_checks


def test_quality_flags_duplicate_event_ids():
    con = duckdb.connect(":memory:")
    con.execute("create schema raw")
    con.execute("create table raw.events(event_id varchar, user_id varchar, event_ts timestamp)")
    con.execute("insert into raw.events values ('e1', 'u1', '2026-01-01'), ('e1', 'u1', '2026-01-01')")

    results = run_quality_checks(con)

    assert not next(r for r in results if r["name"] == "unique_event_id")["passed"]
    con.close()


def test_quality_returns_rule_details_for_missing_user_ids():
    con = duckdb.connect(":memory:")
    con.execute("create schema raw")
    con.execute("create table raw.events(event_id varchar, user_id varchar, event_ts timestamp)")
    con.execute("insert into raw.events values ('e1', null, '2026-01-01')")

    results = run_quality_checks(con)

    rule = next(r for r in results if r["name"] == "required_event_fields")
    assert not rule["passed"]
    assert rule["failures"] == 1
    con.close()


def test_generated_warehouse_passes_all_quality_rules(tmp_path):
    from subscription_analytics.generate import generate_dataset
    from subscription_analytics.quality import run_quality_checks
    from subscription_analytics.warehouse import build_warehouse

    raw = tmp_path / "raw"
    generate_dataset(raw, seed=12, n_users=25, days=35)
    db = tmp_path / "warehouse.duckdb"
    build_warehouse(raw, db)
    con = duckdb.connect(str(db))

    results = run_quality_checks(con)

    assert all(result["passed"] for result in results), results
    con.close()


def test_quality_flags_impossible_event_time(tmp_path):
    from subscription_analytics.generate import generate_dataset
    from subscription_analytics.warehouse import build_warehouse

    raw = tmp_path / "raw"
    generate_dataset(raw, seed=4, n_users=20, days=30)
    db = tmp_path / "warehouse.duckdb"
    build_warehouse(raw, db)
    con = duckdb.connect(str(db))
    con.execute("UPDATE raw.events SET event_ts='2010-01-01' WHERE event_id=(SELECT min(event_id) FROM raw.events)")

    rule = next(r for r in run_quality_checks(con) if r["name"] == "valid_event_time")

    assert not rule["passed"]
    assert rule["failures"] == 1
    con.close()


def test_quality_flags_revenue_subscription_mismatch(tmp_path):
    from subscription_analytics.generate import generate_dataset
    from subscription_analytics.warehouse import build_warehouse

    raw = tmp_path / "raw"
    generate_dataset(raw, seed=5, n_users=30, days=40)
    db = tmp_path / "warehouse.duckdb"
    build_warehouse(raw, db)
    con = duckdb.connect(str(db))
    con.execute("UPDATE mart.fact_revenue_daily SET revenue = revenue + 1 WHERE invoice_id=(SELECT min(invoice_id) FROM mart.fact_revenue_daily)")

    rule = next(r for r in run_quality_checks(con) if r["name"] == "revenue_subscription_match")

    assert not rule["passed"]
    assert rule["failures"] == 1
    con.close()


def test_quality_flags_missing_subscription_invoice(tmp_path):
    from subscription_analytics.generate import generate_dataset
    from subscription_analytics.warehouse import build_warehouse

    raw = tmp_path / "raw"
    generate_dataset(raw, seed=13, n_users=35, days=45)
    db = tmp_path / "warehouse.duckdb"
    build_warehouse(raw, db)
    con = duckdb.connect(str(db))
    con.execute("DELETE FROM mart.fact_revenue_daily WHERE invoice_id=(SELECT min(invoice_id) FROM mart.fact_revenue_daily)")

    rule = next(r for r in run_quality_checks(con) if r["name"] == "revenue_subscription_match")

    assert not rule["passed"]
    assert rule["failures"] == 1
    con.close()
