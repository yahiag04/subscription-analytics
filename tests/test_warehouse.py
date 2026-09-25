import duckdb

from subscription_analytics.generate import generate_dataset
from subscription_analytics.warehouse import build_warehouse


def test_build_warehouse_creates_layered_tables(tmp_path):
    raw_dir = tmp_path / "raw"
    generate_dataset(raw_dir, seed=11, n_users=40, days=45)
    db = tmp_path / "analytics.duckdb"

    build_warehouse(raw_dir, db)

    con = duckdb.connect(str(db), read_only=True)
    assert con.execute("select count(*) from raw.events").fetchone()[0] > 0
    assert con.execute("select count(*) from clean.events").fetchone()[0] > 0
    assert con.execute("select count(*) from mart.fact_events").fetchone()[0] > 0
    assert con.execute("select count(*) from mart.fact_revenue_daily").fetchone()[0] > 0
    con.close()


def test_missing_raw_file_names_the_expected_path(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError, match="users.csv"):
        build_warehouse(tmp_path / "missing", tmp_path / "warehouse.duckdb")
