import subprocess
import sys

import duckdb


def test_build_command_creates_warehouse(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "subscription_analytics.cli", "build",
         "--output", str(tmp_path), "--seed", "9", "--users", "30", "--days", "40"],
        capture_output=True, text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "raw" / "users.csv").exists()
    assert (tmp_path / "warehouse.duckdb").exists()


def test_quality_command_reports_pass(tmp_path):
    from subscription_analytics.generate import generate_dataset
    from subscription_analytics.warehouse import build_warehouse

    raw = tmp_path / "raw"
    generate_dataset(raw, seed=9, n_users=30, days=40)
    db = tmp_path / "warehouse.duckdb"
    build_warehouse(raw, db)

    result = subprocess.run(
        [sys.executable, "-m", "subscription_analytics.cli", "quality", "--db", str(db)],
        capture_output=True, text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "PASS" in result.stdout
