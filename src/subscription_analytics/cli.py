from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from .generate import generate_dataset
from .quality import run_quality_checks
from .warehouse import RAW_SCHEMAS, build_warehouse


def _build(args: argparse.Namespace) -> int:
    output = Path(args.output).expanduser().resolve()
    raw_dir = output / "raw"
    database = output / "warehouse.duckdb"
    files = generate_dataset(raw_dir, seed=args.seed, n_users=args.users, days=args.days)
    build_warehouse(raw_dir, database)
    print(f"Built DuckDB warehouse: {database}")
    for table, path in files.items():
        print(f"  {table}: {path}")
    with duckdb.connect(str(database), read_only=True) as con:
        for table in RAW_SCHEMAS:
            count = con.execute(f"SELECT count(*) FROM clean.{table}").fetchone()[0]
            print(f"  clean.{table}: {count:,} rows")
        results = run_quality_checks(con)
    _print_quality(results)
    return 0 if all(result["passed"] for result in results) else 1


def _quality(args: argparse.Namespace) -> int:
    database = Path(args.db).expanduser().resolve()
    if not database.is_file():
        print(f"Warehouse not found: {database}. Run `python -m subscription_analytics.cli build` first.")
        return 2
    try:
        with duckdb.connect(str(database), read_only=True) as con:
            results = run_quality_checks(con)
    except duckdb.Error as exc:
        print(f"Cannot read warehouse {database}: {exc}. Rebuild it with the documented build command.")
        return 2
    _print_quality(results)
    return 0 if all(result["passed"] for result in results) else 1


def _print_quality(results: list[dict[str, object]]) -> None:
    for result in results:
        label = "PASS" if result["passed"] else "FAIL"
        print(f"{label} {result['name']} (failures={result['failures']})")
        details = result["details"]
        if result["failures"] and isinstance(details, dict):
            for key, value in details.items():
                if value:
                    print(f"  {key}: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synthetic subscription analytics pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="generate raw data and build the DuckDB warehouse")
    build.add_argument("--output", default="data", help="output directory (default: data)")
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--users", type=int, default=5_000)
    build.add_argument("--days", type=int, default=180)
    build.set_defaults(handler=_build)
    quality = commands.add_parser("quality", help="run warehouse data-quality checks")
    quality.add_argument("--db", default="data/warehouse.duckdb")
    quality.set_defaults(handler=_quality)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
