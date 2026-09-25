from pathlib import Path

SIMULATION_END = "2026-09-25"
DEFAULT_SEED = 42
DEFAULT_USERS = 5_000
DEFAULT_DAYS = 180
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DB_PATH = DEFAULT_DATA_DIR / "warehouse.duckdb"
