"""
load_database.py
----------------
Load the curated accident dataset, weather grid data, and road context
lookup into a SQLite database (default) or PostgreSQL.

Configuration (via environment variables or .env file):
    DB_TYPE         sqlite | postgres   (default: sqlite)
    SQLITE_PATH     path to .db file   (default: data/traffic_accidents.db)
    POSTGRES_HOST   hostname
    POSTGRES_PORT   port (default: 5432)
    POSTGRES_DB     database name
    POSTGRES_USER   username
    POSTGRES_PASSWORD  password

Usage:
    python scripts/load_database.py
"""

import logging
import os
import sqlite3
from pathlib import Path

import pandas as pd

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Input files ───────────────────────────────────────────────────────────
ACCIDENTS_FILE = Path("data/cleaned/accident_severity_dataset.csv")
WEATHER_FILE   = Path("data/cleaned/weather_processed.csv")
LOOKUP_FILE    = Path("data/cleaned/road_context_lookup.csv")
SQL_SCHEMA     = Path("sql/schema.sql")

# ── Configuration ─────────────────────────────────────────────────────────
def get_config() -> dict:
    """Read database config from environment (load .env if present)."""
    # Simple .env loader — no python-dotenv dependency required
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())

    return {
        "db_type":         os.getenv("DB_TYPE", "sqlite"),
        "sqlite_path":     os.getenv("SQLITE_PATH", "data/traffic_accidents.db"),
        "postgres_host":   os.getenv("POSTGRES_HOST", "localhost"),
        "postgres_port":   int(os.getenv("POSTGRES_PORT", "5432")),
        "postgres_db":     os.getenv("POSTGRES_DB", "traffic_accidents"),
        "postgres_user":   os.getenv("POSTGRES_USER", ""),
        "postgres_password": os.getenv("POSTGRES_PASSWORD", ""),
    }


def get_sqlite_connection(path: str) -> sqlite3.Connection:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    log.info("Connected to SQLite: %s", db_path)
    return conn


def apply_schema(conn: sqlite3.Connection) -> None:
    """Run schema.sql to create/recreate tables."""
    schema_sql = SQL_SCHEMA.read_text(encoding="utf-8")
    # SQLite doesn't support AUTOINCREMENT the same way — strip PostgreSQL hint
    conn.executescript(schema_sql)
    conn.commit()
    log.info("Schema applied.")


def load_accidents(conn: sqlite3.Connection) -> None:
    log.info("Loading accidents …")
    df = pd.read_csv(ACCIDENTS_FILE, low_memory=False)
    log.info("  %d rows", len(df))

    # Select only the columns defined in the schema (ignore extras)
    schema_cols = [
        "collision_index", "date", "day_of_week", "month", "year",
        "hour_of_day", "is_weekend", "latitude", "longitude",
        "collision_severity", "number_of_vehicles", "number_of_casualties",
        "road_type", "road_type_label", "speed_limit", "speed_limit_label",
        "light_conditions", "light_conditions_label",
        "weather_conditions", "weather_conditions_label",
        "road_surface_conditions", "road_surface_conditions_label",
        "junction_detail", "junction_detail_label",
        "urban_or_rural_area", "urban_or_rural_area_label",
        "grid_latitude", "grid_longitude",
        "wx_temp_mean_c", "wx_precip_mm", "wx_wind_max_kmh",
    ]
    available = [c for c in schema_cols if c in df.columns]
    df_out = df[available].copy()

    # Rename date column to accident_date
    if "date" in df_out.columns:
        df_out.rename(columns={"date": "accident_date"}, inplace=True)

    df_out.to_sql("accidents", conn, if_exists="replace", index=False)
    log.info("  Loaded %d rows into 'accidents'", len(df_out))


def load_weather(conn: sqlite3.Connection) -> None:
    log.info("Loading weather grid …")
    df = pd.read_csv(WEATHER_FILE, parse_dates=["date"])
    df.rename(columns={"date": "obs_date"}, inplace=True)
    df["obs_date"] = df["obs_date"].dt.strftime("%Y-%m-%d")
    df.to_sql("weather_grid", conn, if_exists="replace", index=False)
    log.info("  Loaded %d rows into 'weather_grid'", len(df))


def load_lookup(conn: sqlite3.Connection) -> None:
    log.info("Loading road context lookup …")
    df = pd.read_csv(LOOKUP_FILE)
    df.to_sql("road_context_lookup", conn, if_exists="replace", index=False)
    log.info("  Loaded %d rows into 'road_context_lookup'", len(df))


def run_checks(conn: sqlite3.Connection) -> None:
    log.info("Running integrity checks …")
    cur = conn.cursor()

    # Row counts
    for table in ["accidents", "weather_grid", "road_context_lookup"]:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        n = cur.fetchone()[0]
        log.info("  %-25s  %d rows", table, n)

    # Uniqueness of collision_index
    cur.execute(
        "SELECT COUNT(*) - COUNT(DISTINCT collision_index) FROM accidents"
    )
    dupes = cur.fetchone()[0]
    if dupes > 0:
        log.warning("  Duplicate collision_index values: %d", dupes)
    else:
        log.info("  collision_index uniqueness: PASS")


def main() -> None:
    cfg = get_config()

    if cfg["db_type"] == "postgres":
        try:
            import psycopg2
        except ImportError:
            log.error(
                "psycopg2 is not installed. "
                "Run: pip install psycopg2-binary\n"
                "Or set DB_TYPE=sqlite to use SQLite instead."
            )
            raise

        # PostgreSQL path — connect and load using pandas + psycopg2
        import psycopg2
        conn_pg = psycopg2.connect(
            host=cfg["postgres_host"],
            port=cfg["postgres_port"],
            dbname=cfg["postgres_db"],
            user=cfg["postgres_user"],
            password=cfg["postgres_password"],
        )
        log.info("Connected to PostgreSQL: %s/%s", cfg["postgres_host"], cfg["postgres_db"])
        # For PostgreSQL, use SQLAlchemy-style loading via pandas
        from sqlalchemy import create_engine
        dsn = (
            f"postgresql+psycopg2://{cfg['postgres_user']}:{cfg['postgres_password']}"
            f"@{cfg['postgres_host']}:{cfg['postgres_port']}/{cfg['postgres_db']}"
        )
        engine = create_engine(dsn)
        acc = pd.read_csv(ACCIDENTS_FILE, low_memory=False)
        acc.to_sql("accidents", engine, if_exists="replace", index=False)
        weather = pd.read_csv(WEATHER_FILE)
        weather.to_sql("weather_grid", engine, if_exists="replace", index=False)
        lookup = pd.read_csv(LOOKUP_FILE)
        lookup.to_sql("road_context_lookup", engine, if_exists="replace", index=False)
        log.info("PostgreSQL load complete.")
    else:
        # SQLite (default)
        conn = get_sqlite_connection(cfg["sqlite_path"])
        try:
            apply_schema(conn)
            load_accidents(conn)
            load_weather(conn)
            load_lookup(conn)
            run_checks(conn)
        finally:
            conn.close()

    print("\n========== DATABASE LOAD SUMMARY ==========")
    print(f"Database type : {cfg['db_type']}")
    if cfg["db_type"] == "sqlite":
        print(f"Database file : {cfg['sqlite_path']}")
    else:
        print(f"Database      : {cfg['postgres_host']}/{cfg['postgres_db']}")
    print("Tables loaded : accidents, weather_grid, road_context_lookup")
    print("Status        : Complete")


if __name__ == "__main__":
    main()
