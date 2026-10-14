"""
tests/test_phase1_data.py
--------------------------
Phase 1 tests: data cleaning, integration, and database.

Run with:
    pytest tests/test_phase1_data.py -v

These tests are designed to run AFTER the Phase 1 pipeline has been executed:
    python scripts/clean_data.py
    python scripts/process_weather.py
    python scripts/collect_road_data.py
    python scripts/integrate_data.py
    python scripts/load_database.py
"""

import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

# ── Paths (relative to project root) ──────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

RAW_ACCIDENTS   = ROOT / "data/raw/accidents/dft-road-casualty-statistics-collision-2025.csv"
CLEANED_ACC     = ROOT / "data/cleaned/accidents_cleaned.csv"
REJECTED_ACC    = ROOT / "data/rejected/accidents_rejected.csv"
WEATHER_JSON    = ROOT / "data/raw/weather"
WEATHER_CSV     = ROOT / "data/cleaned/weather_processed.csv"
LOOKUP_CSV      = ROOT / "data/cleaned/road_context_lookup.csv"
INTEGRATED_CSV  = ROOT / "data/cleaned/accident_severity_dataset.csv"
DB_FILE         = ROOT / "data/traffic_accidents.db"
INT_REPORT      = ROOT / "logs/integration_report.json"

# ═══════════════════════════════════════════════════════════════════════════
# STAGE B — Accident cleaning
# ═══════════════════════════════════════════════════════════════════════════

class TestAccidentCleaning:
    """Verify the accident cleaning pipeline output."""

    def test_cleaned_file_exists(self):
        assert CLEANED_ACC.exists(), (
            f"Cleaned accidents file not found: {CLEANED_ACC}\n"
            "Run: python scripts/clean_data.py"
        )

    def test_rejected_file_exists(self):
        assert REJECTED_ACC.exists(), (
            f"Rejected records file not found: {REJECTED_ACC}"
        )

    def test_cleaned_has_records(self):
        df = pd.read_csv(CLEANED_ACC, low_memory=False, nrows=5)
        assert len(df) > 0, "Cleaned accidents file is empty."

    def test_cleaned_has_required_columns(self):
        df = pd.read_csv(CLEANED_ACC, low_memory=False, nrows=1)
        required = ["collision_index", "collision_severity", "latitude", "longitude", "date"]
        for col in required:
            assert col in df.columns, f"Required column missing: {col}"

    def test_no_missing_collision_index(self):
        df = pd.read_csv(CLEANED_ACC, usecols=["collision_index"], low_memory=False)
        n_missing = df["collision_index"].isna().sum()
        assert n_missing == 0, f"{n_missing} rows have missing collision_index in cleaned data."

    def test_no_missing_severity(self):
        df = pd.read_csv(CLEANED_ACC, usecols=["collision_severity"], low_memory=False)
        n_missing = df["collision_severity"].isna().sum()
        assert n_missing == 0, f"{n_missing} rows have missing collision_severity."

    def test_collision_index_unique(self):
        df = pd.read_csv(CLEANED_ACC, usecols=["collision_index"], low_memory=False)
        n_dupes = df["collision_index"].duplicated().sum()
        assert n_dupes == 0, f"{n_dupes} duplicate collision_index values in cleaned data."

    def test_coordinates_valid_range(self):
        df = pd.read_csv(CLEANED_ACC, usecols=["latitude", "longitude"], low_memory=False)
        df["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        bad_lat = ~df["latitude"].between(-90, 90)
        bad_lon = ~df["longitude"].between(-180, 180)
        assert bad_lat.sum() == 0, f"{bad_lat.sum()} records with invalid latitude."
        assert bad_lon.sum() == 0, f"{bad_lon.sum()} records with invalid longitude."

    def test_severity_valid_values(self):
        df = pd.read_csv(CLEANED_ACC, usecols=["collision_severity"], low_memory=False)
        df["collision_severity"] = pd.to_numeric(df["collision_severity"], errors="coerce")
        valid_values = {1, 2, 3}
        invalid = ~df["collision_severity"].isin(valid_values) & df["collision_severity"].notna()
        assert invalid.sum() == 0, (
            f"{invalid.sum()} records have collision_severity outside {{1, 2, 3}}."
        )

    def test_rejected_has_rejection_reason_column(self):
        df = pd.read_csv(REJECTED_ACC, low_memory=False, nrows=1)
        assert "rejection_reason" in df.columns, (
            "rejected file should have a 'rejection_reason' column."
        )

    def test_cleaned_count_reasonable(self):
        """Cleaned dataset should have the vast majority of raw records."""
        df = pd.read_csv(CLEANED_ACC, usecols=["collision_index"], low_memory=False)
        # We know raw has ~101,525 records; cleaned should be > 100,000
        assert len(df) > 100_000, (
            f"Cleaned dataset has only {len(df)} records — unexpectedly low."
        )


# ═══════════════════════════════════════════════════════════════════════════
# STAGE C — Weather processing
# ═══════════════════════════════════════════════════════════════════════════

class TestWeatherProcessing:
    """Verify the weather consolidation output."""

    def test_weather_csv_exists(self):
        assert WEATHER_CSV.exists(), (
            f"Weather CSV not found: {WEATHER_CSV}\n"
            "Run: python scripts/process_weather.py"
        )

    def test_weather_csv_has_records(self):
        df = pd.read_csv(WEATHER_CSV, nrows=5)
        assert len(df) > 0, "Weather CSV is empty."

    def test_weather_required_columns(self):
        df = pd.read_csv(WEATHER_CSV, nrows=1)
        required = ["grid_latitude", "grid_longitude", "date", "temp_mean_c", "precip_mm", "wind_max_kmh"]
        for col in required:
            assert col in df.columns, f"Weather CSV missing column: {col}"

    def test_weather_row_count_expected(self):
        """585 grid cells × 365 days = 213,525 rows."""
        df = pd.read_csv(WEATHER_CSV, usecols=["grid_latitude"])
        expected = 585 * 365
        assert len(df) == expected, (
            f"Expected {expected:,} weather rows (585×365), got {len(df):,}."
        )

    def test_weather_date_range(self):
        df = pd.read_csv(WEATHER_CSV, usecols=["date"], parse_dates=["date"])
        assert df["date"].min().strftime("%Y-%m-%d") == "2025-01-01"
        assert df["date"].max().strftime("%Y-%m-%d") == "2025-12-31"

    def test_weather_no_missing_temp(self):
        df = pd.read_csv(WEATHER_CSV, usecols=["temp_mean_c"])
        n_missing = df["temp_mean_c"].isna().sum()
        assert n_missing == 0, f"{n_missing} missing temperature values."

    def test_weather_grid_cells(self):
        df = pd.read_csv(WEATHER_CSV, usecols=["grid_latitude", "grid_longitude"])
        n_cells = df.drop_duplicates().shape[0]
        assert n_cells == 585, f"Expected 585 grid cells, found {n_cells}."

    def test_json_files_present(self):
        json_files = list(WEATHER_JSON.glob("weather_*.json"))
        json_files = [f for f in json_files if f.name != "weather_api_sample.json"]
        assert len(json_files) == 585, f"Expected 585 JSON files, found {len(json_files)}."


# ═══════════════════════════════════════════════════════════════════════════
# STAGE D — Road context lookup
# ═══════════════════════════════════════════════════════════════════════════

class TestRoadContextLookup:
    """Verify the road context lookup table."""

    def test_lookup_csv_exists(self):
        assert LOOKUP_CSV.exists(), (
            f"Lookup CSV not found: {LOOKUP_CSV}\n"
            "Run: python scripts/collect_road_data.py"
        )

    def test_lookup_required_columns(self):
        df = pd.read_csv(LOOKUP_CSV, nrows=1)
        for col in ["code_type", "code_value", "label"]:
            assert col in df.columns, f"Lookup CSV missing column: {col}"

    def test_lookup_has_road_type(self):
        df = pd.read_csv(LOOKUP_CSV)
        assert "road_type" in df["code_type"].values, "No road_type entries in lookup."

    def test_lookup_has_speed_limit(self):
        df = pd.read_csv(LOOKUP_CSV)
        assert "speed_limit" in df["code_type"].values, "No speed_limit entries in lookup."

    def test_lookup_no_missing_labels(self):
        df = pd.read_csv(LOOKUP_CSV)
        n_missing = df["label"].isna().sum()
        assert n_missing == 0, f"{n_missing} missing labels in lookup CSV."


# ═══════════════════════════════════════════════════════════════════════════
# STAGE E — Data integration
# ═══════════════════════════════════════════════════════════════════════════

class TestDataIntegration:
    """Verify the final integrated dataset."""

    def test_integrated_file_exists(self):
        assert INTEGRATED_CSV.exists(), (
            f"Integrated dataset not found: {INTEGRATED_CSV}\n"
            "Run: python scripts/integrate_data.py"
        )

    def test_integration_report_exists(self):
        assert INT_REPORT.exists(), (
            f"Integration report not found: {INT_REPORT}"
        )

    def test_integrated_has_records(self):
        df = pd.read_csv(INTEGRATED_CSV, nrows=5, low_memory=False)
        assert len(df) > 0, "Integrated dataset is empty."

    def test_collision_index_unique(self):
        df = pd.read_csv(INTEGRATED_CSV, usecols=["collision_index"], low_memory=False)
        n_dupes = df["collision_index"].duplicated().sum()
        assert n_dupes == 0, (
            f"{n_dupes} duplicate collision_index values in integrated dataset — "
            "join introduced fan-out."
        )

    def test_row_count_preserved(self):
        """Row count must equal the cleaned accidents file."""
        n_cleaned    = len(pd.read_csv(CLEANED_ACC, usecols=["collision_index"], low_memory=False))
        n_integrated = len(pd.read_csv(INTEGRATED_CSV, usecols=["collision_index"], low_memory=False))
        assert n_integrated == n_cleaned, (
            f"Row count mismatch: cleaned={n_cleaned}, integrated={n_integrated}. "
            "The join introduced or dropped records."
        )

    def test_integrated_has_weather_columns(self):
        df = pd.read_csv(INTEGRATED_CSV, nrows=1, low_memory=False)
        for col in ["wx_temp_mean_c", "wx_precip_mm", "wx_wind_max_kmh"]:
            assert col in df.columns, f"Missing weather column: {col}"

    def test_weather_match_rate_acceptable(self):
        """At least 90% of records should have weather data."""
        df = pd.read_csv(INTEGRATED_CSV, usecols=["wx_temp_mean_c"], low_memory=False)
        match_rate = df["wx_temp_mean_c"].notna().mean()
        assert match_rate >= 0.90, (
            f"Weather match rate is only {match_rate:.1%} — expected ≥ 90%."
        )

    def test_integration_report_no_fanout(self):
        report = json.loads(INT_REPORT.read_text(encoding="utf-8"))
        dupes = report.get("duplicate_collision_ids", 0)
        assert dupes == 0, f"Integration report shows {dupes} duplicate collision IDs."

    def test_target_column_present(self):
        df = pd.read_csv(INTEGRATED_CSV, nrows=1, low_memory=False)
        assert "collision_severity" in df.columns, "Target column 'collision_severity' missing."

    def test_time_features_present(self):
        df = pd.read_csv(INTEGRATED_CSV, nrows=1, low_memory=False)
        for col in ["hour_of_day", "day_of_week", "month", "is_weekend"]:
            assert col in df.columns, f"Time feature column missing: {col}"


# ═══════════════════════════════════════════════════════════════════════════
# STAGE F — Database
# ═══════════════════════════════════════════════════════════════════════════

class TestDatabase:
    """Verify the SQLite database was loaded correctly."""

    def test_db_file_exists(self):
        assert DB_FILE.exists(), (
            f"Database not found: {DB_FILE}\n"
            "Run: python scripts/load_database.py"
        )

    def test_accidents_table_populated(self):
        conn = sqlite3.connect(str(DB_FILE))
        try:
            cur = conn.execute("SELECT COUNT(*) FROM accidents")
            n = cur.fetchone()[0]
            assert n > 100_000, f"accidents table has only {n} rows."
        finally:
            conn.close()

    def test_weather_grid_table_populated(self):
        conn = sqlite3.connect(str(DB_FILE))
        try:
            cur = conn.execute("SELECT COUNT(*) FROM weather_grid")
            n = cur.fetchone()[0]
            assert n == 585 * 365, f"weather_grid has {n} rows, expected {585*365}."
        finally:
            conn.close()

    def test_lookup_table_populated(self):
        conn = sqlite3.connect(str(DB_FILE))
        try:
            cur = conn.execute("SELECT COUNT(*) FROM road_context_lookup")
            n = cur.fetchone()[0]
            assert n > 0, "road_context_lookup table is empty."
        finally:
            conn.close()

    def test_accidents_collision_index_unique(self):
        conn = sqlite3.connect(str(DB_FILE))
        try:
            cur = conn.execute(
                "SELECT COUNT(*) - COUNT(DISTINCT collision_index) FROM accidents"
            )
            dupes = cur.fetchone()[0]
            assert dupes == 0, f"{dupes} duplicate collision_index in DB accidents table."
        finally:
            conn.close()

    def test_severity_distribution_query(self):
        """Basic analytical query should return 3 severity classes."""
        conn = sqlite3.connect(str(DB_FILE))
        try:
            cur = conn.execute(
                "SELECT collision_severity, COUNT(*) FROM accidents "
                "GROUP BY collision_severity ORDER BY collision_severity"
            )
            rows = cur.fetchall()
            severities = [r[0] for r in rows]
            assert set(severities) == {1, 2, 3}, (
                f"Expected severities {{1,2,3}}, got {set(severities)}."
            )
        finally:
            conn.close()
