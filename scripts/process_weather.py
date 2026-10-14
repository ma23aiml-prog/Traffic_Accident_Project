"""
process_weather.py
------------------
Consolidate all 585 Open-Meteo weather JSON files into a single flat CSV.

Output: data/cleaned/weather_processed.csv
Columns:
    grid_latitude   float   Snapped grid latitude  (0.25° resolution)
    grid_longitude  float   Snapped grid longitude (0.25° resolution)
    date            date    Observation date (YYYY-MM-DD)
    temp_mean_c     float   Daily mean air temperature at 2 m (°C)
    precip_mm       float   Daily precipitation sum (mm)
    wind_max_kmh    float   Daily maximum wind speed at 10 m (km/h)

Each JSON contains 365 rows (2025-01-01 to 2025-12-31).  Files are named
weather_<lat>_<lon>.json and were downloaded by fetch_weather.py.
"""

import json
import re
import logging
from pathlib import Path

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────
RAW_DIR    = Path("data/raw/weather")
OUTPUT_DIR = Path("data/cleaned")
LOG_DIR    = Path("logs")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE   = OUTPUT_DIR / "weather_processed.csv"
REPORT_FILE   = LOG_DIR   / "weather_processing_report.json"

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

EXPECTED_DAYS  = 365
START_DATE     = "2025-01-01"
END_DATE       = "2025-12-31"

# Regex to extract lat / lon from the filename
FILENAME_RE = re.compile(
    r"^weather_(-?\d+\.\d+)_(-?\d+\.\d+)\.json$"
)


def parse_weather_file(path: Path) -> pd.DataFrame | None:
    """
    Parse a single Open-Meteo JSON file into a DataFrame.
    Returns None and logs a warning if the file is invalid.
    """
    match = FILENAME_RE.match(path.name)
    if not match:
        log.warning("Skipping unexpected filename: %s", path.name)
        return None

    grid_lat = float(match.group(1))
    grid_lon = float(match.group(2))

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Cannot read %s: %s", path.name, exc)
        return None

    daily = raw.get("daily", {})
    times = daily.get("time", [])

    if len(times) != EXPECTED_DAYS:
        log.warning(
            "%s: expected %d days, found %d. Skipping.",
            path.name, EXPECTED_DAYS, len(times),
        )
        return None

    df = pd.DataFrame({
        "grid_latitude":  grid_lat,
        "grid_longitude": grid_lon,
        "date":           pd.to_datetime(times),
        "temp_mean_c":    daily.get("temperature_2m_mean"),
        "precip_mm":      daily.get("precipitation_sum"),
        "wind_max_kmh":   daily.get("wind_speed_10m_max"),
    })

    # Basic range validation — flag but do not discard; they may be valid extremes
    numeric_cols = ["temp_mean_c", "precip_mm", "wind_max_kmh"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def main() -> None:
    log.info("Scanning %s for weather JSON files …", RAW_DIR)

    json_files = sorted(
        p for p in RAW_DIR.glob("weather_*.json")
        if p.name != "weather_api_sample.json"
    )
    log.info("Found %d JSON files", len(json_files))

    frames      = []
    skipped     = []
    parsed_ok   = 0

    for path in json_files:
        df = parse_weather_file(path)
        if df is not None:
            frames.append(df)
            parsed_ok += 1
        else:
            skipped.append(path.name)

    if not frames:
        raise RuntimeError(
            "No valid weather files were parsed. "
            "Check that data/raw/weather/ contains the JSON downloads."
        )

    log.info("Combining %d parsed files …", parsed_ok)
    combined = pd.concat(frames, ignore_index=True)

    # Ensure date column is properly typed
    combined["date"] = pd.to_datetime(combined["date"])

    # Sort for deterministic output
    combined.sort_values(
        ["grid_latitude", "grid_longitude", "date"], inplace=True
    )
    combined.reset_index(drop=True, inplace=True)

    # Validation summary
    missing_temp    = combined["temp_mean_c"].isna().sum()
    missing_precip  = combined["precip_mm"].isna().sum()
    missing_wind    = combined["wind_max_kmh"].isna().sum()
    date_min        = combined["date"].min().date()
    date_max        = combined["date"].max().date()

    combined.to_csv(OUTPUT_FILE, index=False)
    log.info("Saved: %s  (%d rows)", OUTPUT_FILE, len(combined))

    report = {
        "files_found":         len(json_files),
        "files_parsed_ok":     parsed_ok,
        "files_skipped":       len(skipped),
        "skipped_files":       skipped,
        "total_rows":          len(combined),
        "unique_grid_cells":   int(combined[["grid_latitude", "grid_longitude"]].drop_duplicates().shape[0]),
        "date_range_min":      str(date_min),
        "date_range_max":      str(date_max),
        "missing_temp_mean":   int(missing_temp),
        "missing_precip_sum":  int(missing_precip),
        "missing_wind_max":    int(missing_wind),
    }

    import json as _json
    REPORT_FILE.write_text(_json.dumps(report, indent=2), encoding="utf-8")
    log.info("Report: %s", REPORT_FILE)

    print("\n========== WEATHER PROCESSING SUMMARY ==========")
    print(f"JSON files found      : {len(json_files)}")
    print(f"Files parsed OK       : {parsed_ok}")
    print(f"Files skipped         : {len(skipped)}")
    print(f"Total rows output     : {len(combined):,}")
    print(f"Unique grid cells     : {report['unique_grid_cells']}")
    print(f"Date range            : {date_min} to {date_max}")
    print(f"Missing temp values   : {missing_temp}")
    print(f"Missing precip values : {missing_precip}")
    print(f"Missing wind values   : {missing_wind}")
    print(f"\nOutput file  : {OUTPUT_FILE}")
    print(f"Report file  : {REPORT_FILE}")


if __name__ == "__main__":
    main()
