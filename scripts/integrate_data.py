"""
integrate_data.py
-----------------
Join accidents (source 1) + weather (source 2) + road-context labels (source 3)
into a single accident-level curated dataset.

Output: data/cleaned/accident_severity_dataset.csv

Join strategy:
    Weather: Each accident's lat/lon is snapped to the nearest 0.25° grid cell
             (same grid used to collect the weather data).  The weather for
             that grid cell on the same calendar date is joined.

    Road context: Integer codes in the accident data (road_type, speed_limit,
                  light_conditions, weather_conditions, road_surface_conditions,
                  junction_detail, urban_or_rural_area) are decoded using the
                  lookup table downloaded by collect_road_data.py.

Guarantees:
    - Row count equals the cleaned accidents dataset (no fan-out from joins).
    - collision_index is unique in the output.
    - Unmatched weather rows receive NaN, not a fallback value.
    - Unmatched lookup codes are left as the original integer.
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────
ACCIDENTS_FILE  = Path("data/cleaned/accidents_cleaned.csv")
WEATHER_FILE    = Path("data/cleaned/weather_processed.csv")
LOOKUP_FILE     = Path("data/cleaned/road_context_lookup.csv")
OUTPUT_FILE     = Path("data/cleaned/accident_severity_dataset.csv")
REPORT_FILE     = Path("logs/integration_report.json")
LOG_DIR         = Path("logs")

LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Weather grid resolution — must match prepare_weather.py / fetch_weather.py
GRID_RES = 0.25

# Columns to decode with lookup table
LOOKUP_COLS = [
    "road_type",
    "speed_limit",
    "light_conditions",
    "weather_conditions",
    "road_surface_conditions",
    "junction_detail",
    "urban_or_rural_area",
]


def snap_to_grid(series: pd.Series, resolution: float = GRID_RES) -> pd.Series:
    """Round a coordinate series to the nearest grid cell centre."""
    return (series / resolution).round() * resolution


def build_lookup_map(lookup_df: pd.DataFrame, code_type: str) -> dict:
    """Return {int_code: str_label} for a given code type."""
    subset = lookup_df[lookup_df["code_type"] == code_type]
    return dict(zip(subset["code_value"], subset["label"]))


def main() -> None:
    # ── 1. Load accidents ────────────────────────────────────────────────
    log.info("Loading accidents …")
    acc = pd.read_csv(ACCIDENTS_FILE, low_memory=False)
    n_accidents_start = len(acc)
    log.info("  %d accident records", n_accidents_start)

    # Parse date column
    acc["date"] = pd.to_datetime(acc["date"], errors="coerce")

    # ── 2. Feature engineering: time-based features ───────────────────────
    log.info("Engineering time features …")
    acc["hour_of_day"]   = pd.to_numeric(acc.get("time", pd.Series()), errors="coerce")
    # Extract hour from time column (format HH:MM)
    if "time" in acc.columns:
        acc["hour_of_day"] = acc["time"].astype(str).str.split(":").str[0].apply(
            lambda x: int(x) if x.isdigit() else np.nan
        )
    acc["day_of_week"]   = acc["date"].dt.dayofweek   # 0=Mon … 6=Sun
    acc["month"]         = acc["date"].dt.month
    acc["year"]          = acc["date"].dt.year
    acc["is_weekend"]    = acc["day_of_week"].isin([5, 6]).astype(int)

    # ── 3. Snap accidents to weather grid ─────────────────────────────────
    log.info("Snapping accident coordinates to weather grid …")
    acc["grid_latitude"]  = snap_to_grid(acc["latitude"])
    acc["grid_longitude"] = snap_to_grid(acc["longitude"])

    # ── 4. Load weather ───────────────────────────────────────────────────
    log.info("Loading weather data …")
    weather = pd.read_csv(WEATHER_FILE, parse_dates=["date"])
    log.info("  %d weather rows (%d grid cells)",
             len(weather),
             weather[["grid_latitude", "grid_longitude"]].drop_duplicates().shape[0])

    # ── 5. Join weather ───────────────────────────────────────────────────
    log.info("Joining weather on (grid_lat, grid_lon, date) …")
    weather_join = weather.rename(columns={
        "temp_mean_c":  "wx_temp_mean_c",
        "precip_mm":    "wx_precip_mm",
        "wind_max_kmh": "wx_wind_max_kmh",
    })

    acc = acc.merge(
        weather_join,
        on=["grid_latitude", "grid_longitude", "date"],
        how="left",
    )

    n_weather_matched   = acc["wx_temp_mean_c"].notna().sum()
    n_weather_unmatched = acc["wx_temp_mean_c"].isna().sum()
    log.info("  Weather matched   : %d", n_weather_matched)
    log.info("  Weather unmatched : %d", n_weather_unmatched)

    # ── 6. Validate no fan-out ────────────────────────────────────────────
    n_after_weather = len(acc)
    if n_after_weather != n_accidents_start:
        log.warning(
            "Row count changed after weather join: %d → %d. "
            "Check for duplicate grid/date keys in weather data.",
            n_accidents_start, n_after_weather,
        )
        # De-duplicate keeping first occurrence per collision_index
        acc.drop_duplicates(subset="collision_index", keep="first", inplace=True)
        log.info("  De-duplicated to %d rows.", len(acc))

    # ── 7. Load road-context lookup ───────────────────────────────────────
    log.info("Loading road-context lookup …")
    lookup = pd.read_csv(LOOKUP_FILE)
    log.info("  %d lookup rows, code types: %s",
             len(lookup), lookup["code_type"].unique().tolist())

    # ── 8. Decode coded columns ───────────────────────────────────────────
    log.info("Decoding road-context coded columns …")
    for col in LOOKUP_COLS:
        if col not in acc.columns:
            log.debug("  Column %s not in accident data — skipping.", col)
            continue

        mapping = build_lookup_map(lookup, col)
        if not mapping:
            log.debug("  No lookup entries for %s — skipping.", col)
            continue

        label_col = col + "_label"
        acc[col]       = pd.to_numeric(acc[col], errors="coerce")
        acc[label_col] = acc[col].map(mapping)

        matched   = acc[label_col].notna().sum()
        unmatched = acc[label_col].isna().sum()
        log.info("  %-30s  matched=%d  unmatched=%d", col, matched, unmatched)

    # ── 9. Final uniqueness check ─────────────────────────────────────────
    n_final = len(acc)
    n_dupes = acc["collision_index"].duplicated().sum()
    if n_dupes > 0:
        log.error("DUPLICATE collision_index values found: %d", n_dupes)
    else:
        log.info("collision_index uniqueness check: PASS (%d unique records)", n_final)

    # ── 10. Save output ───────────────────────────────────────────────────
    acc.to_csv(OUTPUT_FILE, index=False)
    log.info("Saved: %s  (%d rows, %d columns)", OUTPUT_FILE, n_final, acc.shape[1])

    # ── 11. Integration report ────────────────────────────────────────────
    report = {
        "timestamp":                    datetime.now(timezone.utc).isoformat(),
        "accidents_input":              n_accidents_start,
        "accidents_after_weather_join": n_after_weather,
        "accidents_final":              n_final,
        "duplicate_collision_ids":      int(n_dupes),
        "weather_matched":              int(n_weather_matched),
        "weather_unmatched":            int(n_weather_unmatched),
        "weather_match_pct":            round(100 * n_weather_matched / n_final, 2),
        "output_columns":               acc.columns.tolist(),
    }
    REPORT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n========== INTEGRATION REPORT ==========")
    print(f"Accidents input              : {n_accidents_start:>8,}")
    print(f"Accidents after weather join : {n_after_weather:>8,}")
    print(f"Duplicate collision IDs      : {n_dupes:>8,}")
    print(f"Weather matched              : {n_weather_matched:>8,}  "
          f"({report['weather_match_pct']:.1f}%)")
    print(f"Weather unmatched            : {n_weather_unmatched:>8,}")
    print(f"Final record count           : {n_final:>8,}")
    print(f"Output columns               : {acc.shape[1]}")
    print(f"\nOutput : {OUTPUT_FILE}")
    print(f"Report : {REPORT_FILE}")


if __name__ == "__main__":
    main()
