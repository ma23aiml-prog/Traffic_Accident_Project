"""
collect_road_data.py
---------------------
Download the UK Department for Transport STATS19 road safety data guide
(lookup tables) and extract road-type and speed-limit code mappings.

Source (Third data source):
    UK DfT Road Safety Open Data — STATS19 Data Guide (2024 edition)
    URL: https://data.dft.gov.uk/road-accidents-safety-data/
         road-accident-safety-data-guide-2025.xlsx
    Retrieved: on script execution
    Licence: Open Government Licence v3.0

Outputs:
    data/raw/road_context/road_accident_safety_data_guide_2025.xlsx
    data/raw/road_context/source_metadata.json
    data/cleaned/road_context_lookup.csv
        Columns: code_type, code_value, label

The lookup CSV maps every integer code in the accident data to a human-
readable label. The integration script will use this to decode:
    - road_type         (1=Roundabout, 2=One way street, 3=Dual carriageway, …)
    - speed_limit       (20, 30, 40, 50, 60, 70 mph)
    - light_conditions  (1=Daylight, 4=Darkness – lights lit, …)
    - weather_conditions (codes 1–9)
    - road_surface_conditions (codes 1–5)
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

# ── Paths ─────────────────────────────────────────────────────────────────
RAW_DIR    = Path("data/raw/road_context")
CLEAN_DIR  = Path("data/cleaned")
LOG_DIR    = Path("logs")

RAW_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

XLSX_FILE    = RAW_DIR / "road_accident_safety_data_guide_2025.xlsx"
METADATA_FILE = RAW_DIR / "source_metadata.json"
OUTPUT_FILE  = CLEAN_DIR / "road_context_lookup.csv"

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Source URL ────────────────────────────────────────────────────────────
SOURCE_URL = (
    "https://data.dft.gov.uk/road-accidents-safety-data/"
    "road-accident-safety-data-guide-2025.xlsx"
)

# ── Fallback bundled lookup table ─────────────────────────────────────────
# If the download fails the script falls back to this minimal table so that
# the rest of the pipeline can still run.  These codes are taken directly
# from the STATS19 data guide and are NOT invented.
FALLBACK_LOOKUP = [
    # road_type codes
    ("road_type", 1,  "Roundabout"),
    ("road_type", 2,  "One way street"),
    ("road_type", 3,  "Dual carriageway"),
    ("road_type", 6,  "Single carriageway"),
    ("road_type", 7,  "Slip road"),
    ("road_type", 9,  "Unknown"),
    ("road_type", 12, "One way street/Slip road"),
    # speed_limit — these are actual mph values, codes equal the limit
    ("speed_limit", 20, "20 mph"),
    ("speed_limit", 30, "30 mph"),
    ("speed_limit", 40, "40 mph"),
    ("speed_limit", 50, "50 mph"),
    ("speed_limit", 60, "60 mph"),
    ("speed_limit", 70, "70 mph"),
    # light_conditions
    ("light_conditions", 1, "Daylight"),
    ("light_conditions", 4, "Darkness – lights lit"),
    ("light_conditions", 5, "Darkness – lights unlit"),
    ("light_conditions", 6, "Darkness – no lighting"),
    ("light_conditions", 7, "Darkness – lighting unknown"),
    # weather_conditions
    ("weather_conditions", 1, "Fine no high winds"),
    ("weather_conditions", 2, "Raining no high winds"),
    ("weather_conditions", 3, "Snowing no high winds"),
    ("weather_conditions", 4, "Fine + high winds"),
    ("weather_conditions", 5, "Raining + high winds"),
    ("weather_conditions", 6, "Snowing + high winds"),
    ("weather_conditions", 7, "Fog or mist"),
    ("weather_conditions", 8, "Other"),
    ("weather_conditions", 9, "Unknown"),
    # road_surface_conditions
    ("road_surface_conditions", 1, "Dry"),
    ("road_surface_conditions", 2, "Wet or damp"),
    ("road_surface_conditions", 3, "Snow"),
    ("road_surface_conditions", 4, "Frost or ice"),
    ("road_surface_conditions", 5, "Flood over 3cm deep"),
    ("road_surface_conditions", 6, "Oil or diesel"),
    ("road_surface_conditions", 7, "Mud"),
    # junction_detail
    ("junction_detail", 0, "Not at junction or within 20 metres"),
    ("junction_detail", 1, "Roundabout"),
    ("junction_detail", 2, "Mini-roundabout"),
    ("junction_detail", 3, "T or staggered junction"),
    ("junction_detail", 5, "Slip road"),
    ("junction_detail", 6, "Crossroads"),
    ("junction_detail", 7, "More than 4 arms (not roundabout)"),
    ("junction_detail", 8, "Private drive or entrance"),
    ("junction_detail", 9, "Other junction"),
    # urban_or_rural_area
    ("urban_or_rural_area", 1, "Urban"),
    ("urban_or_rural_area", 2, "Rural"),
    ("urban_or_rural_area", 3, "Unallocated"),
]


def download_xlsx() -> bool:
    """Attempt to download the DfT guide. Returns True on success."""
    if XLSX_FILE.exists():
        log.info("XLSX already exists at %s, skipping download.", XLSX_FILE)
        return True

    log.info("Downloading DfT STATS19 data guide from:\n  %s", SOURCE_URL)
    try:
        resp = requests.get(SOURCE_URL, timeout=120, stream=True)
        resp.raise_for_status()
        total = 0
        with XLSX_FILE.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=65536):
                fh.write(chunk)
                total += len(chunk)
        log.info("Downloaded %.1f KB to %s", total / 1024, XLSX_FILE)
        return True
    except requests.RequestException as exc:
        log.warning("Download failed: %s", exc)
        return False


def parse_xlsx_lookup() -> pd.DataFrame | None:
    """
    Parse the DfT guide XLSX.
    The guide contains sheets named after variable names.
    Each sheet has columns like: code | label (exact names vary by year).
    Returns a unified lookup DataFrame or None if parsing fails.
    """
    try:
        xl = pd.ExcelFile(XLSX_FILE, engine="openpyxl")
    except Exception as exc:
        log.warning("Cannot open XLSX: %s", exc)
        return None

    # Sheets we care about and the column order they typically have
    target_sheets = {
        "Road Type":              "road_type",
        "Road Surface Conditions": "road_surface_conditions",
        "Weather Conditions":     "weather_conditions",
        "Light Conditions":       "light_conditions",
        "Junction Detail":        "junction_detail",
        "Urban Rural":            "urban_or_rural_area",
    }

    rows = []
    for sheet_name, code_type in target_sheets.items():
        # Try exact and case-insensitive match
        matched = [
            s for s in xl.sheet_names
            if s.strip().lower() == sheet_name.lower()
        ]
        if not matched:
            log.debug("Sheet not found: %s", sheet_name)
            continue

        df_sheet = xl.parse(matched[0], header=0)
        df_sheet.columns = [str(c).strip().lower() for c in df_sheet.columns]

        # Find code and label columns heuristically
        code_col  = next((c for c in df_sheet.columns if "code" in c), None)
        label_col = next(
            (c for c in df_sheet.columns if any(
                w in c for w in ("label", "description", "name")
            )),
            None,
        )

        if code_col is None or label_col is None:
            log.debug("Cannot identify code/label columns in sheet: %s", sheet_name)
            continue

        for _, row in df_sheet.iterrows():
            code  = row[code_col]
            label = row[label_col]
            if pd.notna(code) and pd.notna(label):
                try:
                    rows.append((code_type, int(float(code)), str(label).strip()))
                except (ValueError, TypeError):
                    pass

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=["code_type", "code_value", "label"])
    df.drop_duplicates(inplace=True)
    log.info("Parsed %d lookup rows from XLSX", len(df))
    return df


def use_fallback() -> pd.DataFrame:
    log.info("Using built-in fallback lookup table (%d rows).", len(FALLBACK_LOOKUP))
    df = pd.DataFrame(FALLBACK_LOOKUP, columns=["code_type", "code_value", "label"])
    return df


def main() -> None:
    # ── Try to download the XLSX ───────────────────────────────────────────
    downloaded = download_xlsx()
    xlsx_parsed = False
    lookup_df: pd.DataFrame | None = None

    if downloaded and XLSX_FILE.exists():
        lookup_df = parse_xlsx_lookup()
        if lookup_df is not None:
            xlsx_parsed = True

    # ── Fall back to embedded table if needed ──────────────────────────────
    if lookup_df is None or lookup_df.empty:
        log.warning(
            "Could not parse XLSX. Using built-in lookup table as fallback. "
            "The pipeline will still run correctly."
        )
        lookup_df = use_fallback()

    # ── Also add speed_limit (these are not coded in the guide — values = mph)
    if "speed_limit" not in lookup_df["code_type"].values:
        speed_rows = pd.DataFrame(
            [("speed_limit", v, f"{v} mph") for v in [20, 30, 40, 50, 60, 70]],
            columns=["code_type", "code_value", "label"],
        )
        lookup_df = pd.concat([lookup_df, speed_rows], ignore_index=True)

    lookup_df.sort_values(["code_type", "code_value"], inplace=True)
    lookup_df.reset_index(drop=True, inplace=True)
    lookup_df.to_csv(OUTPUT_FILE, index=False)

    # ── Source metadata ────────────────────────────────────────────────────
    metadata = {
        "source_name":       "UK DfT Road Safety Open Data — STATS19 Data Guide 2025",
        "source_url":        SOURCE_URL,
        "licence":           "Open Government Licence v3.0",
        "retrieval_date":    datetime.now(timezone.utc).isoformat(),
        "raw_file":          str(XLSX_FILE),
        "lookup_file":       str(OUTPUT_FILE),
        "lookup_rows":       len(lookup_df),
        "source_downloaded": downloaded,
        "xlsx_parsed":       xlsx_parsed,
        "used_fallback":     not xlsx_parsed,
    }
    METADATA_FILE.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n========== ROAD CONTEXT COLLECTION SUMMARY ==========")
    print(f"Source URL    : {SOURCE_URL}")
    print(f"XLSX downloaded : {downloaded}")
    print(f"XLSX parsed OK  : {xlsx_parsed}")
    print(f"Used fallback   : {not xlsx_parsed}")
    print(f"Lookup rows     : {len(lookup_df)}")
    print(f"Code types      : {lookup_df['code_type'].unique().tolist()}")
    print(f"\nRaw file      : {XLSX_FILE}")
    print(f"Lookup file   : {OUTPUT_FILE}")
    print(f"Metadata file : {METADATA_FILE}")


if __name__ == "__main__":
    main()
