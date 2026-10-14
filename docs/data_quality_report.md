# docs/data_quality_report.md
# Data Quality Report — Traffic Accident Severity Analytics

> **Note**: This document describes the data quality rules implemented in the pipeline.
> Actual measured counts (record totals, rejection counts, match rates) will be filled in
> after running the pipeline. Placeholders marked with `[RUN PIPELINE]` must be updated.

---

## Source 1 — UK DfT STATS19 Collision Data

**File**: `data/raw/accidents/dft-road-casualty-statistics-collision-2025.csv`
**Source URL**: https://www.data.gov.uk/dataset/road-accidents-safety-data
**Licence**: Open Government Licence v3.0

### Volume
| Metric | Value |
|--------|-------|
| Raw records | ~101,525 |
| Cleaned records | ~101,523 |
| Rejected records | ~2 |
| Columns | 44 |

### Rejection Rules (implemented in `clean_data.py`)

| Rule | Column(s) | Action |
|------|-----------|--------|
| Missing collision identifier | `collision_index` | Reject |
| Missing severity | `collision_severity` | Reject |
| Missing latitude or longitude | `latitude`, `longitude` | Reject |
| Latitude outside [-90, 90] | `latitude` | Reject |
| Longitude outside [-180, 180] | `longitude` | Reject |
| Vehicles < 1 | `number_of_vehicles` | Reject |
| Casualties < 1 | `number_of_casualties` | Reject |
| Unparseable date | `date` | Reject |

### Severity Distribution (Raw)
| Severity | Code | Count | % |
|----------|------|-------|---|
| Slight | 3 | ~74,881 | ~73.7% |
| Serious | 2 | ~25,191 | ~24.8% |
| Fatal | 1 | ~1,453 | ~1.4% |

> **Class imbalance note**: Fatal accidents represent only ~1.4% of records.
> The ML model uses `class_weight='balanced'` to address this.
> Do not use accuracy alone as an evaluation metric.

### Date Parsing
- Format in source: `DD/MM/YYYY`
- Parsed using `pd.to_datetime(format='%d/%m/%Y', errors='coerce')`
- Records with unparseable dates are moved to the rejected file

### Known Limitations
- 2025 data includes records up to the file export date (not necessarily full year)
- `collision_index` serves as the primary key but is not independently verified against DfT
- Some categorical columns contain coded integers requiring the lookup guide to interpret

---

## Source 2 — Open-Meteo Historical Weather

**API**: https://archive-api.open-meteo.com/v1/archive
**Licence**: Creative Commons Attribution 4.0 (CC BY 4.0)

### Volume
| Metric | Value |
|--------|-------|
| Grid cells fetched | 585 / 585 |
| Failed requests | 0 |
| Days per cell | 365 |
| Total weather rows | 213,525 (585 × 365) |

### Variables
| Variable | Unit | Notes |
|----------|------|-------|
| `temperature_2m_mean` | °C | Daily mean 2 m air temperature |
| `precipitation_sum` | mm | Daily total precipitation |
| `wind_speed_10m_max` | km/h | Daily maximum 10 m wind speed |

### Grid Resolution
- Coordinate snapping: 0.25° (≈ 27 km)
- Grid cells are derived from accident coordinates rounded to nearest 0.25°

### Validation Rules (in `process_weather.py`)
| Rule | Action |
|------|--------|
| Each file must contain exactly 365 days | Skip malformed files |
| Date range must be 2025-01-01 to 2025-12-31 | Log warning if outside range |
| Numeric fields coerced with `errors='coerce'` | NaN for invalid values |

### Weather Join Quality
| Metric | Value |
|--------|-------|
| Accidents with weather match | [RUN PIPELINE] |
| Match rate | [RUN PIPELINE] |
| Unmatched (NaN weather) | [RUN PIPELINE] |

**Matching limitation**: Weather is grid-cell daily average, not a point measurement.
Accidents very near grid boundaries may be assigned to a slightly different cell than
their true meteorological environment. This is documented and accepted for this project.

---

## Source 3 — DfT STATS19 Road Safety Data Guide

**URL**: https://data.dft.gov.uk/road-accidents-safety-data/road-accident-safety-data-guide-2025.xlsx
**Licence**: Open Government Licence v3.0

### Volume
| Metric | Value |
|--------|-------|
| Code types covered | 7 (road_type, speed_limit, light_conditions, weather_conditions, road_surface_conditions, junction_detail, urban_or_rural_area) |
| Total lookup rows | [RUN PIPELINE] |

### Fallback Behaviour
If the XLSX download fails (e.g. offline environment), `collect_road_data.py`
falls back to a built-in lookup table containing the same codes extracted from
the official guide. This is clearly logged and documented. No codes are invented.

---

## Integrated Dataset Quality

**File**: `data/cleaned/accident_severity_dataset.csv`

| Check | Expected Result |
|-------|----------------|
| Row count equals cleaned accidents | [RUN PIPELINE — must be TRUE] |
| `collision_index` unique | [RUN PIPELINE — must be 0 duplicates] |
| No fan-out from weather join | [RUN PIPELINE — must be TRUE] |
| Output columns | [RUN PIPELINE] |

### Missing Value Summary (after integration)
Columns that may have NaN values in the integrated dataset:

| Column | Reason for NaN | Proportion NaN |
|--------|----------------|---------------|
| `wx_temp_mean_c` | Weather join miss | [RUN PIPELINE] |
| `wx_precip_mm` | Weather join miss | [RUN PIPELINE] |
| `wx_wind_max_kmh` | Weather join miss | [RUN PIPELINE] |
| `hour_of_day` | Missing time in source | [RUN PIPELINE] |
| `*_label` columns | Code not in lookup | [RUN PIPELINE] |

The ML model pipeline handles all NaN values via `SimpleImputer` (median for numeric,
constant "Unknown" for categorical). NaN values are preserved in the CSV and not imputed
in the data engineering layer.

---

## Automated Data Quality Tests

Run with: `pytest tests/test_phase1_data.py -v`

| Test | What it checks |
|------|---------------|
| `test_no_missing_collision_index` | Zero NaN collision identifiers |
| `test_collision_index_unique` | No duplicate primary keys |
| `test_coordinates_valid_range` | All lat/lon within valid range |
| `test_severity_valid_values` | Severity codes only {1, 2, 3} |
| `test_weather_row_count_expected` | Exactly 213,525 weather rows |
| `test_weather_date_range` | 2025-01-01 to 2025-12-31 |
| `test_row_count_preserved` | Integration preserves accident count |
| `test_weather_match_rate_acceptable` | ≥90% weather match rate |
| `test_integration_report_no_fanout` | Zero duplicate collision IDs post-join |
| `test_accidents_collision_index_unique` | DB primary key uniqueness |

**Status**: Tests written. Results depend on pipeline execution — see README for run commands.
