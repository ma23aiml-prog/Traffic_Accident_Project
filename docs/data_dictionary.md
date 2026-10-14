# docs/data_dictionary.md
# Data Dictionary — Traffic Accident Severity Analytics

## Curated Dataset: `accident_severity_dataset.csv`

This is the final integrated dataset produced by `scripts/integrate_data.py`.
It contains one row per collision record.

---

## Source Columns (UK DfT STATS19 — Source 1)

| Column | Type | Description |
|--------|------|-------------|
| `collision_index` | string | Unique collision identifier (primary key) |
| `date` | date | Date of collision (YYYY-MM-DD) |
| `time` | string | Time of collision (HH:MM, 24-hour) |
| `collision_severity` | integer | **Target variable**: 1=Fatal, 2=Serious, 3=Slight |
| `number_of_vehicles` | integer | Vehicles involved (≥1) |
| `number_of_casualties` | integer | Casualties (≥1) |
| `latitude` | float | Accident location latitude (WGS84, decimal degrees) |
| `longitude` | float | Accident location longitude (WGS84, decimal degrees) |
| `road_type` | integer | Road type code (see lookup) |
| `speed_limit` | integer | Posted speed limit in mph |
| `light_conditions` | integer | Light conditions code (see lookup) |
| `weather_conditions` | integer | Weather conditions code (see lookup) |
| `road_surface_conditions` | integer | Road surface code (see lookup) |
| `junction_detail` | integer | Junction type code (see lookup) |
| `urban_or_rural_area` | integer | 1=Urban, 2=Rural, 3=Unallocated |

> **Severity code clarification**: Code 1 = Fatal, Code 2 = Serious injury, Code 3 = Slight injury. This follows the official DfT STATS19 variable definitions. Do not assume severity 1 is the most common — it is the rarest.

---

## Engineered Time Features

| Column | Type | Description |
|--------|------|-------------|
| `hour_of_day` | float | Hour extracted from `time` column (0–23) |
| `day_of_week` | integer | 0=Monday, 1=Tuesday, …, 6=Sunday |
| `month` | integer | Calendar month (1=January, …, 12=December) |
| `year` | integer | Calendar year (2025) |
| `is_weekend` | integer | 1 if Saturday or Sunday, 0 otherwise |

---

## Weather Columns (Open-Meteo — Source 2)

Joined via nearest-cell matching: accident coordinates are rounded to the nearest 0.25° grid cell, then matched to the grid cell's daily weather observation.

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `grid_latitude` | float | degrees | Snapped grid latitude (0.25° resolution) |
| `grid_longitude` | float | degrees | Snapped grid longitude (0.25° resolution) |
| `wx_temp_mean_c` | float | °C | Daily mean temperature at 2 m |
| `wx_precip_mm` | float | mm | Daily precipitation sum |
| `wx_wind_max_kmh` | float | km/h | Daily maximum wind speed at 10 m |

> **Limitation**: Weather is a grid-cell average, not a precise point measurement at the accident location. Grid resolution is 0.25° (~27 km). Treat weather features as approximate environmental context.

---

## Road Context Decoded Labels (DfT STATS19 Lookup — Source 3)

| Column | Type | Description |
|--------|------|-------------|
| `road_type_label` | string | e.g. "Single carriageway", "Roundabout" |
| `speed_limit_label` | string | e.g. "30 mph", "70 mph" |
| `light_conditions_label` | string | e.g. "Daylight", "Darkness – lights lit" |
| `weather_conditions_label` | string | e.g. "Fine no high winds", "Raining no high winds" |
| `road_surface_conditions_label` | string | e.g. "Dry", "Wet or damp" |
| `junction_detail_label` | string | e.g. "Not at junction", "T or staggered junction" |
| `urban_or_rural_area_label` | string | "Urban", "Rural", or "Unallocated" |

---

## Road Type Lookup (STATS19 codes)

| Code | Label |
|------|-------|
| 1 | Roundabout |
| 2 | One way street |
| 3 | Dual carriageway |
| 6 | Single carriageway |
| 7 | Slip road |
| 9 | Unknown |
| 12 | One way street/Slip road |

## Light Conditions Lookup

| Code | Label |
|------|-------|
| 1 | Daylight |
| 4 | Darkness – lights lit |
| 5 | Darkness – lights unlit |
| 6 | Darkness – no lighting |
| 7 | Darkness – lighting unknown |

## Weather Conditions Lookup

| Code | Label |
|------|-------|
| 1 | Fine no high winds |
| 2 | Raining no high winds |
| 3 | Snowing no high winds |
| 4 | Fine + high winds |
| 5 | Raining + high winds |
| 6 | Snowing + high winds |
| 7 | Fog or mist |
| 8 | Other |
| 9 | Unknown |

## Road Surface Conditions

| Code | Label |
|------|-------|
| 1 | Dry |
| 2 | Wet or damp |
| 3 | Snow |
| 4 | Frost or ice |
| 5 | Flood over 3cm deep |
| 6 | Oil or diesel |
| 7 | Mud |

---

## Data Sources

| Source | Name | URL | Licence |
|--------|------|-----|---------|
| 1 | UK DfT Road Casualty Statistics (STATS19) | https://www.data.gov.uk/dataset/road-accidents-safety-data | Open Government Licence v3.0 |
| 2 | Open-Meteo Historical Weather API | https://open-meteo.com/ | CC BY 4.0 |
| 3 | DfT STATS19 Road Safety Data Guide 2025 | https://data.dft.gov.uk/road-accidents-safety-data/road-accident-safety-data-guide-2025.xlsx | Open Government Licence v3.0 |
