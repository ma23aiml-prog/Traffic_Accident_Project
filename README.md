# Traffic Accident Severity Analytics and Prediction

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange)](https://scikit-learn.org/)
[![MLflow](https://img.shields.io/badge/tracking-MLflow-blue)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-green)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-red)](https://streamlit.io/)

End-to-end data engineering and machine learning pipeline for UK road accident
severity prediction, built on DfT STATS19 2025 data, Open-Meteo weather, and
DfT road context lookup tables.

---

## Project Structure

```
Traffic_Accident_Project/
├── data/
│   ├── raw/
│   │   ├── accidents/          # DfT STATS19 CSV (source 1) — already downloaded
│   │   ├── weather/            # 585 Open-Meteo JSON files (source 2) — already downloaded
│   │   └── road_context/       # DfT STATS19 guide XLSX (source 3)
│   ├── staging/                # Grid locations, test split
│   ├── cleaned/                # Processed CSVs, integrated dataset
│   └── rejected/               # Rejected records with rejection reasons
├── scripts/
│   ├── inspect_data.py         # Quick data inspection
│   ├── check_data_quality.py   # Pre-cleaning quality check
│   ├── clean_data.py           # Accident data cleaning & validation
│   ├── prepare_weather.py      # Generate weather grid locations
│   ├── fetch_weather.py        # Download weather data (ALREADY DONE — do not re-run)
│   ├── process_weather.py      # Consolidate 585 JSONs → weather_processed.csv
│   ├── collect_road_data.py    # Download/build road context lookup table
│   ├── integrate_data.py       # Join all 3 sources → accident_severity_dataset.csv
│   ├── load_database.py        # Load curated data into SQLite database
│   ├── train_model.py          # Train Random Forest + MLflow tracking
│   ├── evaluate_model.py       # Confusion matrix, ROC, feature importance
│   └── drift_monitor.py        # KS + Chi² drift detection
├── dashboard/
│   └── app.py                  # Streamlit analytics dashboard
├── api/
│   ├── __init__.py
│   ├── main.py                 # FastAPI prediction service
│   └── schemas.py              # Pydantic request/response schemas
├── models/
│   ├── severity_model.pkl      # Trained sklearn Pipeline [after training]
│   ├── metadata.json           # Training metadata + metrics [after training]
│   └── evaluation/             # Confusion matrix, ROC curves, feature importance
├── mlruns/                     # MLflow experiment tracking [after training]
├── sql/
│   ├── schema.sql              # Database schema (SQLite + PostgreSQL compatible)
│   └── analytics_queries.sql  # 12 analytical SQL queries
├── tests/
│   ├── conftest.py             # Pytest sys.path config
│   ├── test_phase1_data.py     # Phase 1 data quality tests (29 tests)
│   └── test_phase2_model.py    # Phase 2 model + API tests (18 tests)
├── docs/
│   ├── architecture.md         # System architecture + Mermaid diagram
│   ├── data_dictionary.md      # Column descriptions for all datasets
│   ├── data_quality_report.md  # Validation rules and quality metrics
│   ├── model_evaluation.md     # Model design and evaluation methodology
│   ├── monitoring.md           # Drift monitoring strategy
│   └── report_outline.md       # College report structure guide
├── logs/                       # Pipeline logs, reports, drift output
├── Dockerfile                  # Container for FastAPI service
├── docker-compose.yml          # API + Dashboard containers
├── requirements.txt            # All Python dependencies (pinned versions)
├── conftest.py                 # Root pytest configuration
├── .env.example                # Environment variable template
└── .gitignore
```

---

## Quick Start

### Prerequisites
- Python 3.11 (check: `python --version`)
- Virtual environment `.venv` (already created)
- DfT STATS19 CSV in `data/raw/accidents/`
- 585 weather JSON files in `data/raw/weather/` (already downloaded)

### Step 0 — Activate your virtual environment

```powershell
# From the project root directory
Set-Location "C:\Users\pjant\OneDrive\Desktop\Traffic_Accident_Project"
.venv\Scripts\Activate.ps1
```

---

## Phase 1 — Data Engineering & Analytics

Run these steps **in order**. Do NOT re-run `fetch_weather.py`.

### Step 1 — Install dependencies

```powershell
pip install -r requirements.txt
```

### Step 2 — Inspect raw accident data

```powershell
python scripts/inspect_data.py
python scripts/check_data_quality.py
```

### Step 3 — Clean accident data

```powershell
python scripts/clean_data.py
```

**Expected output:**
```
Loaded: 101,525 records
Rejected: ~2 records (see data/rejected/)
Cleaned: ~101,523 records → data/cleaned/accidents_cleaned.csv
```

### Step 4 — Process weather data (consolidate 585 JSON files)

```powershell
python scripts/process_weather.py
```

**Expected output:**
```
Processed 585 JSON files
Total rows: 213,525 (585 cells × 365 days)
Saved: data/cleaned/weather_processed.csv
```

### Step 5 — Download road context lookup table

```powershell
python scripts/collect_road_data.py
```

**Expected output:**
```
Downloaded: data/raw/road_context/road-accident-safety-data-guide-2025.xlsx
Generated: data/cleaned/road_context_lookup.csv
```

> **Note**: If the download fails (offline environment), the script falls back
> to a built-in lookup table. You will see a warning — this is expected and safe.

### Step 6 — Integrate all three sources

```powershell
python scripts/integrate_data.py
```

**Expected output:**
```
Accidents input              : 101,523
Accidents after weather join : 101,523   (must equal input — no fan-out)
Duplicate collision IDs      :       0   (must be 0)
Weather matched              : ~101,XXX  (XX.X%)
Final record count           : 101,523
Output: data/cleaned/accident_severity_dataset.csv
```

> **CRITICAL CHECK**: If "Duplicate collision IDs" is not 0, there is a data
> integrity problem. Stop and check the weather CSV for duplicate (lat, lon, date) keys.

### Step 7 — Load database

```powershell
python scripts/load_database.py
```

**Expected output:**
```
accidents                    101,523 rows
weather_grid                 213,525 rows
road_context_lookup          XX rows
collision_index uniqueness: PASS
Database file: data/traffic_accidents.db
```

### Step 8 — Run SQL analytics queries

```powershell
# Install sqlite3 CLI if needed, or view with DB Browser for SQLite
# To run all queries and view output in terminal:
sqlite3 data/traffic_accidents.db ".read sql/analytics_queries.sql"
```

### Step 9 — Run Phase 1 tests

```powershell
pytest tests/test_phase1_data.py -v
```

**Expected**: All 29 tests pass. Fix any failures before continuing to Phase 2.

### Step 10 — Launch Streamlit dashboard

```powershell
streamlit run dashboard/app.py
```

Then open: **http://localhost:8501**

---

## Phase 2 — Machine Learning & MLOps

Run Phase 2 **only after Phase 1 is complete** and `accident_severity_dataset.csv` exists.

### Step 11 — Train the model

```powershell
python scripts/train_model.py
```

**Expected output:**
```
Training Random Forest (200 trees) …
Training complete.
Accuracy  : 0.XXXX
Macro-F1  : 0.XXXX
[Full classification report printed]
Model saved: models/severity_model.pkl
Metadata  : models/metadata.json
MLflow UI : mlflow ui --backend-store-uri <path>/mlruns
Test split: data/staging/test_split.csv
```

> **Note**: Training ~101,523 records with 200 trees may take 5–15 minutes
> depending on your CPU. This is expected.

### Step 12 — Evaluate the model

```powershell
python scripts/evaluate_model.py
```

**Expected output:**
- `models/evaluation/classification_report.json` — per-class metrics
- `models/evaluation/classification_report.txt` — human-readable
- `models/evaluation/confusion_matrix.png` — heatmap
- `models/evaluation/feature_importance.csv` — ranked features
- `models/evaluation/feature_importance.png` — top-20 plot
- `models/evaluation/roc_curves.png` — one-vs-rest ROC

### Step 13 — View MLflow experiment tracking

```powershell
# In a new PowerShell window (keep this running):
mlflow ui --backend-store-uri mlruns
```

Then open: **http://localhost:5000**

### Step 14 — Run drift monitoring

```powershell
python scripts/drift_monitor.py
```

**Expected output:**
- `logs/drift_report.json` — full per-feature drift statistics
- `logs/drift_summary.txt` — human-readable with drift flags

### Step 15 — Start the FastAPI prediction service

```powershell
# In a new PowerShell window:
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Then open:
- **http://localhost:8000/docs** — Swagger UI (interactive API docs)
- **http://localhost:8000/health** — Model health check

### Step 16 — Test a prediction

```powershell
# Using PowerShell (Invoke-WebRequest):
$body = @{
    hour_of_day = 8
    day_of_week = 1
    month = 3
    is_weekend = 0
    number_of_vehicles = 2
    number_of_casualties = 1
    speed_limit = 30
    wx_temp_mean_c = 8.5
    wx_precip_mm = 2.3
    wx_wind_max_kmh = 25.0
    road_type_label = "Single carriageway"
    light_conditions_label = "Daylight"
    weather_conditions_label = "Raining no high winds"
    road_surface_conditions_label = "Wet or damp"
    junction_detail_label = "T or staggered junction"
    urban_or_rural_area_label = "Urban"
} | ConvertTo-Json

Invoke-WebRequest -Uri "http://localhost:8000/predict" `
    -Method POST `
    -ContentType "application/json" `
    -Body $body | Select-Object -ExpandProperty Content
```

**Expected response:**
```json
{
  "prediction": {
    "predicted_class": 3,
    "predicted_label": "Slight",
    "probability_fatal": 0.03,
    "probability_serious": 0.18,
    "probability_slight": 0.79
  },
  "model_version": "2025-..."
}
```

### Step 17 — Run Phase 2 tests

```powershell
# Tests that do NOT require the running API (model file tests):
pytest tests/test_phase2_model.py -v -k "not TestAPI"

# Full tests including API (requires uvicorn running in another window):
pytest tests/test_phase2_model.py -v
```

### Step 18 — Run all tests together

```powershell
pytest tests/ -v --tb=short 2>&1 | Tee-Object -FilePath logs/test_results.txt
Get-Content logs/test_results.txt
```

---

## Docker Deployment

> Requires Docker Desktop installed and running.

```powershell
# Build the API image:
docker build -t accident-api .

# Run API only (mount trained model):
docker run -p 8000:8000 -v "${PWD}/models:/app/models" accident-api

# Run both API + Dashboard:
docker compose up --build
```

Access:
- API: http://localhost:8000
- Dashboard: http://localhost:8501

---

## Data Sources

| # | Source | URL | Licence |
|---|--------|-----|---------|
| 1 | UK DfT STATS19 Collision Data | https://www.data.gov.uk/dataset/road-accidents-safety-data | OGL v3.0 |
| 2 | Open-Meteo Historical Weather API | https://open-meteo.com/ | CC BY 4.0 |
| 3 | DfT STATS19 Road Safety Data Guide | https://data.dft.gov.uk/road-accidents-safety-data/ | OGL v3.0 |

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `FileNotFoundError: accidents_cleaned.csv` | Run `clean_data.py` first |
| `FileNotFoundError: accident_severity_dataset.csv` | Run `integrate_data.py` first |
| `FileNotFoundError: severity_model.pkl` | Run `train_model.py` first |
| `FileNotFoundError: test_split.csv` | Run `train_model.py` first |
| Weather match rate < 90% | Check `process_weather.py` output; verify 585 JSON files present |
| Duplicate collision_index in integration | Check weather CSV for duplicate (lat,lon,date) rows |
| `sparse_output` error (scikit-learn) | Ensure scikit-learn ≥ 1.2 (`pip install -U scikit-learn`) |
| `uvicorn: command not found` | Run `pip install uvicorn` |

---

## Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | 3.11 |
| Data processing | pandas, numpy | 2.x / 1.x |
| Machine learning | scikit-learn | ≥ 1.4 |
| Experiment tracking | MLflow | ≥ 2.x |
| Dashboard | Streamlit | ≥ 1.x |
| API | FastAPI + Uvicorn | ≥ 0.115 |
| Schema validation | Pydantic | v2.x |
| Testing | pytest | ≥ 8 |
| Visualisation | plotly, matplotlib | latest |
| Database | SQLite (built-in) | 3.x |
| Containerisation | Docker + Compose | latest |

---

## Licence

Project code: MIT Licence (see LICENSE file).

Data licences:
- DfT STATS19 data: Open Government Licence v3.0
- Open-Meteo weather: Creative Commons Attribution 4.0

---

## Acknowledgements

- UK Department for Transport for publishing STATS19 accident data openly
- Open-Meteo for providing free historical weather data
- scikit-learn, MLflow, FastAPI, and Streamlit open-source communities
