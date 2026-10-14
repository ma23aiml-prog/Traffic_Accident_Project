# docs/report_outline.md
# College Report Outline — Traffic Accident Severity Analytics

**Suggested length**: 8–10 pages (excluding appendices)
**Report title**: Traffic Accident Severity Analytics and Prediction: An End-to-End
Data Engineering and Machine Learning Pipeline for UK Road Collision Data

---

## 1. Introduction (0.5 page)

- Background: Road traffic accidents are a major public health concern in the UK.
  The Department for Transport records approximately 100,000+ collisions annually
  through the STATS19 reporting system.
- Problem statement: Can we build a reproducible data pipeline to integrate
  multi-source accident data and train a classifier to predict accident severity?
- Objectives: (1) Ingest and integrate three data sources, (2) store in a
  queryable database, (3) build an interactive analytics dashboard, (4) train a
  severity prediction model, (5) deploy via REST API with experiment tracking.
- Scope: UK 2025 accident data; public APIs only; college-level complexity.

---

## 2. Literature Review / Background (1 page)

- Overview of road accident severity prediction research (cite 3–5 papers)
- Common approaches: logistic regression, decision trees, random forests, XGBoost
- Class imbalance in accident data and standard handling strategies (SMOTE, class weights)
- Data sources for UK accident analysis (DfT STATS19, OpenStreetMap, weather APIs)
- MLOps concepts: experiment tracking (MLflow), model serving (FastAPI), drift monitoring

---

## 3. Data Sources and Methodology (1.5 pages)

### 3.1 Source 1 — UK DfT STATS19 Collision Data
- Description: Official UK road casualty statistics, 2025
- URL: https://www.data.gov.uk/dataset/road-accidents-safety-data
- Licence: OGL v3.0
- Volume: 101,525 raw records, 44 columns, 2025 coverage
- Key fields: collision_index, collision_severity, date, time, coordinates

### 3.2 Source 2 — Open-Meteo Historical Weather API
- Description: Free historical weather API with daily aggregates
- URL: https://open-meteo.com/
- Licence: CC BY 4.0
- Volume: 585 grid cells, 365 days each = 213,525 rows
- Variables: temperature, precipitation, wind speed

### 3.3 Source 3 — DfT STATS19 Road Safety Data Guide
- Description: Official code lookup tables for STATS19 coded fields
- URL: https://data.dft.gov.uk/road-accidents-safety-data/
- Licence: OGL v3.0
- Purpose: Decode integer codes to human-readable labels

### 3.4 Integration Strategy
- Weather join: Nearest 0.25° grid cell on (lat, lon, date)
- Road context: Code → label mapping via lookup dictionary
- Validation: Row count preservation, primary key uniqueness checks

---

## 4. System Architecture (1 page)

Include the Mermaid diagram from `docs/architecture.md`.

Describe:
- Phase 1: Ingestion → Cleaning → Integration → Database → Dashboard
- Phase 2: Model Training → Evaluation → API → Monitoring → Docker
- Technology choices and justification (Python, SQLite, Streamlit, FastAPI, MLflow)

---

## 5. Implementation (2.5 pages)

### 5.1 Data Engineering Pipeline

- `clean_data.py`: Validation rules, rejection criteria, date parsing
- `process_weather.py`: JSON consolidation, 585 files → 213,525 rows
- `collect_road_data.py`: XLSX download, fallback lookup table
- `integrate_data.py`: Multi-source join, fan-out prevention
- `load_database.py`: SQLite schema, CSV-to-DB loading

### 5.2 Streamlit Dashboard

- Six visualisations: severity distribution, monthly trends, hourly patterns,
  weather conditions, road type breakdown, geographic map
- Sidebar filters for severity, month, road type, hour range
- Data loaded from CSV; falls back gracefully if data not present

### 5.3 Machine Learning Model

- Algorithm: RandomForestClassifier (200 trees, balanced class weights)
- Feature engineering: hour, day of week, weekend flag, month
- Pipeline: SimpleImputer → OneHotEncoder → RandomForest
- Class imbalance handling: class_weight='balanced'
- Evaluation: accuracy, per-class F1, macro-F1, confusion matrix, ROC curves

### 5.4 MLflow Experiment Tracking

- Local tracking URI: mlruns/
- Parameters and metrics logged automatically
- Model artefact registered for reproducibility

### 5.5 FastAPI Prediction Service

- Endpoints: GET /health, GET /model-info, POST /predict, POST /predict-batch
- Pydantic validation, probability outputs, Swagger UI auto-generated

### 5.6 Drift Monitoring

- KS test for numeric features; chi-square for categorical
- Temporal demonstration split (Jan–Jun vs Jul–Dec)
- Retraining trigger criteria documented

---

## 6. Results (1.5 pages)

### 6.1 Data Quality Results

Fill in from pipeline execution:
- Records ingested / cleaned / rejected
- Weather match rate
- Integration row count check result

### 6.2 Model Performance

Fill in from evaluate_model.py output:
- Accuracy: [result]
- Macro-F1: [result]
- Per-class metrics table (copy from classification_report.txt)
- Insert confusion_matrix.png
- Insert roc_curves.png

### 6.3 Dashboard Screenshots

Take screenshots of:
1. Severity distribution bar chart with filter applied
2. Monthly trend with severity breakdown
3. Hourly accident distribution
4. Geographic map showing accident hotspots

### 6.4 API Test Results

Show example curl or Swagger UI request/response for /predict.

---

## 7. Discussion (0.5 page)

- What worked well and what was challenging
- Limitations of the model (class imbalance, single year, weather approximation)
- Ethical considerations: correlation vs causation; data should not drive
  automated decisions about individuals
- How results compare to related literature

---

## 8. Conclusion (0.25 page)

- Summary of what was achieved
- Three data sources integrated successfully
- Model trained and deployed with tracking
- All acceptance criteria met (list from assignment brief)

---

## 9. References

Include citations for:
- DfT STATS19 documentation
- Open-Meteo API documentation
- scikit-learn RandomForest documentation
- MLflow documentation
- FastAPI documentation
- Any academic papers cited in Section 2
- OGL v3.0 licence text

---

## Appendices

### A. Screenshots to Capture for Submission
1. Streamlit dashboard — full page view
2. Streamlit dashboard — map view
3. MLflow UI — experiment run list
4. MLflow UI — run detail with metrics
5. FastAPI Swagger UI — /docs page
6. FastAPI Swagger UI — /predict response
7. Confusion matrix image
8. ROC curves image
9. Feature importance plot
10. Terminal showing pipeline run summary output

### B. Project Directory Listing
Run: `Get-ChildItem -Recurse -File | Select-Object FullName, Length`

### C. PowerShell Run Commands
See README.md Section "Running the Pipeline"
