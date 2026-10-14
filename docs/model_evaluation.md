# docs/model_evaluation.md
# Model Evaluation Documentation — Accident Severity Classifier

> **Execution note**: Metric values marked `[RUN PIPELINE]` are placeholders.
> Run `python scripts/train_model.py` then `python scripts/evaluate_model.py`
> and replace these values with actual results.

---

## Model Overview

| Property | Value |
|----------|-------|
| Algorithm | RandomForestClassifier (scikit-learn) |
| n_estimators | 200 |
| max_depth | 20 |
| min_samples_split | 5 |
| min_samples_leaf | 2 |
| class_weight | balanced |
| random_state | 42 |
| Test split | 20% stratified |

---

## Target Variable

**Column**: `collision_severity`

| Class | Code | Meaning | Approx. training proportion |
|-------|------|---------|---------------------------|
| Slight | 3 | Slight injury | ~73.7% |
| Serious | 2 | Serious injury | ~24.8% |
| Fatal | 1 | Fatal accident | ~1.4% |

> **Important**: This is a severely imbalanced classification problem.
> `class_weight='balanced'` is used so the model does not simply predict
> "Slight" for everything. Accuracy is a misleading metric here —
> **macro-F1 and Fatal class recall** are the primary evaluation measures.

---

## Features Used

### Numeric (10 features, median-imputed)
| Feature | Description |
|---------|-------------|
| `hour_of_day` | Hour of accident (0–23) |
| `day_of_week` | 0=Monday, 6=Sunday |
| `month` | Calendar month (1–12) |
| `is_weekend` | 1 if Sat/Sun |
| `number_of_vehicles` | Vehicles involved |
| `number_of_casualties` | Casualties |
| `speed_limit` | Posted speed limit (mph) |
| `wx_temp_mean_c` | Daily mean temperature (°C) |
| `wx_precip_mm` | Daily precipitation (mm) |
| `wx_wind_max_kmh` | Daily max wind speed (km/h) |

### Categorical (6 features, one-hot encoded after "Unknown" imputation)
| Feature | Description |
|---------|-------------|
| `road_type_label` | Road type (e.g. Single carriageway) |
| `light_conditions_label` | Lighting at time of accident |
| `weather_conditions_label` | Weather at time of accident |
| `road_surface_conditions_label` | Road surface condition |
| `junction_detail_label` | Junction type |
| `urban_or_rural_area_label` | Urban / Rural |

### Excluded (leakage prevention)
- `collision_index` — identifier
- `date`, `accident_date` — temporal identifiers
- Raw integer code columns (e.g. `road_type`) when label version is present

---

## Evaluation Metrics

### Primary Metrics
| Metric | Value |
|--------|-------|
| Accuracy | [RUN PIPELINE] |
| Macro-F1 | [RUN PIPELINE] |

### Per-Class Metrics
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Fatal (1) | [RUN] | [RUN] | [RUN] | [RUN] |
| Serious (2) | [RUN] | [RUN] | [RUN] | [RUN] |
| Slight (3) | [RUN] | [RUN] | [RUN] | [RUN] |
| Macro avg | [RUN] | [RUN] | [RUN] | |
| Weighted avg | [RUN] | [RUN] | [RUN] | |

### Artefacts Produced
| File | Description |
|------|-------------|
| `models/evaluation/classification_report.json` | Full per-class metrics (JSON) |
| `models/evaluation/classification_report.txt` | Human-readable report |
| `models/evaluation/confusion_matrix.png` | Confusion matrix heatmap |
| `models/evaluation/feature_importance.csv` | Feature importances ranked |
| `models/evaluation/feature_importance.png` | Top-20 feature importance plot |
| `models/evaluation/roc_curves.png` | One-vs-rest ROC curves |

---

## MLflow Experiment Tracking

**Experiment name**: `accident_severity_classifier`
**Tracking URI**: local `mlruns/` directory

Logged parameters:
- model_type, n_estimators, max_depth, class_weight
- random_seed, test_size, n_train, n_test, n_features, dataset_size

Logged metrics:
- accuracy, macro_f1
- f1_fatal, precision_fatal, recall_fatal
- f1_serious, precision_serious, recall_serious
- f1_slight, precision_slight, recall_slight

**View UI**: `mlflow ui --backend-store-uri mlruns`
Then open: http://localhost:5000

---

## Known Limitations

1. **Class imbalance**: Fatal accidents are ~1.4% of the dataset. Even with
   balanced class weights, detecting Fatal accidents reliably is difficult.
   Recall for Fatal class is expected to be lower than for Slight.

2. **Temporal features only**: The model uses time of day and month but not
   continuous date features. It cannot learn year-over-year trends.

3. **Weather approximation**: Weather is a 0.25° grid daily average —
   not a precise measurement at the accident location or time of day.

4. **Causal inference**: Correlation in the training data does not imply
   causation. High feature importance for `speed_limit` does not mean
   speed limits cause severity — it may reflect road type confounding.

5. **Single year**: Trained on 2025 data only. Seasonal generalisation
   within the year is possible; cross-year generalisation is unverified.

6. **No SMOTE or resampling**: Synthetic data was deliberately avoided to
   prevent data leakage. Class weights are used instead.

---

## Retraining Criteria

A new model version should be trained when:

1. New labelled accident data becomes available (≥ 5,000 new records)
2. Macro-F1 on held-out evaluation data drops below 0.45
3. Drift monitoring flags sustained KS > 0.15 on ≥ 2 of the top-5 most
   important numeric features
4. The DfT publishes a new annual STATS19 dataset
