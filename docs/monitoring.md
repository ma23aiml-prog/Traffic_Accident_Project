# docs/monitoring.md
# Drift Monitoring & Retraining Strategy

## Overview

Drift monitoring detects when the statistical properties of input features
change over time, which can degrade model performance even without visible errors.

This project implements a **demonstration** drift monitor using a temporal
split of the 2025 data. Only one year of data is available, so no real
production drift has been observed. The method is documented clearly as
a demonstration of the approach.

---

## Method

**Script**: `scripts/drift_monitor.py`

**Reference dataset**: Accidents from January–June 2025
**Current dataset**: Accidents from July–December 2025

This split is clearly labelled as a demonstration. In production, the
reference set would be the training data and the current set would be
newly collected production data.

---

## Statistical Tests

### Numeric Features — Kolmogorov-Smirnov Two-Sample Test

The KS test measures the maximum absolute difference between two empirical
cumulative distribution functions. It is non-parametric and does not assume
normality.

| Threshold | Meaning |
|-----------|---------|
| KS statistic > 0.10 | Feature distribution has changed meaningfully |
| p-value < 0.05 | Change is statistically significant |
| Both conditions | Feature flagged as **drifted** |

Features monitored:
- `hour_of_day`, `day_of_week`, `number_of_vehicles`, `number_of_casualties`
- `speed_limit`, `wx_temp_mean_c`, `wx_precip_mm`, `wx_wind_max_kmh`

### Categorical Features — Chi-Square Test

The chi-square test compares observed frequencies in the current set
against expected frequencies derived from the reference set.

| Threshold | Meaning |
|-----------|---------|
| p-value < 0.05 | Category distribution has shifted significantly |

Features monitored:
- `road_type_label`, `light_conditions_label`, `weather_conditions_label`
- `road_surface_conditions_label`, `urban_or_rural_area_label`

### Target Distribution Shift

The distribution of `collision_severity` classes is compared between
reference and current periods. A shift in class proportions (e.g. more
Fatal accidents in a period) can indicate both real-world changes and
potential labelling issues.

---

## Outputs

| File | Description |
|------|-------------|
| `logs/drift_report.json` | Full drift results per feature |
| `logs/drift_summary.txt` | Human-readable summary with flags |

---

## Retraining Trigger Criteria

A retraining run is triggered when **all** of the following conditions are met:

| Condition | Threshold |
|-----------|-----------|
| Numeric drift | ≥ 2 of the top-5 most important features show KS > 0.15 |
| New data available | ≥ 1,000 new labelled accident records |
| **OR** performance degradation | Macro-F1 on held-out data falls below 0.45 |

### Retraining Process

1. Combine original training data with new labelled data
2. Re-run `scripts/clean_data.py` and `scripts/integrate_data.py` on new data
3. Re-run `scripts/train_model.py` (new MLflow run is created automatically)
4. Re-run `scripts/evaluate_model.py` to compare metrics with previous version
5. If new model is better on macro-F1, replace `models/severity_model.pkl`
6. Update the API without downtime by copying new model file then restarting

---

## Limitations

1. **Single year**: Seasonal effects are conflated with drift. In a Jan–Jun vs
   Jul–Dec comparison, temperature drift is expected (winter vs summer) and
   does not necessarily indicate model degradation.

2. **No ground-truth labels**: Drift in features can be detected automatically,
   but performance degradation can only be confirmed with labelled evaluation data.

3. **Manual review required**: The script flags drift but does not automatically
   retrain. A human should review drift reports before triggering retraining.

4. **No real production data**: This implementation has never been tested against
   actual new data in a production environment.

---

## Running the Monitor

```powershell
# Activate venv and run drift monitor
.venv\Scripts\Activate.ps1
python scripts/drift_monitor.py

# View output
Get-Content logs\drift_summary.txt
```
