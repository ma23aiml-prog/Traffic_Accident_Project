"""
drift_monitor.py
----------------
Basic feature drift monitoring — Phase 2 demonstration.

Since we have only one year of data (2025), we demonstrate the drift
detection method by comparing:
  - REFERENCE SET: accidents from January–June 2025
  - CURRENT SET:   accidents from July–December 2025

This is clearly labelled as a DEMONSTRATION, not real production monitoring.

Tests used:
  - Numeric features  : Kolmogorov-Smirnov two-sample test
  - Categorical features: Chi-square test on frequency distributions

Thresholds:
  - KS statistic  > 0.1   → flag as drifted
  - p-value       < 0.05  → statistically significant drift

Outputs:
  - logs/drift_report.json   — full results with drift flags
  - logs/drift_summary.txt   — human-readable summary

Retraining trigger criteria (documented):
  - Any of the top-5 most important numeric features show KS > 0.15
  - AND new labelled data is available (≥ 1,000 new records)
  - OR macro-F1 on held-out evaluation drops below 0.45
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ── Paths -------------------------------------------------------──────────
DATA_FILE   = Path("data/cleaned/accident_severity_dataset.csv")
LOG_DIR     = Path("logs")
REPORT_FILE = LOG_DIR / "drift_report.json"
SUMMARY_FILE = LOG_DIR / "drift_summary.txt"

LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging -------------------------------------------------------────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Feature sets to monitor ───────────────────────────────────────────────
NUMERIC_MONITOR = [
    "hour_of_day", "day_of_week", "number_of_vehicles",
    "number_of_casualties", "speed_limit",
    "wx_temp_mean_c", "wx_precip_mm", "wx_wind_max_kmh",
]

CATEGORICAL_MONITOR = [
    "road_type_label", "light_conditions_label",
    "weather_conditions_label", "road_surface_conditions_label",
    "urban_or_rural_area_label",
]

# Drift thresholds
KS_THRESHOLD      = 0.10    # flag if KS statistic exceeds this
PVALUE_THRESHOLD  = 0.05    # flag if p-value below this
CHI_PVALUE_THRESH = 0.05    # for categorical features


def ks_test(ref: pd.Series, cur: pd.Series) -> dict:
    """Run KS two-sample test on two numeric series."""
    ref_clean = ref.dropna().values
    cur_clean = cur.dropna().values
    if len(ref_clean) < 30 or len(cur_clean) < 30:
        return {"ks_stat": None, "p_value": None, "drifted": False,
                "reason": "insufficient_data"}
    stat, pvalue = stats.ks_2samp(ref_clean, cur_clean)
    drifted = (stat > KS_THRESHOLD) and (pvalue < PVALUE_THRESHOLD)
    return {
        "ks_stat":   round(float(stat), 6),
        "p_value":   round(float(pvalue), 6),
        "drifted":   bool(drifted),
        "threshold_ks":  KS_THRESHOLD,
        "threshold_pval": PVALUE_THRESHOLD,
    }


def chi2_test(ref: pd.Series, cur: pd.Series) -> dict:
    """Run chi-square test on two categorical series using contingency table."""
    from scipy.stats import chi2_contingency
    ref_clean = ref.dropna()
    cur_clean = cur.dropna()
    if len(ref_clean) < 30 or len(cur_clean) < 30:
        return {"chi2_stat": None, "p_value": None, "drifted": False,
                "reason": "insufficient_data"}

    all_cats = sorted(set(ref_clean.unique()) | set(cur_clean.unique()))
    if len(all_cats) < 2:
        return {"chi2_stat": None, "p_value": None, "drifted": False,
                "reason": "too_few_categories"}

    ref_counts = ref_clean.value_counts().reindex(all_cats, fill_value=0).values
    cur_counts = cur_clean.value_counts().reindex(all_cats, fill_value=0).values

    # Build 2×k contingency table and run chi2_contingency
    # (avoids floating-point normalisation issues of chisquare())
    contingency = np.array([ref_counts, cur_counts])
    # Drop columns where both counts are 0 to avoid degenerate cells
    nonzero_cols = (contingency.sum(axis=0) > 0)
    contingency = contingency[:, nonzero_cols]
    if contingency.shape[1] < 2:
        return {"chi2_stat": None, "p_value": None, "drifted": False,
                "reason": "too_few_categories"}

    chi2, pvalue, dof, _ = chi2_contingency(contingency)
    drifted = pvalue < CHI_PVALUE_THRESH
    return {
        "chi2_stat":  round(float(chi2), 4),
        "p_value":    round(float(pvalue), 6),
        "drifted":    bool(drifted),
        "threshold_pval": CHI_PVALUE_THRESH,
    }



def severity_distribution_shift(ref: pd.Series, cur: pd.Series) -> dict:
    """Check whether class distribution has shifted."""
    ref_pct = ref.value_counts(normalize=True).sort_index().to_dict()
    cur_pct = cur.value_counts(normalize=True).sort_index().to_dict()
    classes = sorted(set(ref_pct) | set(cur_pct))
    shifts  = {}
    for cls in classes:
        r = ref_pct.get(cls, 0.0)
        c = cur_pct.get(cls, 0.0)
        shifts[str(cls)] = {
            "reference_pct": round(r * 100, 2),
            "current_pct":   round(c * 100, 2),
            "abs_change_pct": round(abs(c - r) * 100, 2),
        }
    return shifts


def main() -> None:
    # ── Load data -------------------------------------------------------───
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Run the Phase 1 pipeline first."
        )

    log.info("Loading dataset …")
    df = pd.read_csv(DATA_FILE, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    log.info("  %d records loaded", len(df))

    # ── Split by half-year ─────────────────────────────────────────────────
    ref = df[df["date"].dt.month <= 6].copy()   # Jan-Jun = REFERENCE
    cur = df[df["date"].dt.month >  6].copy()   # Jul-Dec = CURRENT

    log.info(
        "Reference period (Jan-Jun): %d records | "
        "Current period (Jul-Dec): %d records",
        len(ref), len(cur),
    )

    results = {
        "generated_at":      datetime.now(timezone.utc).isoformat(),
        "note":              (
            "DEMONSTRATION ONLY. Reference=Jan-Jun 2025, Current=Jul-Dec 2025. "
            "This split is used to demonstrate the drift detection method, "
            "not to claim actual production drift has been observed."
        ),
        "reference_n":       len(ref),
        "current_n":         len(cur),
        "numeric_drift":     {},
        "categorical_drift": {},
        "severity_shift":    {},
        "summary":           {},
    }

    # ── Numeric drift ──────────────────────────────────────────────────────
    log.info("Testing numeric features …")
    drifted_numeric = []
    for col in NUMERIC_MONITOR:
        if col not in df.columns:
            log.debug("  Skipping missing column: %s", col)
            continue
        result = ks_test(ref[col], cur[col])
        results["numeric_drift"][col] = result
        if result.get("drifted"):
            drifted_numeric.append(col)
            log.info("  DRIFT: %-30s  KS=%.4f  p=%.4f",
                     col, result["ks_stat"], result["p_value"])
        else:
            log.info("  OK   : %-30s  KS=%.4f  p=%.4f",
                     col,
                     result.get("ks_stat", 0) or 0,
                     result.get("p_value", 1) or 1)

    # ── Categorical drift ──────────────────────────────────────────────────
    log.info("Testing categorical features …")
    drifted_cat = []
    for col in CATEGORICAL_MONITOR:
        if col not in df.columns:
            continue
        result = chi2_test(ref[col], cur[col])
        results["categorical_drift"][col] = result
        if result.get("drifted"):
            drifted_cat.append(col)
            log.info("  DRIFT: %-30s  chi2=%.2f  p=%.4f",
                     col, result.get("chi2_stat", 0), result.get("p_value", 1))
        else:
            log.info("  OK   : %-30s  chi2=%.2f  p=%.4f",
                     col,
                     result.get("chi2_stat", 0) or 0,
                     result.get("p_value", 1) or 1)

    # ── Severity distribution ──────────────────────────────────────────────
    if "collision_severity" in df.columns:
        results["severity_shift"] = severity_distribution_shift(
            ref["collision_severity"], cur["collision_severity"]
        )

    # ── Summary -------------------------------------------------------─────
    results["summary"] = {
        "total_numeric_tested":     len(results["numeric_drift"]),
        "numeric_drifted":          len(drifted_numeric),
        "drifted_numeric_features": drifted_numeric,
        "total_categorical_tested": len(results["categorical_drift"]),
        "categorical_drifted":      len(drifted_cat),
        "drifted_cat_features":     drifted_cat,
        "retraining_triggered":     False,
        "retraining_criteria": (
            "Retrain when: (1) ≥2 top-5 numeric features show KS>0.15 AND "
            "≥1000 new labelled records are available, OR "
            "(2) macro-F1 on new held-out data falls below 0.45."
        ),
    }

    # ── Save reports -------------------------------------------------------
    REPORT_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")

    summary_lines = [
        "=" * 55,
        "DRIFT MONITORING REPORT  (DEMONSTRATION)",
        "=" * 55,
        f"Generated   : {results['generated_at']}",
        f"Reference   : Jan-Jun 2025  ({len(ref):,} records)",
        f"Current     : Jul-Dec 2025  ({len(cur):,} records)",
        "",
        "NOTE: This uses a temporal split on the same year to",
        "demonstrate the method. Real monitoring would compare",
        "training data against newly collected production data.",
        "",
        "─" * 55,
        "NUMERIC FEATURE DRIFT (Kolmogorov-Smirnov test)",
        "─" * 55,
    ]
    for col, r in results["numeric_drift"].items():
        flag = "! DRIFT" if r.get("drifted") else "  OK   "
        ks   = r.get("ks_stat") or 0
        pv   = (r.get("p_value") if r.get("p_value") is not None else 1.0)
        summary_lines.append(f"  {flag}  {col:<32}  KS={ks:.4f}  p={pv:.4f}")

    summary_lines += [
        "",
        "─" * 55,
        "CATEGORICAL DRIFT (Chi-square test)",
        "─" * 55,
    ]
    for col, r in results["categorical_drift"].items():
        flag = "! DRIFT" if r.get("drifted") else "  OK   "
        chi2 = r.get("chi2_stat") or 0
        pv   = (r.get("p_value") if r.get("p_value") is not None else 1.0)
        summary_lines.append(f"  {flag}  {col:<32}  chi2={chi2:.2f}  p={pv:.4f}")

    summary_lines += [
        "",
        "─" * 55,
        f"Drifted numeric features   : {len(drifted_numeric)}",
        f"Drifted categorical features: {len(drifted_cat)}",
        f"Retraining triggered        : No (demonstration only)",
        "─" * 55,
    ]

    summary_text = "\n".join(summary_lines)
    SUMMARY_FILE.write_text(summary_text, encoding="utf-8")

    import sys
    _safe = summary_text.encode('utf-8', errors='replace')
    sys.stdout.buffer.write(_safe + b'\n')
    sys.stdout.buffer.flush()
    print(f"\nFull report : {REPORT_FILE}")
    print(f"Summary     : {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
