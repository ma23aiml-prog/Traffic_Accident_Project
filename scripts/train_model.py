"""
train_model.py
--------------
Train a Random Forest accident severity classifier.

Target variable : collision_severity  (1=Fatal, 2=Serious, 3=Slight)
Algorithm       : RandomForestClassifier with class_weight='balanced'
Tracking        : MLflow (local mlruns/)
Outputs         : models/severity_model.pkl
                  models/metadata.json
                  models/evaluation/  (created by evaluate_model.py)

Usage:
    python scripts/train_model.py

MLflow UI:
    mlflow ui --backend-store-uri mlruns
"""

import json
import logging
import pickle
import warnings
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)   # suppress mlflow version mismatch noise

# ── Paths ─────────────────────────────────────────────────────────────────
DATA_FILE   = Path("data/cleaned/accident_severity_dataset.csv")
MODEL_DIR   = Path("models")
MODEL_FILE  = MODEL_DIR / "severity_model.pkl"
META_FILE   = MODEL_DIR / "metadata.json"
# SQLite backend — compatible with mlflow-tracing 3.x default configuration
MLFLOW_DB   = "sqlite:///mlflow.db"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────
RANDOM_SEED    = 42
TEST_SIZE      = 0.20
TARGET_COL     = "collision_severity"
EXPERIMENT_NAME = "accident_severity_classifier"

# ── Feature selection ──────────────────────────────────────────────────────
# Numeric features (impute with median)
NUMERIC_FEATURES = [
    "hour_of_day",
    "day_of_week",
    "month",
    "is_weekend",
    "number_of_vehicles",
    "number_of_casualties",
    "speed_limit",
    "wx_temp_mean_c",
    "wx_precip_mm",
    "wx_wind_max_kmh",
]

# Categorical features (impute with constant, one-hot encode)
CATEGORICAL_FEATURES = [
    "road_type_label",
    "light_conditions_label",
    "weather_conditions_label",
    "road_surface_conditions_label",
    "junction_detail_label",
    "urban_or_rural_area_label",
]

# These columns reveal the target or are identifiers — must be excluded
LEAKAGE_COLS = [
    "collision_index",
    "collision_severity",   # target itself
    TARGET_COL,
    "accident_date",
    "date",
    # label columns we want to include as categoricals — handled separately
]


def load_and_validate(path: Path) -> pd.DataFrame:
    log.info("Loading dataset: %s", path)
    df = pd.read_csv(path, low_memory=False)
    log.info("  Rows: %d, Columns: %d", df.shape[0], df.shape[1])

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found. "
            "Run integrate_data.py first."
        )

    # Drop rows with missing target
    before = len(df)
    df = df.dropna(subset=[TARGET_COL])
    after = len(df)
    if before != after:
        log.info("  Dropped %d rows with missing target.", before - after)

    # Ensure target is integer
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    log.info("  Severity distribution:\n%s", df[TARGET_COL].value_counts().to_string())
    return df


def build_pipeline(numeric_features: list, categorical_features: list) -> Pipeline:
    """Build a sklearn preprocessing + model pipeline."""
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ], remainder="drop")

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )),
    ])
    return model


def main() -> None:
    # ── Load data ──────────────────────────────────────────────────────────
    df = load_and_validate(DATA_FILE)

    # ── Select features actually present in the data ────────────────────────
    present_numeric  = [c for c in NUMERIC_FEATURES if c in df.columns]
    present_cat      = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    all_features     = present_numeric + present_cat

    log.info("Numeric features  : %s", present_numeric)
    log.info("Categorical features: %s", present_cat)

    X = df[all_features]
    y = df[TARGET_COL]

    # ── Train/test split ────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    log.info("Train size: %d  |  Test size: %d", len(X_train), len(X_test))

    # ── MLflow tracking (SQLite backend) ─────────────────────────────────
    # mlflow-tracing 3.x uses sqlite:///mlflow.db as its default URI.
    # Do NOT override with a filesystem path — that triggers maintenance mode.
    import os
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    mlflow.set_tracking_uri(MLFLOW_DB)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="random_forest_v1") as run:
        log.info("MLflow run: %s", run.info.run_id)

        # Log parameters
        mlflow.log_params({
            "model_type":         "RandomForestClassifier",
            "n_estimators":       200,
            "max_depth":          20,
            "class_weight":       "balanced",
            "random_seed":        RANDOM_SEED,
            "test_size":          TEST_SIZE,
            "n_train":            len(X_train),
            "n_test":             len(X_test),
            "n_features":         len(all_features),
            "numeric_features":   len(present_numeric),
            "categorical_features": len(present_cat),
            "dataset_size":       len(df),
        })

        # ── Train ──────────────────────────────────────────────────────────
        log.info("Training Random Forest …")
        pipeline = build_pipeline(present_numeric, present_cat)
        pipeline.fit(X_train, y_train)
        log.info("Training complete.")

        # ── Evaluate ───────────────────────────────────────────────────────
        y_pred = pipeline.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        report = classification_report(y_test, y_pred, output_dict=True)

        log.info("Accuracy   : %.4f", acc)
        log.info("Macro-F1   : %.4f", macro_f1)
        log.info("\n%s", classification_report(y_test, y_pred))

        # Log metrics
        mlflow.log_metrics({
            "accuracy":      round(acc, 6),
            "macro_f1":      round(macro_f1, 6),
        })

        # Per-class metrics
        for cls_key, cls_label in {
            "1": "fatal", "2": "serious", "3": "slight"
        }.items():
            if cls_key in report:
                mlflow.log_metrics({
                    f"f1_{cls_label}":        round(report[cls_key]["f1-score"], 6),
                    f"precision_{cls_label}": round(report[cls_key]["precision"], 6),
                    f"recall_{cls_label}":    round(report[cls_key]["recall"], 6),
                })

        # ── Save pipeline locally FIRST (before MLflow) ───────────────────
        with MODEL_FILE.open("wb") as fh:
            pickle.dump(pipeline, fh)
        log.info("Model saved: %s", MODEL_FILE)

        # ── Log model to MLflow (MLflow 3.x requires skops_trusted_types) ─
        SKOPS_TRUSTED = [
            "numpy.dtype",
            "numpy.int64",
            "numpy.float64",
            "sklearn.compose._column_transformer._RemainderColsList",
            "sklearn.ensemble._forest.RandomForestClassifier",
            "sklearn.pipeline.Pipeline",
            "sklearn.tree._tree.Tree",
            "sklearn.impute._base.SimpleImputer",
            "sklearn.preprocessing._encoders.OneHotEncoder",
            "sklearn.compose._column_transformer.ColumnTransformer",
        ]
        try:
            mlflow.sklearn.log_model(
                pipeline,
                name="model",
                skops_trusted_types=SKOPS_TRUSTED,
            )
            log.info("MLflow sklearn model logged.")
        except Exception as _exc:
            log.warning("mlflow.sklearn.log_model failed: %s", _exc)
            try:
                mlflow.log_artifact(str(MODEL_FILE), artifact_path="model")
                log.info("Logged pickle as MLflow artifact fallback.")
            except Exception as _exc2:
                log.warning("MLflow artifact fallback also failed: %s", _exc2)

        # ── Metadata ───────────────────────────────────────────────────────
        meta = {
            "trained_at":          datetime.now(timezone.utc).isoformat(),
            "mlflow_run_id":       run.info.run_id,
            "model_type":          "RandomForestClassifier",
            "target_column":       TARGET_COL,
            "severity_classes":    {1: "Fatal", 2: "Serious", 3: "Slight"},
            "numeric_features":    present_numeric,
            "categorical_features": present_cat,
            "all_features":        all_features,
            "n_features":          len(all_features),
            "dataset_size":        len(df),
            "n_train":             len(X_train),
            "n_test":              len(X_test),
            "test_size":           TEST_SIZE,
            "random_seed":         RANDOM_SEED,
            "accuracy":            round(acc, 6),
            "macro_f1":            round(macro_f1, 6),
            "model_file":          str(MODEL_FILE),
            "note": (
                "Severity codes: 1=Fatal, 2=Serious, 3=Slight. "
                "Class imbalance handled with class_weight='balanced'. "
                "Do not rely on accuracy alone — examine per-class metrics."
            ),
        }
        META_FILE.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        try:
            mlflow.log_artifact(str(META_FILE))
        except Exception as _exc:
            log.warning("Could not log metadata artifact: %s", _exc)
        log.info("Metadata saved: %s", META_FILE)

    print("\n========== MODEL TRAINING SUMMARY ==========")
    print(f"Algorithm   : Random Forest (200 trees, max_depth=20)")
    print(f"Dataset     : {len(df):,} records")
    print(f"Train / Test: {len(X_train):,} / {len(X_test):,}")
    print(f"Features    : {len(all_features)} ({len(present_numeric)} numeric, {len(present_cat)} categorical)")
    print(f"\nAccuracy    : {acc:.4f}")
    print(f"Macro-F1    : {macro_f1:.4f}")
    print(f"\nFull classification report:")
    print(classification_report(y_test, y_pred,
                                 target_names=["Fatal (1)", "Serious (2)", "Slight (3)"]))
    print(f"\nModel file  : {MODEL_FILE}")
    print(f"Metadata    : {META_FILE}")
    print(f"\nMLflow UI   : mlflow ui --backend-store-uri {MLFLOW_DB}")

    # Store test split for evaluate_model.py
    splits_dir = Path("data/staging")
    splits_dir.mkdir(exist_ok=True)
    X_test.assign(**{TARGET_COL: y_test}).to_csv(
        splits_dir / "test_split.csv", index=False
    )
    log.info("Test split saved to data/staging/test_split.csv")


if __name__ == "__main__":
    main()
