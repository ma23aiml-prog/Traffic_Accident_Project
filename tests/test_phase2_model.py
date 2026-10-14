"""
tests/test_phase2_model.py
---------------------------
Phase 2 tests: model loading, prediction, API endpoints.

Run with:
    pytest tests/test_phase2_model.py -v

These tests run AFTER Phase 2 scripts:
    python scripts/train_model.py
    python scripts/evaluate_model.py
    uvicorn api.main:app --host 0.0.0.0 --port 8000  (for API tests)
"""

import json
import pickle
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent

MODEL_FILE  = ROOT / "models/severity_model.pkl"
META_FILE   = ROOT / "models/metadata.json"
TEST_SPLIT  = ROOT / "data/staging/test_split.csv"
EVAL_DIR    = ROOT / "models/evaluation"
DATASET     = ROOT / "data/cleaned/accident_severity_dataset.csv"
DRIFT_LOG   = ROOT / "logs/drift_report.json"


# ═══════════════════════════════════════════════════════════════════════════
# Model file tests
# ═══════════════════════════════════════════════════════════════════════════

class TestModelFiles:
    def test_model_file_exists(self):
        assert MODEL_FILE.exists(), (
            f"Model not found: {MODEL_FILE}\n"
            "Run: python scripts/train_model.py"
        )

    def test_metadata_file_exists(self):
        assert META_FILE.exists(), f"Metadata not found: {META_FILE}"

    def test_test_split_exists(self):
        assert TEST_SPLIT.exists(), (
            f"Test split not found: {TEST_SPLIT}\n"
            "Run: python scripts/train_model.py"
        )

    def test_model_loads(self):
        with MODEL_FILE.open("rb") as fh:
            pipeline = pickle.load(fh)
        assert pipeline is not None

    def test_metadata_valid_json(self):
        meta = json.loads(META_FILE.read_text(encoding="utf-8"))
        assert "model_type" in meta
        assert "all_features" in meta
        assert "accuracy" in meta
        assert "macro_f1" in meta

    def test_metadata_has_expected_fields(self):
        meta = json.loads(META_FILE.read_text(encoding="utf-8"))
        for field in [
            "trained_at", "target_column", "numeric_features",
            "categorical_features", "dataset_size", "n_train", "n_test",
        ]:
            assert field in meta, f"Metadata missing field: {field}"


# ═══════════════════════════════════════════════════════════════════════════
# Model prediction correctness
# ═══════════════════════════════════════════════════════════════════════════

class TestModelPredictions:
    @pytest.fixture(scope="class")
    def pipeline(self):
        with MODEL_FILE.open("rb") as fh:
            return pickle.load(fh)

    @pytest.fixture(scope="class")
    def meta(self):
        return json.loads(META_FILE.read_text(encoding="utf-8"))

    @pytest.fixture(scope="class")
    def test_data(self, meta):
        df = pd.read_csv(TEST_SPLIT, low_memory=False)
        features = meta.get("all_features", [])
        features = [f for f in features if f in df.columns]
        y = df["collision_severity"].astype(int)
        X = df[features]
        return X, y

    def test_predict_returns_valid_classes(self, pipeline, test_data):
        X, _ = test_data
        preds = pipeline.predict(X.head(50))
        assert set(preds).issubset({1, 2, 3}), (
            f"Predictions contain unexpected classes: {set(preds)}"
        )

    def test_predict_proba_sums_to_one(self, pipeline, test_data):
        X, _ = test_data
        probs = pipeline.predict_proba(X.head(50))
        sums = probs.sum(axis=1)
        assert all(abs(s - 1.0) < 1e-6 for s in sums), (
            "predict_proba rows do not sum to 1.0"
        )

    def test_all_three_classes_predicted(self, pipeline, test_data):
        """Model must predict at least some Fatal, Serious, and Slight."""
        X, _ = test_data
        # Use enough samples to likely see all 3 classes
        preds = pipeline.predict(X.head(500))
        assert 1 in preds, "Model never predicts Fatal (class 1) in 500 test records."
        assert 2 in preds, "Model never predicts Serious (class 2) in 500 test records."
        assert 3 in preds, "Model never predicts Slight (class 3) in 500 test records."

    def test_single_row_prediction(self, pipeline, meta):
        """Single-record prediction must work without errors."""
        features = meta.get("all_features", [])
        row = pd.DataFrame([{f: None for f in features}])
        pred = pipeline.predict(row)
        assert pred[0] in {1, 2, 3}

    def test_macro_f1_above_minimum(self, pipeline, test_data, meta):
        """Macro-F1 on test set must be above 0.20 (better than random chance)."""
        from sklearn.metrics import f1_score
        X, y = test_data
        preds   = pipeline.predict(X)
        macro_f1 = f1_score(y, preds, average="macro")
        assert macro_f1 > 0.20, (
            f"Macro-F1 is {macro_f1:.3f} — below minimum threshold 0.20."
        )


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation artefacts
# ═══════════════════════════════════════════════════════════════════════════

class TestEvaluationArtefacts:
    def test_eval_dir_exists(self):
        assert EVAL_DIR.exists(), (
            f"Evaluation directory not found: {EVAL_DIR}\n"
            "Run: python scripts/evaluate_model.py"
        )

    def test_classification_report_exists(self):
        assert (EVAL_DIR / "classification_report.json").exists()

    def test_confusion_matrix_exists(self):
        assert (EVAL_DIR / "confusion_matrix.png").exists()

    def test_feature_importance_exists(self):
        assert (EVAL_DIR / "feature_importance.csv").exists()

    def test_classification_report_valid(self):
        path = EVAL_DIR / "classification_report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        assert "macro avg" in report
        assert "weighted avg" in report

    def test_feature_importance_columns(self):
        fi = pd.read_csv(EVAL_DIR / "feature_importance.csv")
        assert "feature" in fi.columns
        assert "importance" in fi.columns
        assert len(fi) > 0


# ═══════════════════════════════════════════════════════════════════════════
# FastAPI endpoint tests (requires running server)
# ═══════════════════════════════════════════════════════════════════════════

try:
    from httpx import Client
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

API_BASE = "http://localhost:8000"

@pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not installed")
class TestAPI:
    """
    These tests require the API to be running:
        uvicorn api.main:app --host 0.0.0.0 --port 8000

    Skip these with: pytest tests/test_phase2_model.py -k "not TestAPI"
    """

    @pytest.fixture(scope="class")
    def client(self):
        client = Client(base_url=API_BASE, timeout=10.0)
        # Check server is reachable before running tests
        try:
            client.get("/")
        except Exception:
            pytest.skip("API server not running. Start with: uvicorn api.main:app")
        return client

    def test_root_endpoint(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "service" in data

    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "model_loaded" in data

    def test_model_info_endpoint(self, client):
        resp = client.get("/model-info")
        assert resp.status_code in (200, 404)
        if resp.status_code == 200:
            data = resp.json()
            assert "model_type" in data
            assert "n_features" in data

    def test_predict_with_full_input(self, client):
        payload = {
            "hour_of_day": 8,
            "day_of_week": 1,
            "month": 3,
            "is_weekend": 0,
            "number_of_vehicles": 2,
            "number_of_casualties": 1,
            "speed_limit": 30,
            "wx_temp_mean_c": 8.5,
            "wx_precip_mm": 2.3,
            "wx_wind_max_kmh": 25.0,
            "road_type_label": "Single carriageway",
            "light_conditions_label": "Daylight",
            "weather_conditions_label": "Raining no high winds",
            "road_surface_conditions_label": "Wet or damp",
            "junction_detail_label": "T or staggered junction",
            "urban_or_rural_area_label": "Urban",
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200, f"Unexpected status: {resp.status_code}, {resp.text}"
        data = resp.json()
        assert "prediction" in data
        pred = data["prediction"]
        assert pred["predicted_class"] in [1, 2, 3]
        assert 0.0 <= pred["probability_fatal"] <= 1.0
        assert 0.0 <= pred["probability_serious"] <= 1.0
        assert 0.0 <= pred["probability_slight"] <= 1.0

    def test_predict_with_missing_fields(self, client):
        """Model should handle all-None inputs gracefully."""
        payload = {}
        resp = client.post("/predict", json=payload)
        assert resp.status_code in (200, 422, 503)

    def test_predict_batch(self, client):
        payload = {
            "records": [
                {"hour_of_day": 8, "speed_limit": 30, "month": 6},
                {"hour_of_day": 22, "speed_limit": 70, "month": 11},
            ]
        }
        resp = client.post("/predict-batch", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        assert len(data["predictions"]) == 2

    def test_predict_response_probabilities_sum_to_one(self, client):
        payload = {"hour_of_day": 14, "speed_limit": 60}
        resp = client.post("/predict", json=payload)
        if resp.status_code == 200:
            pred = resp.json()["prediction"]
            total = (
                pred["probability_fatal"]
                + pred["probability_serious"]
                + pred["probability_slight"]
            )
            assert abs(total - 1.0) < 0.01, (
                f"Probabilities sum to {total:.4f}, expected ~1.0"
            )


# ═══════════════════════════════════════════════════════════════════════════
# Drift monitoring tests
# ═══════════════════════════════════════════════════════════════════════════

class TestDriftMonitoring:
    def test_drift_report_exists(self):
        assert DRIFT_LOG.exists(), (
            f"Drift report not found: {DRIFT_LOG}\n"
            "Run: python scripts/drift_monitor.py"
        )

    def test_drift_report_valid_json(self):
        report = json.loads(DRIFT_LOG.read_text(encoding="utf-8"))
        assert "numeric_drift" in report
        assert "categorical_drift" in report
        assert "summary" in report

    def test_drift_report_has_retraining_criteria(self):
        report = json.loads(DRIFT_LOG.read_text(encoding="utf-8"))
        summary = report.get("summary", {})
        assert "retraining_criteria" in summary, (
            "Drift report must document retraining criteria."
        )
