"""
evaluate_model.py
-----------------
Load the saved model and produce evaluation artefacts:
  - models/evaluation/classification_report.json
  - models/evaluation/confusion_matrix.png
  - models/evaluation/feature_importance.csv
  - models/evaluation/roc_curve.png  (one-vs-rest)

Run AFTER train_model.py:
    python scripts/evaluate_model.py

Reads the held-out test split saved by train_model.py at:
    data/staging/test_split.csv
"""

import json
import logging
import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # non-interactive backend, safe in scripts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────
MODEL_FILE  = Path("models/severity_model.pkl")
META_FILE   = Path("models/metadata.json")
TEST_SPLIT  = Path("data/staging/test_split.csv")
EVAL_DIR    = Path("models/evaluation")
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

TARGET_COL = "collision_severity"
CLASS_NAMES = {1: "Fatal", 2: "Serious", 3: "Slight"}


def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_FILE}\n"
            "Run: python scripts/train_model.py"
        )
    with MODEL_FILE.open("rb") as fh:
        pipeline = pickle.load(fh)
    log.info("Model loaded: %s", MODEL_FILE)
    return pipeline


def load_test_data(pipeline) -> tuple[pd.DataFrame, pd.Series]:
    if not TEST_SPLIT.exists():
        raise FileNotFoundError(
            f"Test split not found: {TEST_SPLIT}\n"
            "Run: python scripts/train_model.py first."
        )
    df = pd.read_csv(TEST_SPLIT, low_memory=False)
    y_test = df[TARGET_COL].astype(int)

    # Recover feature names from metadata
    meta = {}
    if META_FILE.exists():
        meta = json.loads(META_FILE.read_text(encoding="utf-8"))

    all_features = meta.get("all_features", [])
    available    = [c for c in all_features if c in df.columns]
    if not available:
        # Fallback: use all columns except target
        available = [c for c in df.columns if c != TARGET_COL]

    X_test = df[available]
    log.info("Test set: %d rows, %d features", len(X_test), len(available))
    return X_test, y_test


def save_classification_report(y_test, y_pred) -> dict:
    report = classification_report(
        y_test, y_pred,
        target_names=["Fatal (1)", "Serious (2)", "Slight (3)"],
        output_dict=True,
    )
    out_path = EVAL_DIR / "classification_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Also save a human-readable text version
    text_path = EVAL_DIR / "classification_report.txt"
    text_path.write_text(
        classification_report(
            y_test, y_pred,
            target_names=["Fatal (1)", "Serious (2)", "Slight (3)"],
        ),
        encoding="utf-8",
    )
    log.info("Classification report saved: %s", out_path)
    return report


def save_confusion_matrix(y_test, y_pred) -> None:
    labels  = sorted(y_test.unique())
    names   = [CLASS_NAMES.get(l, str(l)) for l in labels]
    cm      = confusion_matrix(y_test, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=names)
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title("Confusion Matrix — Accident Severity Classifier", pad=14)
    plt.tight_layout()

    out_path = EVAL_DIR / "confusion_matrix.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Confusion matrix saved: %s", out_path)


def save_feature_importance(pipeline) -> None:
    """Extract feature importances from the Random Forest inside the pipeline."""
    try:
        clf = pipeline.named_steps["classifier"]
        pre = pipeline.named_steps["preprocessor"]

        # Get feature names from the preprocessor
        feature_names = pre.get_feature_names_out()
        importances   = clf.feature_importances_

        fi = pd.DataFrame({
            "feature":    feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False)

        out_path = EVAL_DIR / "feature_importance.csv"
        fi.to_csv(out_path, index=False)
        log.info("Feature importances saved: %s", out_path)

        # Top-20 plot
        top20 = fi.head(20)
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(top20["feature"][::-1], top20["importance"][::-1], color="#3b82f6")
        ax.set_xlabel("Importance")
        ax.set_title("Top 20 Feature Importances")
        plt.tight_layout()
        plot_path = EVAL_DIR / "feature_importance.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Feature importance plot saved: %s", plot_path)

    except Exception as exc:
        log.warning("Could not extract feature importances: %s", exc)


def save_roc_curves(pipeline, X_test, y_test) -> None:
    """One-vs-rest ROC curves for each severity class."""
    try:
        classes = sorted(y_test.unique())
        y_bin   = label_binarize(y_test, classes=classes)
        y_prob  = pipeline.predict_proba(X_test)

        fig, ax = plt.subplots(figsize=(8, 6))
        colours = ["#ef4444", "#f97316", "#22c55e"]

        for i, (cls, colour) in enumerate(zip(classes, colours)):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            roc_auc     = auc(fpr, tpr)
            label = f"{CLASS_NAMES.get(cls, cls)} (AUC={roc_auc:.3f})"
            ax.plot(fpr, tpr, color=colour, label=label, linewidth=2)

        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves — One-vs-Rest (Accident Severity)")
        ax.legend(loc="lower right")
        plt.tight_layout()

        out_path = EVAL_DIR / "roc_curves.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("ROC curves saved: %s", out_path)

    except Exception as exc:
        log.warning("Could not generate ROC curves: %s", exc)


def main() -> None:
    pipeline        = load_model()
    X_test, y_test  = load_test_data(pipeline)

    log.info("Generating predictions …")
    y_pred = pipeline.predict(X_test)

    # ── Metrics ───────────────────────────────────────────────────────────
    acc      = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    # ── Save artefacts ────────────────────────────────────────────────────
    report = save_classification_report(y_test, y_pred)
    save_confusion_matrix(y_test, y_pred)
    save_feature_importance(pipeline)
    save_roc_curves(pipeline, X_test, y_test)

    # ── Print summary ──────────────────────────────────────────────────────
    print("\n========== MODEL EVALUATION REPORT ==========")
    print(f"Test set size : {len(y_test):,} records")
    print(f"Accuracy      : {acc:.4f}")
    print(f"Macro-F1      : {macro_f1:.4f}")
    print()
    print(classification_report(
        y_test, y_pred,
        target_names=["Fatal (1)", "Serious (2)", "Slight (3)"]
    ))
    print("Artefacts saved to:", EVAL_DIR)
    print("  classification_report.json / .txt")
    print("  confusion_matrix.png")
    print("  feature_importance.csv / .png")
    print("  roc_curves.png")

    # ── Important note on class imbalance ─────────────────────────────────
    fatal_count = (y_test == 1).sum()
    print(f"\n[NOTE] Fatal class in test set: {fatal_count} records "
          f"({100*fatal_count/len(y_test):.1f}%). "
          "Accuracy alone is misleading — use per-class recall for Fatal.")


if __name__ == "__main__":
    main()
