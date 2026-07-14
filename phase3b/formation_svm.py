"""Formation-proxy SVM selected on a validation-only held-out split."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from phase3b.formation_features import build_formation_features
from phase3b.graphs import FORMATION2ID
from phase3b.metrics import confusion, macro_f1
from phase3b.train import split_items

ID2FORMATION = {v: k for k, v in FORMATION2ID.items()}
VAL_SEQS = [78, 79, 80, 81, 82, 83]
MODEL_PATH = "phase3b/formation_svm.pklz"


def _model():
    # Fixed after validation selection; do not tune against the test split.
    return make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf", class_weight="balanced", probability=True, random_state=7))


def _arrays(items):
    return np.stack([item["features"] for item in items]), np.array([item["label"] for item in items])


def validate():
    items = build_formation_features("valid")
    train, holdout = split_items(items, val_seqs=VAL_SEQS)
    model = _model()
    x_train, y_train = _arrays(train)
    x_holdout, y_holdout = _arrays(holdout)
    model.fit(x_train, y_train)
    p = model.predict(x_holdout)
    result = {
        "architecture": "RBF SVM on positional distribution features",
        "train_examples": len(train), "holdout_examples": len(holdout),
        "holdout_macro_f1": round(macro_f1(y_holdout, p, len(FORMATION2ID)), 6),
        "target": "GT-derived geometric formation proxy",
    }
    Path("phase3b/formation_svm_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


def train_final():
    """Fit the fixed selected model on all validation clips, without test labels."""
    items = build_formation_features("valid")
    model = _model()
    x, y = _arrays(items)
    model.fit(x, y)
    joblib.dump(model, MODEL_PATH, compress=3)
    print(f"trained {len(items)} validation examples -> {MODEL_PATH}")
    return model


def evaluate():
    model = joblib.load(MODEL_PATH)
    items = build_formation_features("test")
    x, y = _arrays(items)
    predictions = model.predict(x)
    probabilities = model.predict_proba(x)
    rows = []
    for item, pred, probs in zip(items, predictions, probabilities):
        rows.append({
            "seq": item["seq"], "team": item["team"],
            "formation_reference": item["formation"],
            "formation_pred": ID2FORMATION[int(pred)],
            "confidence": round(float(probs.max()), 6),
        })
    pd.DataFrame(rows).to_csv("phase3b/formation_svm_test_predictions.csv", index=False)
    result = {
        "architecture": "RBF SVM on positional distribution features",
        "test_examples": len(items),
        "test_macro_f1": round(macro_f1(y, predictions, len(FORMATION2ID)), 6),
        "confusion_rows_reference_cols_prediction": confusion(y, predictions, len(FORMATION2ID)).tolist(),
        "target": "GT-derived geometric formation proxy",
    }
    Path("phase3b/formation_svm_test_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "train", "evaluate"))
    args = parser.parse_args()
    {"validate": validate, "train": train_final, "evaluate": evaluate}[args.command]()
