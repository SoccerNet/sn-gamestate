"""Evaluate a trained formation-proxy model on the held-out test split."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from phase3b.graphs import FORMATION2ID, build_formation_dataset
from phase3b.metrics import confusion, macro_f1
from phase3b.models import GAT, GCN

ID2FORMATION = {v: k for k, v in FORMATION2ID.items()}


def evaluate(arch="gcn", checkpoint=None):
    items = build_formation_dataset("test")
    checkpoint = checkpoint or f"phase3b/formation_{arch}.pt"
    model = (GCN() if arch == "gcn" else GAT())
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    rows = []
    with torch.no_grad():
        for item in items:
            logits = model(torch.tensor(item["nodes"]), torch.tensor(item["edge"]))
            pred = int(logits.argmax())
            rows.append({
                "seq": item["seq"], "team": item["team"],
                "formation_reference": item["formation"],
                "formation_pred": ID2FORMATION[pred],
                "confidence": round(float(torch.softmax(logits, dim=0)[pred]), 6),
            })
    predictions = pd.DataFrame(rows)
    y = [FORMATION2ID[row["formation_reference"]] for row in rows]
    p = [FORMATION2ID[row["formation_pred"]] for row in rows]
    result = {
        "architecture": arch,
        "test_examples": len(rows),
        "test_macro_f1": round(macro_f1(y, p, n_classes=len(FORMATION2ID)), 6),
        "confusion_rows_reference_cols_prediction": confusion(y, p, len(FORMATION2ID)).tolist(),
        "target": "GT-derived geometric formation proxy",
    }
    predictions.to_csv("phase3b/formation_test_predictions.csv", index=False)
    Path("phase3b/formation_test_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return predictions, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("arch", choices=("gcn", "gat"), nargs="?", default="gcn")
    args = parser.parse_args()
    evaluate(args.arch)
