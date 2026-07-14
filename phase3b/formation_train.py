"""Train graph models to predict the GT-derived formation proxy labels."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from phase3b.graphs import FORMATION2ID, build_formation_dataset
from phase3b.metrics import macro_f1
from phase3b.models import GAT, GCN
from phase3b.train import split_items

VAL_SEQS = [78, 79, 80, 81, 82, 83]


def _predict(model, items):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for item in items:
            logits = model(torch.tensor(item["nodes"]), torch.tensor(item["edge"]))
            y_true.append(item["label"])
            y_pred.append(int(logits.argmax()))
    return np.asarray(y_true), np.asarray(y_pred)


def train(arch="gcn", epochs=120, lr=1e-3, out=None, val_seqs=VAL_SEQS):
    """Train on validation clips, retaining the epoch with best held-out F1."""
    torch.manual_seed(7)
    np.random.seed(7)
    all_items = build_formation_dataset("valid")
    train_items, holdout_items = split_items(all_items, val_seqs=val_seqs)
    if not train_items or not holdout_items:
        raise RuntimeError("Formation train/hold-out split is empty")

    model = (GCN() if arch == "gcn" else GAT())
    counts = np.bincount([item["label"] for item in train_items], minlength=len(FORMATION2ID))
    weights = torch.tensor(len(train_items) / np.maximum(counts, 1), dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    out = Path(out or f"phase3b/formation_{arch}.pt")
    best = {"holdout_macro_f1": -1.0, "epoch": -1}

    for epoch in range(epochs):
        model.train()
        for index in np.random.permutation(len(train_items)):
            item = train_items[index]
            logits = model(torch.tensor(item["nodes"]), torch.tensor(item["edge"]))
            loss = torch.nn.functional.cross_entropy(
                logits[None], torch.tensor([item["label"]]), weight=weights
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
        y, p = _predict(model, holdout_items)
        score = macro_f1(y, p, n_classes=len(FORMATION2ID))
        if score > best["holdout_macro_f1"]:
            best = {"holdout_macro_f1": round(float(score), 6), "epoch": epoch}
            torch.save(model.state_dict(), out)

    best.update({
        "architecture": arch,
        "train_examples": len(train_items),
        "holdout_examples": len(holdout_items),
        "class_counts_train": dict(zip(FORMATION2ID, counts.tolist())),
        "target": "GT-derived geometric formation proxy",
    })
    metrics_path = out.with_suffix(".json")
    metrics_path.write_text(json.dumps(best, indent=2) + "\n")
    print(json.dumps(best, indent=2))
    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("arch", choices=("gcn", "gat"), nargs="?", default="gcn")
    parser.add_argument("--epochs", type=int, default=120)
    args = parser.parse_args()
    train(args.arch, epochs=args.epochs)
