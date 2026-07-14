#!/usr/bin/env python3
"""Bake all clips of a split efficiently: scan the GAT graphs once, reuse per seq."""
import sys, os, glob
from collections import defaultdict
import torch
import pandas as pd

# Support both `python -m dashboard.bake_all` and `python dashboard/bake_all.py`.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from dashboard.prepare import player_frame_table, forecast_table, ID2LABEL, WIN
from phase3b.graphs import build_dataset
from phase3b.models import GAT

def bake_all(split, force=False):
    items = build_dataset(split)                          # scan once
    gat = GAT(); gat.load_state_dict(torch.load("phase3b/gat.pt", map_location="cpu")); gat.eval()
    pred, gt = defaultdict(dict), defaultdict(dict)
    with torch.no_grad():
        for it in items:
            s = int(it["seq"])
            logit = gat(torch.tensor(it["nodes"]), torch.tensor(it["edge"]))
            pred[s][(it["team"], it["window"])] = ID2LABEL[int(logit.argmax())]
            gt[s][(it["team"], it["window"])] = ID2LABEL[it["label"]]
    out_dir = os.path.join("dashboard", "data", split)
    os.makedirs(out_dir, exist_ok=True)
    for s in sorted(pred):
        player_path = os.path.join(out_dir, f"{s:03d}.parquet")
        forecast_path = os.path.join(out_dir, f"{s:03d}_forecast.parquet")
        player_current = os.path.exists(player_path)
        forecast_current = os.path.exists(forecast_path) and {"x0", "y0"}.issubset(
            pd.read_parquet(forecast_path).columns
        )
        if player_current and forecast_current and not force:
            continue
        if force or not player_current:
            t = player_frame_table(s, split)
            t["window"] = (t.frame - 1) // WIN
            key = list(zip(t.team, t.window))
            t["press_pred"] = [pred[s].get(k, "") for k in key]
            t["press_gt"] = [gt[s].get(k, "") for k in key]
            t.to_parquet(player_path)
            print(f"baked player data {split} {s:03d} ({len(t)} rows)")
        if force or not forecast_current:
            forecast_table(s, split).to_parquet(forecast_path)
            print(f"baked forecast data {split} {s:03d}")

if __name__ == "__main__":
    args = sys.argv[1:]
    force = "--force" in args
    splits = [arg for arg in args if arg != "--force"] or ["valid", "test"]
    for sp in splits:
        bake_all(sp, force=force)
    print("done ->", len(glob.glob(os.path.join("dashboard", "data", "*", "*.parquet"))) // 2, "clips")
