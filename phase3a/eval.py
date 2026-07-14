import sys, numpy as np, pandas as pd, torch
from phase3a.windows import build_windows
from phase3a.physics import const_velocity, const_acceleration, interp_const_accel
from phase3a.metrics import ade_fde
from phase3a.models import GRUForecaster

COLS = ["ade_1s", "fde_1s", "ade_2s", "fde_2s", "ade_3s", "fde_3s"]

def rows_from_windows(w, model=None, device="cpu"):
    rows = {}
    rows["const_velocity"] = ade_fde(const_velocity(w["x"]), w["gt"])
    rows["const_acceleration"] = ade_fde(const_acceleration(w["x"]), w["gt"])
    if model is not None:
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(w["x"]).to(device)).cpu().numpy()
        rows["gru"] = ade_fde(interp_const_accel(pred, w["x"][:, -1, 2:4]), w["gt"])
    return pd.DataFrame(rows).T[COLS]

def evaluate(test_csv, weights="phase3a/gru.pt"):
    w = build_windows(test_csv)
    model = GRUForecaster(); model.load_state_dict(torch.load(weights, map_location="cpu"))
    df = rows_from_windows(w, model=model)
    print(f"test windows: {len(w['x'])}\n")
    print(df.round(3).to_string())
    df.to_csv("phase3a/phase3a_results.csv")
    return df

if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else "thesis_results/phase2/features_test_smoothed.csv"
    evaluate(csv)